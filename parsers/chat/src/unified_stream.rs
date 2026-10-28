// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The v2 stream adapter: one `dynamo-parsers-v2` `UnifiedParser` per response
//! choice, mapped onto OpenAI streaming chunks.
//!
//! Moved from dynamo `lib/llm/src/protocols/openai/chat_completions/unified_parser.rs`.
//! The batch path (`parse_complete*`) is not here.
//!
//! # Why one chunk per delta
//!
//! [`ChatCompletionStreamResponseDelta`] carries `content`, `reasoning_content` and
//! `tool_calls` side by side with no way to say which came first. Packing a whole
//! `push` into one delta object would throw away exactly the ordering this path exists
//! to preserve, so every [`UnifiedParserEvent`] becomes its own chunk.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;

use async_stream::stream;
use dynamo_parsers::sources::{ChoiceSources, SourceSlot};
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, UnifiedParser, UnifiedParserEvent, UnifiedParserInit,
    UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode,
    create_unified_parser_for_family,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
    ChatCompletionStreamResponseDelta, FinishReason, FunctionCallStream, FunctionType,
};
use futures::{Stream, StreamExt};
use uuid::Uuid;

use crate::families::{
    DEEPSEEK_V41_UNIFIED_FAMILY, GLM47_UNIFIED_FAMILY, MUSE_UNIFIED_FAMILY, QWEN3_TOOL_CALL_PARSER,
    QWEN3_UNIFIED_FAMILY,
};
use crate::guided::GuidedToolConstraint;
use crate::spans::{FeedMap, Spans, append_spans, spans_len, split_spans};
use crate::stream_item::{ToolCallCompletion, UnifiedStreamChunk, UnifiedStreamItem};

#[cfg(test)]
use dynamo_protocols::types::ChatCompletionToolChoiceOption;

/// Request facts that affect channel interpretation before any bytes are consumed.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnifiedRequestPolicy {
    pub reasoning_disabled: bool,
    pub structured_response: bool,
    /// The operator explicitly selected V2 parsing (`DYN_PARSER_VERSION=2` in
    /// dynamo). It is a deployment fact, not a request fact: dynamo read it from
    /// the environment inside this code, and the caller now passes it.
    pub parsers_v2_selected: bool,
}

impl UnifiedRequestPolicy {
    fn starting_state(
        self,
        family: &str,
        inferred: UnifiedParserStartingState,
    ) -> UnifiedParserStartingState {
        if self.reasoning_disabled && family == "gemma4" {
            // Gemma's Response state preserves literal thought delimiters and still parses tools.
            UnifiedParserStartingState::Response
        } else {
            inferred
        }
    }

    fn inspects_json_prefill(self, constraint: &GuidedToolConstraint) -> bool {
        self.structured_response || constraint.installs_guided_json()
    }
}

/// Scalar and container roots can also begin genuine reasoning. Hold only that
/// ambiguous prefix; process each incoming byte once and release it when a suffix
/// disproves JSON.
#[derive(Default)]
struct JsonPrefillProbe {
    text: String,
    state: JsonPrefixState,
    starts_json: Option<bool>,
    inspected_bytes: usize,
    container_start_validated: bool,
    marker_probe: Option<QuotedControlMarkerProbe>,
    marker_suffix: String,
    marker_decision: Option<UnifiedParserStartingState>,
    marker_inspected_bytes: usize,
}

#[derive(Default)]
enum JsonPrefixState {
    #[default]
    Start,
    ContainerStart {
        opener: u8,
        validated: bool,
    },
    String {
        escaped: bool,
        unicode_left: u8,
    },
    Literal {
        expected: &'static [u8],
        position: usize,
    },
    Number(NumberPrefixState),
    Done,
    Invalid,
}

#[derive(Clone, Copy)]
enum NumberPrefixState {
    Minus,
    Zero,
    Digits,
    Dot,
    Fraction,
    Exponent,
    ExponentSign,
    ExponentDigits,
}

impl JsonPrefillProbe {
    fn push(&mut self, text: &str) {
        self.text.push_str(text);
        for byte in text.bytes() {
            self.inspected_bytes += 1;
            let next_state = match &self.state {
                JsonPrefixState::Start => match byte {
                    b' ' | b'\r' | b'\n' | b'\t' => JsonPrefixState::Start,
                    b'{' | b'[' => JsonPrefixState::ContainerStart {
                        opener: byte,
                        validated: false,
                    },
                    b'"' => JsonPrefixState::String {
                        escaped: false,
                        unicode_left: 0,
                    },
                    b'-' => JsonPrefixState::Number(NumberPrefixState::Minus),
                    b'0' => JsonPrefixState::Number(NumberPrefixState::Zero),
                    b'1'..=b'9' => JsonPrefixState::Number(NumberPrefixState::Digits),
                    b't' => JsonPrefixState::Literal {
                        expected: b"true",
                        position: 1,
                    },
                    b'f' => JsonPrefixState::Literal {
                        expected: b"false",
                        position: 1,
                    },
                    b'n' => JsonPrefixState::Literal {
                        expected: b"null",
                        position: 1,
                    },
                    _ => JsonPrefixState::Invalid,
                },
                JsonPrefixState::String {
                    escaped,
                    unicode_left,
                } => {
                    if *unicode_left > 0 {
                        if byte.is_ascii_hexdigit() {
                            JsonPrefixState::String {
                                escaped: false,
                                unicode_left: unicode_left - 1,
                            }
                        } else {
                            JsonPrefixState::Invalid
                        }
                    } else if *escaped {
                        match byte {
                            b'u' => JsonPrefixState::String {
                                escaped: false,
                                unicode_left: 4,
                            },
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                                JsonPrefixState::String {
                                    escaped: false,
                                    unicode_left: 0,
                                }
                            }
                            _ => JsonPrefixState::Invalid,
                        }
                    } else {
                        match byte {
                            b'"' => JsonPrefixState::Done,
                            b'\\' => JsonPrefixState::String {
                                escaped: true,
                                unicode_left: 0,
                            },
                            0..=31 => JsonPrefixState::Invalid,
                            _ => JsonPrefixState::String {
                                escaped: false,
                                unicode_left: 0,
                            },
                        }
                    }
                }
                JsonPrefixState::Literal { expected, position } => {
                    if byte != expected[*position] {
                        JsonPrefixState::Invalid
                    } else if position + 1 == expected.len() {
                        JsonPrefixState::Done
                    } else {
                        JsonPrefixState::Literal {
                            expected,
                            position: position + 1,
                        }
                    }
                }
                JsonPrefixState::Number(number) => {
                    use NumberPrefixState::*;
                    let next = match (*number, byte) {
                        (Minus, b'0') => Some(Zero),
                        (Minus, b'1'..=b'9') => Some(Digits),
                        (Digits, b'0'..=b'9') => Some(Digits),
                        (Zero | Digits, b'.') => Some(Dot),
                        (Dot | Fraction, b'0'..=b'9') => Some(Fraction),
                        (Zero | Digits | Fraction, b'e' | b'E') => Some(Exponent),
                        (Exponent, b'+' | b'-') => Some(ExponentSign),
                        (Exponent | ExponentSign | ExponentDigits, b'0'..=b'9') => {
                            Some(ExponentDigits)
                        }
                        _ => None,
                    };
                    if let Some(next) = next {
                        JsonPrefixState::Number(next)
                    } else if matches!(number, Zero | Digits | Fraction | ExponentDigits)
                        && byte.is_ascii_whitespace()
                    {
                        JsonPrefixState::Done
                    } else {
                        JsonPrefixState::Invalid
                    }
                }
                JsonPrefixState::Done => {
                    if matches!(byte, b' ' | b'\r' | b'\n' | b'\t') {
                        JsonPrefixState::Done
                    } else {
                        JsonPrefixState::Invalid
                    }
                }
                JsonPrefixState::ContainerStart { opener, validated } => {
                    if *validated {
                        JsonPrefixState::ContainerStart {
                            opener: *opener,
                            validated: true,
                        }
                    } else if matches!(byte, b' ' | b'\r' | b'\n' | b'\t') {
                        JsonPrefixState::ContainerStart {
                            opener: *opener,
                            validated: false,
                        }
                    } else {
                        let valid_first_value = matches!(
                            (*opener, byte),
                            (b'{', b'"' | b'}')
                                | (
                                    b'[',
                                    b']' | b'{' | b'[' | b'"' | b'-' | b'0'
                                        ..=b'9' | b't' | b'f' | b'n',
                                )
                        );
                        if valid_first_value {
                            JsonPrefixState::ContainerStart {
                                opener: *opener,
                                validated: true,
                            }
                        } else {
                            JsonPrefixState::Invalid
                        }
                    }
                }
                JsonPrefixState::Invalid => JsonPrefixState::Invalid,
            };
            if self.starts_json.is_none() && !matches!(next_state, JsonPrefixState::Start) {
                self.starts_json = Some(!matches!(next_state, JsonPrefixState::Invalid));
            }
            if matches!(
                &next_state,
                JsonPrefixState::ContainerStart {
                    validated: true,
                    ..
                }
            ) {
                self.container_start_validated = true;
            }
            self.state = next_state;
        }
    }

    fn unopened_channel_decision(&self, terminal: bool) -> Option<UnifiedParserStartingState> {
        // JSON roots are unambiguous when the prompt has not opened reasoning;
        // waiting for a complete scalar would unnecessarily buffer its stream.
        match self.starts_json {
            Some(true) => Some(UnifiedParserStartingState::Response),
            Some(false) => Some(UnifiedParserStartingState::None),
            None if terminal => Some(UnifiedParserStartingState::Response),
            None => None,
        }
    }

    fn decision(&self, terminal: bool) -> Option<UnifiedParserStartingState> {
        if self.container_start_validated {
            return Some(UnifiedParserStartingState::None);
        }
        match self.state {
            JsonPrefixState::ContainerStart {
                validated: true, ..
            } if terminal => Some(UnifiedParserStartingState::None),
            JsonPrefixState::ContainerStart {
                validated: true, ..
            } => None,
            JsonPrefixState::ContainerStart { .. } if terminal => {
                Some(UnifiedParserStartingState::None)
            }
            JsonPrefixState::ContainerStart { .. } => None,
            JsonPrefixState::Invalid => Some(UnifiedParserStartingState::Reasoning),
            JsonPrefixState::Done
            | JsonPrefixState::Number(
                NumberPrefixState::Zero
                | NumberPrefixState::Digits
                | NumberPrefixState::Fraction
                | NumberPrefixState::ExponentDigits,
            ) if terminal => Some(UnifiedParserStartingState::None),
            _ if terminal => Some(UnifiedParserStartingState::Reasoning),
            _ => None,
        }
    }

    fn decision_for_family(
        &mut self,
        family: &str,
        terminal: bool,
        latest_text: &str,
    ) -> anyhow::Result<Option<UnifiedParserStartingState>> {
        if let Some(decision) = self.marker_decision {
            return Ok(Some(decision));
        }
        let first_push = self.marker_probe.is_none();
        let shield = self
            .marker_probe
            .get_or_insert_with(|| QuotedControlMarkerProbe::new(family));
        let incoming = if first_push { &self.text } else { latest_text };
        let mut unquoted = shield.push(incoming);
        if terminal {
            unquoted.push_str(&shield.finish());
        }
        self.marker_suffix.push_str(&unquoted);
        self.marker_inspected_bytes += self.marker_suffix.len();
        let has_candidate = shield
            .markers
            .iter()
            .any(|marker| self.marker_suffix.contains(marker));
        let suffix_limit = shield.markers.iter().map(String::len).max().unwrap_or(0);
        let mut keep_from = self.marker_suffix.len().saturating_sub(suffix_limit);
        while !self.marker_suffix.is_char_boundary(keep_from) {
            keep_from += 1;
        }
        self.marker_suffix.drain(..keep_from);

        // Only complete unquoted controls require raw-prefix classification;
        // rescanning growing JSON for every punctuation-bearing chunk is quadratic.
        if has_candidate && matches!(self.state, JsonPrefixState::ContainerStart { .. }) {
            self.marker_inspected_bytes += self.text.len();
            if let Some(state) = reasoning_marker_prefill(family, &self.text)? {
                self.marker_decision = Some(state);
                return Ok(Some(state));
            }
        }
        Ok(self.decision(terminal))
    }
}

fn structured_json_prefill(
    family: &str,
    content: &str,
    terminal: bool,
) -> anyhow::Result<Option<UnifiedParserStartingState>> {
    let mut probe = JsonPrefillProbe::default();
    probe.push(content);
    probe.decision_for_family(family, terminal, content)
}

fn structured_json_response_prefill(
    family: &str,
    content: &str,
    terminal: bool,
) -> anyhow::Result<Option<UnifiedParserStartingState>> {
    if let Some(state) = reasoning_marker_prefill(family, content)? {
        return Ok(Some(state));
    }
    let json_state = structured_json_prefill(family, content, terminal)?;
    if json_state == Some(UnifiedParserStartingState::None) {
        return Ok(Some(UnifiedParserStartingState::Response));
    }
    Ok(json_state)
}

fn reasoning_marker_prefill(
    family: &str,
    content: &str,
) -> anyhow::Result<Option<UnifiedParserStartingState>> {
    let has_marker = if let Some((opener, closer)) = reasoning_marker_pair(family) {
        contains_unquoted_marker(content, opener) || contains_unquoted_marker(content, closer)
    } else {
        match family {
            "kimi_k3" => {
                kimi_channel_marker_position(content, "<|open|>", "think").is_some()
                    || kimi_channel_marker_position(content, "<|close|>", "think").is_some()
            }
            "muse_glimmer" => {
                contains_unquoted_marker(content, "<|start|>")
                    || contains_unquoted_marker(content, "<|eom|>")
            }
            _ => false,
        }
    };
    has_marker
        .then(|| detect_prefill(family, content))
        .transpose()
}

fn starts_bare_json(content: &str) -> bool {
    matches!(content.trim_start().as_bytes().first(), Some(b'[' | b'{'))
}

/// Which channel the rendered generation prompt already opened, for the streaming path.
///
/// `prompt_injected_reasoning` is the per-request fact the preprocessor already
/// computes: the rendered prompt ended with the family's reasoning opener, so generated
/// output starts inside a thought the model will close without ever emitting the
/// opener. Absent that, the answer is per family.
#[derive(Clone, Copy)]
pub struct StreamPrefill {
    pub starting_state: UnifiedParserStartingState,
    pub prefix: Option<&'static str>,
}

impl From<UnifiedParserStartingState> for StreamPrefill {
    fn from(starting_state: UnifiedParserStartingState) -> Self {
        Self {
            starting_state,
            prefix: None,
        }
    }
}

pub fn stream_prefill(family: &str, prompt_injected_reasoning: bool) -> UnifiedParserStartingState {
    if prompt_injected_reasoning {
        return UnifiedParserStartingState::Reasoning;
    }
    match family {
        // Qwen3's generation prompt ends at the assistant header with no channel open,
        // so the model emits `<think>` itself when it thinks.
        QWEN3_UNIFIED_FAMILY => UnifiedParserStartingState::None,
        DEEPSEEK_V41_UNIFIED_FAMILY => UnifiedParserStartingState::None,
        _ => UnifiedParserStartingState::None,
    }
}

/// Resolve a `Reasoning`-prefill choice's ACTUAL starting state when guided JSON is
/// installed, from the shape of that choice's first content-bearing chunk.
///
/// Guided decoding forbids the model from emitting the reasoning closer, so a request
/// whose rendered prompt already opened reasoning (`prompt_injected_reasoning`) would,
/// if started unconditionally at `Reasoning`, have its ENTIRE guided JSON payload
/// swallowed as `reasoning_content` with zero tool calls ever extracted — the parser
/// waits forever for a closer guided decoding will never produce. But committing to
/// `None` unconditionally instead is equally wrong: a turn that genuinely reasons before
/// its guided JSON (`reasoning</think>JSON`) would then be misread as pure content, and
/// the legitimate `reasoning_content`/`content` split would be lost.
///
/// Mirrors `bypass_bare_guided_json`'s shape check in
/// `Preprocessor::parse_reasoning_content_from_stream_inner` (preprocessor.rs), which
/// makes the identical call for the non-unified reasoning-parser path: a payload whose
/// first non-whitespace byte is `[` or `{` is bare guided JSON with no reasoning to
/// split out (start at `None`, so the whole payload reaches the tool/response path
/// untouched); anything else is ordinary text, meaning this really is a reasoning block
/// that will close normally (start at `Reasoning`).
///
/// Nonterminal empty and whitespace-only chunks are held by `apply_stream` until this
/// check can classify the first non-whitespace content. A terminal empty stream keeps
/// the prompt's `Reasoning` state because there is no later payload to classify.
fn bare_guided_json_prefill(
    first_content: Option<&ChatCompletionMessageContent>,
) -> UnifiedParserStartingState {
    let Some(ChatCompletionMessageContent::Text(text)) = first_content else {
        return UnifiedParserStartingState::Reasoning;
    };
    let trimmed = text.trim_start();
    if trimmed.is_empty() {
        return UnifiedParserStartingState::Reasoning;
    }
    if starts_bare_json(trimmed) {
        UnifiedParserStartingState::None
    } else {
        UnifiedParserStartingState::Reasoning
    }
}

/// Which channel the prompt opened, inferred from complete output text.
///
/// The batch path has no prompt in hand, only what the model produced, so the channel
/// state is read back off the text: a `<think>` opener means the model opened reasoning
/// itself; a bare `</think>` with no opener means the prompt had already opened it; and
/// neither marker means reasoning never ran for this turn.
fn detect_prefill(family: &str, content: &str) -> anyhow::Result<UnifiedParserStartingState> {
    match family {
        "qwen3" | "deepseek_v4" | "deepseek_v41" | "glm47" | "kimi_k2" => {
            let (opener_marker, closer_marker) = reasoning_marker_pair(family)
                .expect("think-marker families have a configured reasoning pair");
            // Compare FIRST-occurrence positions, not mere presence: a prompt that
            // pre-opened reasoning produces a leading `</think>` with no opener before
            // it, but a later `<think>...</think>` pair from the model can still follow
            // in the same output (e.g. after a tool-call gap). Testing "does an opener
            // exist anywhere" before "does a closer exist anywhere" would misclassify
            // that case as `None` instead of `Reasoning`.
            let opener = first_unquoted_marker_position(content, opener_marker);
            let closer = first_unquoted_marker_position(content, closer_marker);
            Ok(match (opener, closer) {
                // No opener before it (or no opener at all): the prompt opened reasoning.
                (None, Some(_)) => UnifiedParserStartingState::Reasoning,
                (Some(open_at), Some(close_at)) if close_at < open_at => {
                    UnifiedParserStartingState::Reasoning
                }
                // An opener exists (and any closer does not precede it): the model will
                // open reasoning itself later, currently visible as ordinary text.
                (Some(_), _) => UnifiedParserStartingState::None,
                (None, None) => UnifiedParserStartingState::Response,
            })
        }
        "gemma4" => {
            let (opener, closer) = reasoning_marker_pair(family)
                .expect("Gemma 4 has a configured reasoning marker pair");
            Ok(infer_marker_prefill(content, opener, closer))
        }
        "kimi_k3" => Ok(
            match (
                kimi_channel_marker_position(content, "<|open|>", "think"),
                kimi_channel_marker_position(content, "<|close|>", "think"),
            ) {
                (None, Some(_)) => UnifiedParserStartingState::Reasoning,
                (Some(open), Some(close)) if close < open => UnifiedParserStartingState::Reasoning,
                _ => UnifiedParserStartingState::None,
            },
        ),
        "gpt_oss" => Ok(UnifiedParserStartingState::None),
        "muse_glimmer" => Ok(infer_marker_prefill(content, "<|start|>", "<|eom|>")),
        other => anyhow::bail!("no prefill detector for unified parser family '{other}'"),
    }
}

fn reasoning_marker_pair(family: &str) -> Option<(&'static str, &'static str)> {
    match family {
        "qwen3" | "deepseek_v4" | "deepseek_v41" | "glm47" | "kimi_k2" => {
            Some(("<think>", "</think>"))
        }
        "gemma4" => Some(("<|channel>", "<channel|>")),
        _ => None,
    }
}

fn kimi_channel_marker_position(content: &str, prefix: &str, channel: &str) -> Option<usize> {
    content
        .match_indices(prefix)
        .filter_map(|(offset, _)| {
            let after = offset + prefix.len();
            let (end, suffix_len) = if prefix == "<|close|>" {
                let end = after
                    + content[after..]
                        .find(['<', '[', '{'])
                        .unwrap_or(content[after..].len());
                let suffix_len = if content[end..].starts_with("<|sep|>") {
                    "<|sep|>".len()
                } else {
                    0
                };
                (end, suffix_len)
            } else {
                (after + content[after..].find("<|sep|>")?, "<|sep|>".len())
            };
            if content[after..end].split_whitespace().next() != Some(channel) {
                return None;
            }
            let marker = &content[offset..end + suffix_len];
            first_unquoted_marker_position(content, marker)
        })
        .min()
}

fn infer_marker_prefill(content: &str, opener: &str, closer: &str) -> UnifiedParserStartingState {
    match (
        first_unquoted_marker_position(content, opener),
        first_unquoted_marker_position(content, closer),
    ) {
        (None, Some(_)) => UnifiedParserStartingState::Reasoning,
        (Some(open), Some(close)) if close < open => UnifiedParserStartingState::Reasoning,
        _ => UnifiedParserStartingState::None,
    }
}

/// Marker-looking text quoted as prose is visible content, not parser control syntax.
pub(crate) fn contains_unquoted_marker(content: &str, marker: &str) -> bool {
    first_unquoted_marker_position(content, marker).is_some()
}

/// The byte offset of the first unquoted occurrence of `marker` in `content`, or `None`
/// if it never appears outside a quoted span. Prefill detection compares positions
/// so a later opener cannot hide an earlier prompt-prefilled closer.
pub(crate) fn first_unquoted_marker_position(content: &str, marker: &str) -> Option<usize> {
    first_unquoted_marker_position_after(content, marker, 0)
}

fn first_unquoted_marker_position_after(
    content: &str,
    marker: &str,
    minimum: usize,
) -> Option<usize> {
    let chars: Vec<(usize, char)> = content.char_indices().collect();
    // Whether an unescaped `"`, `'`, or `` ` `` appears anywhere at or after each
    // position, precomputed once in O(n) so the scan below doesn't rescan the
    // remainder of the string for every candidate quote character (which would be
    // O(n^2) on adversarial text with many non-closing quote-like characters).
    let closes_later = later_unescaped_quote_closes(&chars);

    let mut quote = None;
    let mut escaped = false;
    for (position, &(offset, character)) in chars.iter().enumerate() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                let previous = content[..offset].chars().next_back();
                let next = content[offset + character.len_utf8()..].chars().next();
                if active_quote == '\'' && is_internal_apostrophe(previous, next) {
                    continue;
                }
                quote = None;
            }
            continue;
        }
        let previous = content[..offset].chars().next_back();
        let next = content[offset + character.len_utf8()..].chars().next();
        let apostrophe = character == '\''
            && (previous.is_some_and(char::is_alphanumeric)
                || next.is_some_and(char::is_alphanumeric) && !closes_later[position + 1][1]);
        // A quote that never closes before end-of-string was never really quoting
        // anything; treating it as an open span would blind the rest of the scan to
        // a real marker that follows (e.g. a stray `"` before a genuine `</think>`).
        if let Some(slot) = quote_slot(character)
            && !apostrophe
            && closes_later[position + 1][slot]
        {
            quote = Some(character);
            continue;
        }
        if offset >= minimum && content[offset..].starts_with(marker) {
            return Some(offset);
        }
    }
    None
}

/// Return the first unquoted native tool-call marker configured for `parser`.
///
/// Native markers quoted as prose remain visible text in both streaming and
/// aggregated responses.
pub fn first_unquoted_native_tool_call_marker(content: &str, parser: &str) -> Option<usize> {
    native_tool_call_start_tokens(parser)?
        .iter()
        .filter_map(|marker| first_unquoted_marker_position(content, marker))
        .min()
}

/// Like [`first_unquoted_native_tool_call_marker`], but only for markers that cannot be
/// ordinary prose.
///
/// Truncating a response at a marker is safe only when the marker is control syntax. Some
/// parsers open a call with a plain word — `phi4` uses `functools` — and an answer about
/// the Python module of that name would lose everything from the word onward. A marker
/// carrying a punctuation character (`<`, `|`, `[`) cannot be written by accident.
pub fn first_unquoted_structural_tool_call_marker(content: &str, parser: &str) -> Option<usize> {
    native_tool_call_start_tokens(parser)?
        .iter()
        .filter(|marker| !marker.chars().all(char::is_alphanumeric))
        .filter_map(|marker| first_unquoted_marker_position(content, marker))
        .min()
}

/// Return the start of an unquoted native marker, including an unquoted
/// partial marker suffix that must remain buffered for the next stream chunk.
pub fn unquoted_native_tool_call_marker_or_prefix_start(
    content: &str,
    parser: &str,
) -> Option<usize> {
    let markers = native_tool_call_start_tokens(parser)?;
    if let Some(marker_start) = markers
        .iter()
        .filter_map(|marker| first_unquoted_marker_position(content, marker))
        .min()
    {
        return Some(marker_start);
    }

    markers
        .iter()
        .flat_map(|marker| {
            marker
                .char_indices()
                .skip(1)
                .map(move |(end, _)| &marker[..end])
        })
        .filter(|prefix| {
            let before = content.strip_suffix(prefix).unwrap_or(content);
            first_unquoted_marker_position(before, prefix).is_none()
        })
        .filter_map(|prefix| {
            let before = content.strip_suffix(prefix)?;
            let position = first_unquoted_marker_position(content, prefix)?;
            (position == before.len()).then_some(position)
        })
        .min()
}

fn native_tool_call_start_tokens(parser: &str) -> Option<Vec<String>> {
    let parser_key = if parser.is_empty() { "default" } else { parser };
    Some(
        dynamo_parsers::tool_calling::parsers::get_tool_parser_map()
            .get(parser_key)?
            .parser_config
            .tool_call_start_tokens()
            .iter()
            .filter(|marker| !marker.is_empty())
            .cloned()
            .collect(),
    )
}

/// Index into the fixed-size "which quote characters close later" arrays used by
/// [`later_unescaped_quote_closes`].
fn quote_slot(character: char) -> Option<usize> {
    match character {
        '"' => Some(0),
        '\'' => Some(1),
        '`' => Some(2),
        _ => None,
    }
}

fn is_internal_apostrophe(previous: Option<char>, next: Option<char>) -> bool {
    previous.is_some_and(char::is_alphanumeric) && next.is_some_and(char::is_alphanumeric)
}

/// For each position in `chars`, whether an unescaped `"`, `'`, or `` ` `` appears
/// anywhere later in the string — `result[i][quote_slot(c)]` answers "does an
/// unescaped `c` exist in `chars[i..]`". Built in one right-to-left O(n) pass: the
/// answer at `i` depends only on `chars[i]` and the answer at `i + 1` (or `i + 2` for
/// a backslash, which escapes exactly the next character), so this is an exact
/// backward re-expression of the left-to-right escaped-flag scan that used to be
/// re-run from scratch for every candidate quote character.
fn later_unescaped_quote_closes(chars: &[(usize, char)]) -> Vec<[bool; 3]> {
    let n = chars.len();
    let mut closes = vec![[false; 3]; n + 1];
    let mut i = n;
    while i > 0 {
        i -= 1;
        let character = chars[i].1;
        closes[i] = if character == '\\' {
            closes.get(i + 2).copied().unwrap_or([false; 3])
        } else {
            let mut next = closes[i + 1];
            let internal_apostrophe = character == '\''
                && is_internal_apostrophe(
                    i.checked_sub(1)
                        .and_then(|previous| chars.get(previous).map(|(_, c)| *c)),
                    chars.get(i + 1).map(|(_, c)| *c),
                );
            if let Some(slot) = quote_slot(character)
                && !internal_apostrophe
            {
                next[slot] = true;
            }
            next
        };
    }
    closes
}

/// Map dynamo's v1 [`ToolDefinition`]s onto the v2 parser's [`Tool`] shape.
pub fn to_v2_tools(tools: Option<&[ToolDefinition]>) -> Vec<Tool> {
    tools
        .unwrap_or(&[])
        .iter()
        .map(|tool| Tool {
            name: tool.name.clone(),
            description: None,
            parameters: tool.parameters.clone().unwrap_or(serde_json::Value::Null),
            strict: tool.strict,
        })
        .collect()
}

/// Map the request's `tool_choice` onto the wire format the backend will produce.
///
/// A named or `required` choice is served by guided decoding, which constrains the
/// model to bare JSON instead of Qwen's native `<tool_call>` XML: a named choice to
/// that one tool's argument object, `required` to a call object or an array of them.
/// Unset / `auto` / `none` leave the model in native markup.
///
/// Callers must not consult this when a structural tag is active — see
/// [`apply_stream`]'s `uses_tool_call_structural_tag`.
fn tool_output_mode(constraint: &GuidedToolConstraint) -> UnifiedToolOutputMode {
    match constraint {
        GuidedToolConstraint::GuidedJsonNamed { tool_name } => UnifiedToolOutputMode::GuidedJson {
            named_tool: Some(tool_name.clone()),
        },
        GuidedToolConstraint::GuidedJsonRequired => {
            UnifiedToolOutputMode::GuidedJson { named_tool: None }
        }
        GuidedToolConstraint::None | GuidedToolConstraint::StructuralTag => {
            UnifiedToolOutputMode::Native
        }
    }
}

/// The malformed-guided-payload contract for one request, selected by the
/// guided tool-call streaming rollback lever.
///
/// Only [`InvalidGuidedPayloadPolicy::StreamBestEffort`] may emit a call before its
/// payload closes; the other two contracts buffer to completion (see the policy's
/// own doc table). So this IS the streaming decision on the unified path, the same
/// way `guided_streaming` is on the jail path — the caller passes down one already-
/// made decision rather than re-deriving it here.
///
/// Rolling back picks `RecoverAsText`, not `Reject`: the lever exists to stop early
/// emission, not to convert a malformed payload from recovered text into a typed
/// error, which would be a second, louder behaviour change riding along with it.
fn invalid_guided_payload_policy(guided_streaming: bool) -> InvalidGuidedPayloadPolicy {
    if guided_streaming {
        InvalidGuidedPayloadPolicy::StreamBestEffort
    } else {
        InvalidGuidedPayloadPolicy::RecoverAsText
    }
}

/// Ask the grammar owner which source bytes remain uncommitted. A literal tool
/// closer inside an argument must not hide an unfinished sibling from recovery.
fn glm_unfinished_call(input: &str) -> anyhow::Result<Option<usize>> {
    let mut parser = create_unified_parser_for_family(GLM47_UNIFIED_FAMILY, &[])?;
    let mut output = UnifiedParserOutput::default();
    parser.parse_into(input, &mut output)?;
    let pending = parser.reset();
    let prefix = input
        .strip_suffix(&pending)
        .ok_or_else(|| anyhow::anyhow!("GLM pending input is not a source suffix"))?;
    // Only prose after the last completed call can quote a pending marker.
    // Argument quotes belong to the native grammar and cannot span sibling calls.
    let mut prose = String::new();
    for event in output.events {
        match event {
            UnifiedParserEvent::ToolCall(call) if call.complete => prose.clear(),
            UnifiedParserEvent::Text(text) | UnifiedParserEvent::Reasoning(text) => {
                prose.push_str(&text)
            }
            _ => {}
        }
    }
    let pending_offset = prose.len();
    prose.push_str(&pending);
    let markers = native_tool_call_start_tokens(GLM47_UNIFIED_FAMILY)
        .expect("registered GLM native grammar has start tokens");
    Ok(markers
        .iter()
        .filter_map(|marker| first_unquoted_marker_position_after(&prose, marker, pending_offset))
        .min()
        .map(|start| prefix.len() + start - pending_offset))
}

fn unified_parser_control_markers(family: &str) -> Vec<String> {
    let mut markers = reasoning_marker_pair(family)
        .map(|(opener, closer)| vec![opener.to_string(), closer.to_string()])
        .unwrap_or_default();
    if family == "kimi_k3" {
        for channel in ["think", "response", "tools", "call"] {
            markers.push(format!("<|open|>{channel}"));
            markers.push(format!("<|close|>{channel}"));
        }
        markers.push("<|sep|>".to_string());
    }
    if family == "muse_glimmer" {
        markers.extend(
            ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]
                .into_iter()
                .map(str::to_string),
        );
    }
    if let Some(marker) = native_family_marker(family) {
        markers.push(marker.to_string());
    }
    if let Some(native_markers) = native_tool_call_start_tokens(family) {
        markers.extend(native_markers);
    }
    markers.sort_by_key(|marker| std::cmp::Reverse(marker.len()));
    markers.dedup();
    markers
}

fn native_family_marker(family: &str) -> Option<&'static str> {
    match family {
        "qwen3" | "glm47" => Some("<tool_call>"),
        "deepseek_v4" => Some("<｜DSML｜tool_calls>"),
        "deepseek_v41" => Some("<｜DSML｜ calls>"),
        "gemma4" => Some("<|tool_call>"),
        "kimi_k2" => Some("<|tool_call_begin|>"),
        "kimi_k3" => Some("<|open|>call"),
        "muse_glimmer" => Some("<atem:invoke"),
        _ => None,
    }
}

/// Merge adjacent same-kind text/reasoning deltas so one `push` does not become three
/// chunks that say the same thing.
///
/// Tool-call deltas never merge: two fragments belonging to different `tool_index`es
/// would fuse into one call, and even two fragments of the SAME call must keep their
/// `name`-carrying first delta distinct from later argument-only ones.
#[cfg(test)]
fn coalesce(deltas: Vec<UnifiedParserEvent>) -> Vec<UnifiedParserEvent> {
    coalesce_deltas(Deltas {
        events: deltas,
        spans: None,
    })
    .events
}

/// Join adjacent text runs and adjacent reasoning runs, with their ranges.
fn coalesce_deltas(deltas: Deltas) -> Deltas {
    let mut out = Deltas {
        events: Vec::with_capacity(deltas.events.len()),
        spans: deltas.spans.as_ref().map(|_| Vec::new()),
    };
    let mut spans = deltas.spans.map(Vec::into_iter);
    for delta in deltas.events {
        let delta_spans = spans.as_mut().and_then(Iterator::next).unwrap_or_default();
        match (out.events.last_mut(), delta) {
            (Some(UnifiedParserEvent::Text(prev)), UnifiedParserEvent::Text(text))
            | (Some(UnifiedParserEvent::Reasoning(prev)), UnifiedParserEvent::Reasoning(text)) => {
                prev.push_str(&text);
                if let Some(last) = out.spans.as_mut().and_then(|all| all.last_mut()) {
                    append_spans(last, delta_spans);
                }
            }
            (_, delta) => out.push(delta, delta_spans),
        }
    }
    out
}

/// Pick the one fanout child that keeps the source chunk's token metrics.
///
/// The reasoning-usage estimator runs after unified parsing and attributes the source
/// chunk's tokens to reasoning when any emitted child carries `reasoning_content`.
/// Keeping metrics on the last child unconditionally loses that classification whenever
/// reasoning is followed by visible text, a tool call, or a later response choice.
pub(crate) fn fanout_llm_metrics_position(choices: &[ChatChoiceStream]) -> Option<usize> {
    choices
        .iter()
        .position(|choice| choice.delta.reasoning_content.is_some())
        .or_else(|| choices.len().checked_sub(1))
}

/// An empty streaming choice for `index`, used as the base every emitted chunk is
/// filled in from.
///
/// `logprobs` is dropped on purpose: once parsing rewrites a choice, the emitted text
/// no longer lines up token-for-token with the backend's raw stream, so per-token
/// logprobs would be attached to the wrong characters.
pub(crate) fn empty_choice(index: u32) -> ChatChoiceStream {
    #[allow(deprecated)]
    ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role: None,
            content: None,
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    }
}

/// Per-choice streaming state: one parser instance plus the bookkeeping the OpenAI
/// streaming tool-call contract needs. One instance parses exactly one choice of one
/// request, which is what gives per-stream isolation by construction.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum QwenNativeLexicalMode {
    #[default]
    OutsideBlock,
    InsideBlock,
    ParameterHeader,
    ParameterValue,
}

#[derive(Clone, Default)]
struct QwenNativeLexicalState {
    mode: QwenNativeLexicalMode,
    marker_tail: String,
}

impl QwenNativeLexicalState {
    fn observe(&mut self, text: &str) {
        const BLOCK_START: &str = "<tool_call>";
        const BLOCK_END: &str = "</tool_call>";
        const PARAMETER_START: &str = "<parameter=";
        const PARAMETER_END: &str = "</parameter>";
        const FUNCTION_END: &str = "</function>";

        let mut input = std::mem::take(&mut self.marker_tail);
        input.push_str(text);
        loop {
            match self.mode {
                QwenNativeLexicalMode::OutsideBlock => {
                    if let Some(start) = input.find(BLOCK_START) {
                        input.drain(..start + BLOCK_START.len());
                        self.mode = QwenNativeLexicalMode::InsideBlock;
                    } else {
                        self.retain_partial_suffix(&input, &[BLOCK_START]);
                        return;
                    }
                }
                QwenNativeLexicalMode::InsideBlock => {
                    let parameter = input.find(PARAMETER_START);
                    let block_end = input.find(BLOCK_END);
                    match (parameter, block_end) {
                        (Some(parameter), Some(block_end)) if block_end < parameter => {
                            input.drain(..block_end + BLOCK_END.len());
                            self.mode = QwenNativeLexicalMode::OutsideBlock;
                        }
                        (Some(parameter), _) => {
                            input.drain(..parameter + PARAMETER_START.len());
                            self.mode = QwenNativeLexicalMode::ParameterHeader;
                        }
                        (None, Some(block_end)) => {
                            input.drain(..block_end + BLOCK_END.len());
                            self.mode = QwenNativeLexicalMode::OutsideBlock;
                        }
                        (None, None) => {
                            self.retain_partial_suffix(&input, &[PARAMETER_START, BLOCK_END]);
                            return;
                        }
                    }
                }
                QwenNativeLexicalMode::ParameterHeader => {
                    let header_end = input.find('>');
                    if let Some(function_end) = input.find(FUNCTION_END)
                        && header_end.is_none_or(|end| function_end < end)
                    {
                        input.drain(..function_end + FUNCTION_END.len());
                        self.mode = QwenNativeLexicalMode::InsideBlock;
                    } else if let Some(header_end) = header_end {
                        input.drain(..header_end + 1);
                        self.mode = QwenNativeLexicalMode::ParameterValue;
                    } else {
                        self.retain_partial_suffix(&input, &[FUNCTION_END]);
                        return;
                    }
                }
                QwenNativeLexicalMode::ParameterValue => {
                    // Native parameter values own literal function/block markers.
                    // Only their own delimiter permits a detour to start another call.
                    if let Some(at) = input.find(PARAMETER_END) {
                        input.drain(..at + PARAMETER_END.len());
                        self.mode = QwenNativeLexicalMode::InsideBlock;
                    } else {
                        self.retain_partial_suffix(&input, &[PARAMETER_END]);
                        return;
                    }
                }
            }
        }
    }

    fn retain_partial_suffix(&mut self, input: &str, markers: &[&str]) {
        let suffix_length = markers
            .iter()
            .flat_map(|marker| (1..marker.len()).map(move |length| &marker[..length]))
            .filter(|prefix| input.ends_with(prefix))
            .map(str::len)
            .max()
            .unwrap_or_default();
        self.marker_tail = input[input.len() - suffix_length..].to_string();
    }
}

fn qwen_opener_resolution(held: &str, text: &str) -> Option<(usize, bool)> {
    const BLOCK_START: &str = "<tool_call>";
    for (start, character) in text.char_indices() {
        let end = start + character.len_utf8();
        let candidate = format!("{held}{}", &text[..end]);
        if candidate == BLOCK_START {
            return Some((end, true));
        }
        if !BLOCK_START.starts_with(&candidate) {
            return Some((end, false));
        }
    }
    None
}

fn qwen_partial_opener_suffix_start(
    lexical_state: &QwenNativeLexicalState,
    text: &str,
) -> Option<usize> {
    const BLOCK_START: &str = "<tool_call>";
    let start = (1..BLOCK_START.len())
        .rev()
        .find(|length| text.ends_with(&BLOCK_START[..*length]))
        .map(|length| text.len() - length)?;
    let mut before_candidate = lexical_state.clone();
    before_candidate.observe(&text[..start]);
    (before_candidate.mode == QwenNativeLexicalMode::OutsideBlock).then_some(start)
}

#[derive(Default)]
struct QuotedControlMarkerProbe {
    markers: Vec<String>,
    quote: Option<char>,
    escaped: bool,
    pending_quote_candidate: bool,
    previous_character: Option<char>,
    pending: String,
    pending_marker: Option<String>,
    inspected_bytes: usize,
}

impl QuotedControlMarkerProbe {
    fn new(family: &str) -> Self {
        Self {
            markers: unified_parser_control_markers(family),
            ..Default::default()
        }
    }

    fn push(&mut self, text: &str) -> String {
        self.inspected_bytes += text.len();
        let marker_strings = self.markers.clone();
        let markers = marker_strings
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let mut ready = String::new();
        let mut cursor = 0;

        if self.pending_quote_candidate {
            let Some(next) = text.chars().next() else {
                return ready;
            };
            self.pending_quote_candidate = false;
            if !is_internal_apostrophe(self.previous_character, Some(next)) {
                self.quote = None;
                self.mask_pending_markers(&markers);
                ready.push_str(&std::mem::take(&mut self.pending));
                self.pending_marker = None;
            }
            self.previous_character = Some('\'');
        }

        while !self.pending.is_empty() && self.pending_marker.is_none() {
            let Some(character) = text[cursor..].chars().next() else {
                return ready;
            };
            let mut candidate = self.pending.clone();
            candidate.push(character);
            if markers.iter().any(|marker| marker.starts_with(&candidate)) {
                let is_complete_marker = markers.iter().any(|marker| **marker == candidate);
                self.pending.push(character);
                cursor += character.len_utf8();
                self.previous_character = Some(character);
                if is_complete_marker {
                    // A configured marker can also prefix a longer marker. Stop on
                    // the complete marker so the next byte cannot flush it unmasked.
                    self.pending_marker = Some(candidate);
                }
            } else {
                ready.push_str(&std::mem::take(&mut self.pending));
            }
        }

        while cursor < text.len() {
            if self.pending_marker.is_some() {
                let character = text[cursor..]
                    .chars()
                    .next()
                    .expect("cursor is before the end of the input");
                let previous = self.previous_character;
                self.pending.push(character);
                cursor += character.len_utf8();
                let next = text[cursor..].chars().next();
                let mut defer_previous_update = false;
                if self.escaped {
                    self.escaped = false;
                } else if character == '\\' {
                    self.escaped = true;
                } else if self.quote == Some(character) {
                    if character == '\'' && is_internal_apostrophe(previous, next) {
                        // Apostrophes inside words do not close a quoted span.
                    } else if character == '\''
                        && next.is_none()
                        && previous.is_some_and(char::is_alphanumeric)
                    {
                        self.pending_quote_candidate = true;
                        defer_previous_update = true;
                    } else {
                        self.quote = None;
                        self.mask_pending_markers(&markers);
                        ready.push_str(&std::mem::take(&mut self.pending));
                        self.pending_marker = None;
                    }
                }
                if !defer_previous_update {
                    self.previous_character = Some(character);
                }
                continue;
            }

            let remaining = &text[cursor..];
            if self.quote.is_some() {
                if let Some(marker) = markers
                    .iter()
                    .find(|marker| remaining.starts_with(**marker))
                {
                    self.pending.push_str(marker);
                    self.pending_marker = Some((*marker).to_string());
                    cursor += marker.len();
                    self.previous_character = marker.chars().next_back();
                    continue;
                }
                if markers.iter().any(|marker| marker.starts_with(remaining)) {
                    self.pending.push_str(remaining);
                    self.previous_character = remaining.chars().next_back();
                    break;
                }
            }

            let character = remaining
                .chars()
                .next()
                .expect("cursor is before the end of the input");
            let previous = self.previous_character;
            ready.push(character);
            cursor += character.len_utf8();
            let next = text[cursor..].chars().next();
            let mut defer_previous_update = false;
            if let Some(quote) = self.quote {
                if self.escaped {
                    self.escaped = false;
                } else if character == '\\' {
                    self.escaped = true;
                } else if character == quote {
                    if quote == '\'' && is_internal_apostrophe(previous, next) {
                        // Apostrophes inside words do not close a quoted span.
                    } else if quote == '\''
                        && next.is_none()
                        && previous.is_some_and(char::is_alphanumeric)
                    {
                        self.pending_quote_candidate = true;
                        defer_previous_update = true;
                    } else {
                        self.quote = None;
                    }
                }
            } else if character == '\'' {
                if !previous.is_some_and(char::is_alphanumeric) {
                    self.quote = Some(character);
                }
            } else if matches!(character, '"' | '`') {
                self.quote = Some(character);
            }
            if !defer_previous_update {
                self.previous_character = Some(character);
            }
        }
        ready
    }

    fn mask_pending_markers(&mut self, markers: &[&str]) {
        let pending = std::mem::take(&mut self.pending);
        self.inspected_bytes += pending.len();
        let mut cursor = 0;
        while cursor < pending.len() {
            if let Some(marker) = markers
                .iter()
                .find(|marker| pending[cursor..].starts_with(**marker))
            {
                self.pending.extend(std::iter::repeat_n('\0', marker.len()));
                cursor += marker.len();
            } else {
                let character = pending[cursor..]
                    .chars()
                    .next()
                    .expect("cursor is before the end of pending text");
                self.pending.push(character);
                cursor += character.len_utf8();
            }
        }
    }

    fn finish(&mut self) -> String {
        if self.pending_quote_candidate {
            self.quote = None;
            self.pending_quote_candidate = false;
        }
        if self.quote.is_none() {
            let marker_strings = self.markers.clone();
            let markers = marker_strings
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            self.mask_pending_markers(&markers);
        }
        self.pending_marker = None;
        self.quote = None;
        self.escaped = false;
        std::mem::take(&mut self.pending)
    }
}

// A test may swap the family's parser for one that records sources, because
// the `dynamo-parsers-v2` families do not record yet.
#[cfg(test)]
type ParserFactory = Box<dyn Fn(&str, &[Tool]) -> Box<dyn UnifiedParser>>;

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_PARSER: std::cell::RefCell<Option<ParserFactory>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn test_parser_override(family: &str, tools: &[Tool]) -> Option<Box<dyn UnifiedParser>> {
    TEST_PARSER.with(|factory| {
        factory
            .borrow()
            .as_ref()
            .map(|factory| factory(family, tools))
    })
}

#[cfg(not(test))]
fn test_parser_override(_family: &str, _tools: &[Tool]) -> Option<Box<dyn UnifiedParser>> {
    None
}

/// Events in output order. While a choice records sources, the caller ranges of
/// each event's text sit beside it (empty for a tool call).
#[derive(Default)]
struct Deltas {
    events: Vec<UnifiedParserEvent>,
    spans: Option<Vec<Spans>>,
}

impl Deltas {
    fn new(recording: bool) -> Self {
        Self {
            events: Vec::new(),
            spans: recording.then(Vec::new),
        }
    }

    fn push(&mut self, event: UnifiedParserEvent, spans: Spans) {
        if let Some(all) = &mut self.spans {
            all.push(spans);
        }
        self.events.push(event);
    }

    fn append(&mut self, mut other: Deltas) {
        debug_assert_eq!(
            self.spans.is_some(),
            other.spans.is_some(),
            "both runs record sources, or neither does"
        );
        self.events.append(&mut other.events);
        if let (Some(all), Some(more)) = (&mut self.spans, other.spans) {
            all.extend(more);
        }
    }
}

/// Why a choice could not start.
pub(crate) enum ChoiceStateError {
    /// The parser could not be built; the adapter passes the choice through.
    Parser(anyhow::Error),
    /// The input asked for sources that this choice cannot record; the adapter
    /// fails the stream.
    Recording(String),
}

impl From<anyhow::Error> for ChoiceStateError {
    fn from(error: anyhow::Error) -> Self {
        Self::Parser(error)
    }
}

/// Source recording for one choice. Present only when the input chunks carry a
/// source slot, so the stock path keeps one `None` field and no other cost.
struct Recording {
    fed: FeedMap,
    scratch: Vec<Range<usize>>,
    /// Beside the events that `push` and `finish` returned, until `choices_for`.
    event_spans: Vec<Spans>,
    /// Beside `pending_qwen_opener_text`.
    qwen_opener: Spans,
    /// Beside each entry of `pending_qwen_opener_reasoning`.
    qwen_reasoning: Vec<Spans>,
    /// The current chunk's upstream `reasoning_content`, for `choices_for`.
    upstream_reasoning: Spans,
    /// Beside the choices that `choices_for` returned.
    choice_sources: Vec<ChoiceSources>,
    error: Option<String>,
}

impl Recording {
    fn fail(&mut self, message: String) {
        self.error.get_or_insert(message);
    }
}

/// Per-choice streaming state: one parser instance plus the bookkeeping the OpenAI
/// streaming tool-call contract needs. One instance parses exactly one choice of one
/// request, which is what gives per-stream isolation by construction.
pub(crate) struct ChoiceState {
    glm_native_input: Option<String>,
    incomplete_native_call: bool,
    family: String,
    parser: Box<dyn UnifiedParser>,
    tool_output_mode: UnifiedToolOutputMode,
    qwen_native_lexical_state: QwenNativeLexicalState,
    /// Qwen's native tool opener is held until it is complete or disproven.
    pending_qwen_opener_text: String,
    /// Reasoning chunks arriving while a split Qwen block opener is ambiguous.
    /// They must follow recovered opener text on mismatch and precede call events
    /// when the opener completes.
    pending_qwen_opener_reasoning: Vec<String>,
    qwen_reasoning_event_positions: HashSet<usize>,
    qwen_reasoning_choice_positions: HashSet<usize>,
    /// Tool indices that have emitted at least one `tool_calls` delta.
    opened_calls: HashSet<usize>,
    /// Upstream IDs and types may arrive after an index opens. Names remain fragments.
    tool_ids_emitted: HashSet<usize>,
    tool_types_emitted: HashSet<usize>,
    /// Stable externally visible indices for parser-local and pass-through calls.
    parser_tool_indices: HashMap<usize, usize>,
    passthrough_tool_indices: HashMap<usize, usize>,
    /// Offset applied to tool indices from a parser instance rebuilt after a detour.
    tool_index_offset: usize,
    /// Next externally visible tool index, retained across parser-instance detours.
    next_tool_index: usize,
    /// Tool indices whose UnifiedParser events have not emitted an explicit completion.
    pending_tool_calls: HashSet<usize>,
    parser_call_completion: HashMap<usize, bool>,
    /// Pass-through OpenAI call argument fragments awaiting terminal or JSON completion.
    pending_parsed_tool_calls: HashMap<usize, String>,
    /// Whether any tool-call delta was emitted, including a streamed fragment.
    tool_emitted: bool,
    /// The parser errored. Later chunks pass through as plain text instead of failing
    /// the request — a parser bug must not turn a served answer into a 500.
    failed: bool,
    /// The operator explicitly selected V2 parsing (`DYN_PARSER_VERSION=2` in dynamo).
    parsers_v2_selected: bool,
    recording: Option<Box<Recording>>,
}

impl ChoiceState {
    #[cfg(test)]
    pub(crate) fn new(
        family: &str,
        tools: &[Tool],
        prefill: impl Into<StreamPrefill>,
        tool_output_mode: UnifiedToolOutputMode,
        guided_streaming: bool,
        parsers_v2_selected: bool,
    ) -> anyhow::Result<Self> {
        Self::build(
            family,
            tools,
            prefill,
            tool_output_mode,
            guided_streaming,
            parsers_v2_selected,
            false,
        )
        .map_err(|error| match error {
            ChoiceStateError::Parser(error) => error,
            ChoiceStateError::Recording(message) => anyhow::anyhow!(message),
        })
    }

    /// Build the choice's parser. With `record`, turn source recording on before
    /// the prompt prefix replays, and count input offsets from after it.
    pub(crate) fn build(
        family: &str,
        tools: &[Tool],
        prefill: impl Into<StreamPrefill>,
        tool_output_mode: UnifiedToolOutputMode,
        guided_streaming: bool,
        parsers_v2_selected: bool,
        record: bool,
    ) -> Result<Self, ChoiceStateError> {
        let prefill = prefill.into();
        // Guided JSON rebuilds call arguments, so it has no byte sources.
        if record && !matches!(&tool_output_mode, UnifiedToolOutputMode::Native) {
            return Err(ChoiceStateError::Recording(format!(
                "the {family} parser cannot record sources in guided mode"
            )));
        }
        let mut parser = test_parser_override(family, tools)
            .map(Ok)
            .unwrap_or_else(|| create_unified_parser_for_family(family, tools))
            .map_err(ChoiceStateError::Parser)?;
        // Harmony needs the consumed assistant header to recognize a generated
        // `to=` recipient. Other families infer their state from the rendered text.
        //
        // Streaming guided calls commit per call once the name and argument object opener
        // are unambiguous. A committed fragment cannot later be rejected or recovered as
        // text, so this path opts into that contract explicitly — unless the request
        // rolled guided tool streaming back, in which case it buffers to completion.
        // `guided_streaming` is decided once per request by
        // `OpenAIPreprocessor::guided_tool_streaming_release` and threaded down here;
        // it is inert in `UnifiedToolOutputMode::Native`, where the parser builds no
        // guided state at all.
        parser
            .initialize_request(UnifiedParserInit {
                prompt_token_ids: if family == "gpt_oss"
                    && matches!(&tool_output_mode, UnifiedToolOutputMode::Native)
                    && prefill.starting_state == UnifiedParserStartingState::None
                {
                    dynamo_parsers_v2::unified::harmony::encode_harmony("<|start|>assistant")?
                } else {
                    Vec::new()
                },
                starting_state: if prefill.prefix.is_some() {
                    UnifiedParserStartingState::None
                } else {
                    prefill.starting_state
                },
                tool_output_mode: tool_output_mode.clone(),
                invalid_guided_payload: invalid_guided_payload_policy(guided_streaming),
            })
            .map_err(ChoiceStateError::Parser)?;
        if record && !parser.record_sources() {
            return Err(ChoiceStateError::Recording(format!(
                "the {family} parser cannot record sources"
            )));
        }
        if let Some(prefix) = prefill.prefix {
            // Replay only consumed structural bytes; the grammar retains ownership
            // of an optional channel label that generation has not completed yet.
            let mut output = UnifiedParserOutput::default();
            parser
                .parse_into(prefix, &mut output)
                .map_err(ChoiceStateError::Parser)?;
            if !output.events.is_empty() {
                return Err(ChoiceStateError::Parser(anyhow::anyhow!(
                    "prompt prefix emitted visible output"
                )));
            }
        }
        let recording = record.then(|| {
            let mut scratch = Vec::new();
            // The prefix emitted nothing, so it committed no sources.
            parser.take_sources(&mut scratch);
            scratch.clear();
            Box::new(Recording {
                fed: FeedMap::new(prefill.prefix.map_or(0, str::len)),
                scratch,
                event_spans: Vec::new(),
                qwen_opener: Vec::new(),
                qwen_reasoning: Vec::new(),
                upstream_reasoning: Vec::new(),
                choice_sources: Vec::new(),
                error: None,
            })
        });
        Ok(Self {
            glm_native_input: (family == GLM47_UNIFIED_FAMILY
                && matches!(&tool_output_mode, UnifiedToolOutputMode::Native))
            .then(String::new),
            incomplete_native_call: false,
            family: family.to_string(),
            parser,
            tool_output_mode,
            qwen_native_lexical_state: QwenNativeLexicalState::default(),
            pending_qwen_opener_text: String::new(),
            pending_qwen_opener_reasoning: Vec::new(),
            qwen_reasoning_event_positions: HashSet::new(),
            qwen_reasoning_choice_positions: HashSet::new(),
            opened_calls: HashSet::new(),
            tool_ids_emitted: HashSet::new(),
            tool_types_emitted: HashSet::new(),
            parser_tool_indices: HashMap::new(),
            passthrough_tool_indices: HashMap::new(),
            tool_index_offset: 0,
            next_tool_index: 0,
            pending_tool_calls: HashSet::new(),
            parser_call_completion: HashMap::new(),
            pending_parsed_tool_calls: HashMap::new(),
            tool_emitted: false,
            failed: false,
            parsers_v2_selected,
            recording,
        })
    }

    fn recording(&self) -> bool {
        self.recording.is_some()
    }

    /// The first source-recording failure, if any. The adapter fails the stream.
    pub(crate) fn take_recording_error(&mut self) -> Option<String> {
        self.recording
            .as_mut()
            .and_then(|recording| recording.error.take())
    }

    /// The sources of the choices that the last `choices_for` returned, in order.
    pub(crate) fn take_choice_sources(&mut self) -> Option<Vec<ChoiceSources>> {
        self.recording
            .as_mut()
            .map(|recording| std::mem::take(&mut recording.choice_sources))
    }

    /// Ranges of the upstream `reasoning_content` that the next `choices_for`
    /// copies from its `original` choice.
    pub(crate) fn set_upstream_reasoning(&mut self, spans: Spans) {
        if let Some(recording) = &mut self.recording {
            recording.upstream_reasoning = spans;
        }
    }

    fn take_qwen_opener_spans(&mut self) -> Spans {
        self.recording
            .as_mut()
            .map(|recording| std::mem::take(&mut recording.qwen_opener))
            .unwrap_or_default()
    }

    fn set_qwen_opener_spans(&mut self, spans: Spans) {
        if let Some(recording) = &mut self.recording {
            recording.qwen_opener = spans;
        }
    }

    /// Hand the events back and keep their ranges for `choices_for`.
    fn stash(&mut self, deltas: Deltas) -> Vec<UnifiedParserEvent> {
        if let (Some(recording), Some(spans)) = (&mut self.recording, deltas.spans) {
            recording.event_spans.extend(spans);
        }
        deltas.events
    }

    /// Drop the sources of text the parser committed but this choice discards.
    fn discard_parser_sources(&mut self) {
        if let Some(recording) = &mut self.recording {
            self.parser.take_sources(&mut recording.scratch);
            recording.scratch.clear();
        }
    }

    /// Give each text event the caller ranges of the bytes the parser committed.
    fn attach_parser_sources(&mut self, events: Vec<UnifiedParserEvent>) -> Deltas {
        let Some(recording) = self.recording.as_deref_mut() else {
            return Deltas {
                events,
                spans: None,
            };
        };
        self.parser.take_sources(&mut recording.scratch);
        let mut local = std::mem::take(&mut recording.scratch);
        let mut cursor = 0;
        let mut deltas = Deltas::new(true);
        for event in events {
            let text = match &event {
                UnifiedParserEvent::Text(text) | UnifiedParserEvent::Reasoning(text) => {
                    text.as_bytes()
                }
                _ => &[],
            };
            let mut need = text.len();
            let mut spans = Vec::new();
            while need > 0 {
                let Some(range) = local.get_mut(cursor) else {
                    recording.fail(format!(
                        "the {} parser reported fewer source bytes than it emitted",
                        self.family
                    ));
                    break;
                };
                let take = need.min(range.len());
                let done = text.len() - need;
                if let Err(message) = recording.fed.map(
                    range.start..range.start + take,
                    &text[done..done + take],
                    &mut spans,
                ) {
                    recording.fail(message);
                }
                range.start += take;
                need -= take;
                if range.start >= range.end {
                    cursor += 1;
                }
            }
            deltas.push(event, spans);
        }
        if local[cursor.min(local.len())..]
            .iter()
            .any(|range| range.start < range.end)
        {
            recording.fail(format!(
                "the {} parser reported more source bytes than it emitted",
                self.family
            ));
        }
        local.clear();
        recording.scratch = local;
        deltas
    }

    /// Feed one decoded text delta through the parser.
    #[cfg(test)]
    pub(crate) fn push(&mut self, text: &str) -> Vec<UnifiedParserEvent> {
        self.push_sourced(text, Vec::new())
    }

    /// [`push`](Self::push), with the caller ranges that tile `text`.
    pub(crate) fn push_sourced(&mut self, text: &str, spans: Spans) -> Vec<UnifiedParserEvent> {
        let deltas = self.push_deltas(text, spans);
        self.stash(deltas)
    }

    fn push_deltas(&mut self, text: &str, spans: Spans) -> Deltas {
        if self.failed {
            return self.text_delta_sourced(text.to_string(), spans);
        }
        if self.is_qwen_native_outside_block() {
            let held = std::mem::take(&mut self.pending_qwen_opener_text);
            let mut held_spans = self.take_qwen_opener_spans();
            if !self.pending_qwen_opener_reasoning.is_empty() && !held.is_empty() {
                if let Some((_resolution_end, opener_completed)) =
                    qwen_opener_resolution(&held, text)
                {
                    let mut combined = held.clone();
                    combined.push_str(text);
                    append_spans(&mut held_spans, spans);
                    let mut events = self.parse_segment(&combined, held_spans);
                    if opener_completed {
                        self.prepend_pending_qwen_reasoning(&mut events);
                    } else {
                        self.insert_pending_qwen_reasoning_after_recovered_opener(
                            &held,
                            &mut events,
                        );
                    }
                    return events;
                }
                self.pending_qwen_opener_text = held;
                self.pending_qwen_opener_text.push_str(text);
                append_spans(&mut held_spans, spans);
                self.set_qwen_opener_spans(held_spans);
                return Deltas::new(self.recording());
            }

            let mut combined = held;
            combined.push_str(text);
            append_spans(&mut held_spans, spans);
            if let Some(start) =
                qwen_partial_opener_suffix_start(&self.qwen_native_lexical_state, &combined)
            {
                let safe_text = combined[..start].to_string();
                self.pending_qwen_opener_text = combined[start..].to_string();
                let tail = split_spans(&mut held_spans, start);
                self.set_qwen_opener_spans(tail);
                return if safe_text.is_empty() {
                    Deltas::new(self.recording())
                } else {
                    self.parse_segment(&safe_text, held_spans)
                };
            }
            return self.parse_segment(&combined, held_spans);
        }
        self.parse_segment(text, spans)
    }

    fn qwen_resumed_marker_can_start_call(&self) -> bool {
        self.qwen_native_lexical_state.mode != QwenNativeLexicalMode::ParameterValue
    }

    fn is_qwen_native_outside_block(&self) -> bool {
        self.family == QWEN3_UNIFIED_FAMILY
            && matches!(&self.tool_output_mode, UnifiedToolOutputMode::Native)
            && self.qwen_native_lexical_state.mode == QwenNativeLexicalMode::OutsideBlock
    }

    fn parse_segment(&mut self, text: &str, spans: Spans) -> Deltas {
        if let Some(input) = &mut self.glm_native_input {
            // EOF recovery needs source context; scan once there, not on every push.
            input.push_str(text);
        }
        if self.family == QWEN3_UNIFIED_FAMILY
            && matches!(&self.tool_output_mode, UnifiedToolOutputMode::Native)
        {
            self.qwen_native_lexical_state.observe(text);
        }
        if let Some(recording) = &mut self.recording {
            if spans_len(&spans) != text.len() {
                recording.fail(format!(
                    "input sources cover {} bytes of {} bytes of text",
                    spans_len(&spans),
                    text.len()
                ));
            }
            recording.fed.feed(&spans, text);
        }
        let mut output = UnifiedParserOutput::default();
        match self.parser.parse_into(text, &mut output) {
            Ok(()) => self.attach_parser_sources(output.events),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    family = self.family,
                    "unified parser push failed; falling back to plain text for this choice"
                );
                let mut events = self.attach_parser_sources(output.events);
                events.append(self.give_up_sourced(text, spans));
                events
            }
        }
    }

    fn queue_qwen_opener_reasoning_sourced(&mut self, reasoning: String, spans: Spans) {
        self.pending_qwen_opener_reasoning.push(reasoning);
        if let Some(recording) = &mut self.recording {
            recording.qwen_reasoning.push(spans);
        }
    }

    /// The queued reasoning as events, with their ranges.
    fn take_pending_qwen_reasoning(&mut self) -> Deltas {
        let reasoning = std::mem::take(&mut self.pending_qwen_opener_reasoning);
        let mut spans = self
            .recording
            .as_mut()
            .map(|recording| std::mem::take(&mut recording.qwen_reasoning).into_iter());
        let mut deltas = Deltas::new(self.recording());
        for text in reasoning {
            let text_spans = spans.as_mut().and_then(Iterator::next).unwrap_or_default();
            deltas.push(UnifiedParserEvent::Reasoning(text), text_spans);
        }
        deltas
    }

    fn prepend_pending_qwen_reasoning(&mut self, events: &mut Deltas) {
        let mut reasoning = self.take_pending_qwen_reasoning();
        self.qwen_reasoning_event_positions
            .extend(0..reasoning.events.len());
        reasoning.append(std::mem::take(events));
        *events = reasoning;
    }

    fn append_pending_qwen_reasoning(&mut self, events: &mut Deltas) {
        let reasoning = self.take_pending_qwen_reasoning();
        self.qwen_reasoning_event_positions
            .extend(events.events.len()..events.events.len() + reasoning.events.len());
        events.append(reasoning);
    }

    fn insert_pending_qwen_reasoning_after_recovered_opener(
        &mut self,
        held: &str,
        events: &mut Deltas,
    ) {
        let pending = self.take_pending_qwen_reasoning();
        let recording = events.spans.is_some();
        let mut ordered = Deltas::new(recording);
        let mut inserted = false;
        let taken = std::mem::take(events);
        let mut spans = taken.spans.map(Vec::into_iter);
        let mut pending = Some(pending);
        for event in taken.events {
            let mut event_spans = spans.as_mut().and_then(Iterator::next).unwrap_or_default();
            if !inserted
                && let UnifiedParserEvent::Text(text) = &event
                && let Some(remainder) = text.strip_prefix(held)
            {
                let remainder_spans = split_spans(&mut event_spans, held.len());
                if !held.is_empty() {
                    ordered.push(UnifiedParserEvent::Text(held.to_string()), event_spans);
                }
                let pending = pending.take().unwrap_or_default();
                self.qwen_reasoning_event_positions
                    .extend(ordered.events.len()..ordered.events.len() + pending.events.len());
                ordered.append(pending);
                if !remainder.is_empty() {
                    ordered.push(
                        UnifiedParserEvent::Text(remainder.to_string()),
                        remainder_spans,
                    );
                }
                inserted = true;
                continue;
            }
            ordered.push(event, event_spans);
        }
        if let Some(pending) = pending {
            // TODO(frontend-crates): parser 0.7.17 malformed-call recovery still needs
            // a source-order barrier. Recovered text can span this thought at EOF;
            // matching a repeated opener cannot recover its original position.
            self.qwen_reasoning_event_positions
                .extend(ordered.events.len()..ordered.events.len() + pending.events.len());
            ordered.append(pending);
        }
        *events = ordered;
    }

    /// Flush buffered partial state at end of stream.
    pub(crate) fn finish(&mut self) -> Vec<UnifiedParserEvent> {
        let deltas = self.finish_deltas();
        self.stash(deltas)
    }

    fn finish_deltas(&mut self) -> Deltas {
        if self.failed {
            return Deltas::new(self.recording());
        }
        if let Some(input) = &self.glm_native_input {
            match glm_unfinished_call(input) {
                Ok(Some(start)) => {
                    self.incomplete_native_call = true;
                    tracing::warn!(
                        family = self.family,
                        why = "truncated_native_tool_call_suppressed",
                        suppressed_bytes = input.len() - start,
                        "suppressing unfinished native tool call"
                    );
                    self.parser.reset();
                    self.discard_parser_sources();
                    return Deltas::new(self.recording());
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(%error, why = "glm_pending_input_failed", "using parser finish recovery")
                }
            }
        }
        let pending_qwen_opener = std::mem::take(&mut self.pending_qwen_opener_text);
        let pending_spans = self.take_qwen_opener_spans();
        let mut events = if pending_qwen_opener.is_empty() {
            Deltas::new(self.recording())
        } else {
            self.parse_segment(&pending_qwen_opener, pending_spans)
        };
        match self.parser.finish() {
            // `finish` now hands back the whole `UnifiedParserOutput`; this path only ever
            // wants the ordered events out of it.
            Ok(output) => events.append(self.attach_parser_sources(output.events)),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    family = self.family,
                    "unified parser finish failed; recovering buffered text"
                );
                // Output from the failed finish is dropped, so are its sources.
                self.discard_parser_sources();
                events.append(self.give_up_sourced("", Vec::new()));
            }
        }
        self.append_pending_qwen_reasoning(&mut events);
        events
    }

    /// Stop using the parser and surface whatever it was holding.
    ///
    /// `reset()` hands back the bytes the parser had buffered but not yet emitted.
    /// Dropping them would silently delete model output — the caller would see a
    /// truncated answer with no indication anything was lost — so they go out as
    /// visible text. When the parser had nothing buffered, the chunk that broke it
    /// does instead, so that chunk is not lost either.
    #[cfg(test)]
    fn give_up(&mut self, fallback: &str) -> Vec<UnifiedParserEvent> {
        self.give_up_sourced(fallback, Vec::new()).events
    }

    /// [`give_up`](Self::give_up), with the ranges of `fallback`.
    ///
    /// When the parser hands nothing back, `fallback` is the chunk that broke it.
    /// If the parser had committed part of that chunk before the error, those
    /// bytes go out twice, and so do their ranges (see the TODO below).
    fn give_up_sourced(&mut self, fallback: &str, fallback_spans: Spans) -> Deltas {
        self.failed = true;
        let mut recovered = self.parser.reset();
        let mut spans = Vec::new();
        if let Some(recording) = self.recording.as_deref_mut() {
            self.parser.take_sources(&mut recording.scratch);
            let mut done = 0;
            for range in recording.scratch.drain(..) {
                let expected = recovered.as_bytes().get(done..done + range.len());
                done += range.len();
                let result = match expected {
                    Some(expected) => recording.fed.map(range, expected, &mut spans),
                    None => Err("reset sources cover more bytes than the reset text".to_string()),
                };
                if let Err(message) = result {
                    recording.error.get_or_insert(message);
                }
            }
        }
        if recovered.is_empty() {
            // TODO: Preserve the parser's consumed offset so a partial-commit error cannot replay input.
            recovered.push_str(fallback);
            spans = fallback_spans;
        }
        recovered.push_str(&std::mem::take(&mut self.pending_qwen_opener_text));
        append_spans(&mut spans, self.take_qwen_opener_spans());
        if let Some(recording) = &mut self.recording
            && spans_len(&spans) != recovered.len()
        {
            recording.fail(format!(
                "recovered text of {} bytes has sources for {} bytes",
                recovered.len(),
                spans_len(&spans)
            ));
        }
        self.text_delta_sourced(recovered, spans)
    }

    fn text_delta_sourced(&self, text: String, spans: Spans) -> Deltas {
        let mut deltas = Deltas::new(self.recording());
        if !text.is_empty() {
            deltas.push(UnifiedParserEvent::Text(text), spans);
        }
        deltas
    }

    /// Convert one ordered delta into a streaming choice for `index`.
    fn delta_to_choice(&mut self, index: u32, delta: UnifiedParserEvent) -> ChatChoiceStream {
        let mut choice = empty_choice(index);
        match delta {
            // dynamo-parsers-v2 0.8 adds message phase events. A parser emits them
            // only after `preserve_message_phases(true)`, which this adapter never
            // calls.
            UnifiedParserEvent::MessageStart(_) | UnifiedParserEvent::MessageEnd => {}
            UnifiedParserEvent::Text(text) => {
                choice.delta.content = Some(ChatCompletionMessageContent::Text(text));
            }
            UnifiedParserEvent::Reasoning(text) => {
                choice.delta.reasoning_content = Some(text);
            }
            UnifiedParserEvent::ToolCall(call) => {
                let parser_index = call.tool_index + self.tool_index_offset;
                let tool_index = stable_tool_index(
                    parser_index,
                    &mut self.parser_tool_indices,
                    &self.opened_calls,
                    &mut self.next_tool_index,
                );
                self.tool_emitted = true;
                let complete = self
                    .parser_call_completion
                    .entry(tool_index)
                    .and_modify(|complete| *complete |= call.complete)
                    .or_insert(call.complete);
                if *complete {
                    self.pending_tool_calls.remove(&tool_index);
                    self.pending_parsed_tool_calls.remove(&tool_index);
                } else {
                    self.pending_tool_calls.insert(tool_index);
                }
                // The OpenAI streaming tool-call contract: the FIRST chunk for a tool
                // index carries id + type + name, later chunks carry only argument
                // fragments. Preserve model IDs when the native grammar supplies one.
                let first = self.opened_calls.insert(tool_index);
                if first {
                    self.tool_ids_emitted.insert(tool_index);
                    self.tool_types_emitted.insert(tool_index);
                }
                choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
                    index: tool_index as u32,
                    id: first.then(|| {
                        self.parser
                            .tool_call_id(call.tool_index)
                            .map(str::to_string)
                            .unwrap_or_else(|| format!("call-{}", Uuid::new_v4()))
                    }),
                    r#type: first.then_some(FunctionType::Function),
                    function: Some(FunctionCallStream {
                        name: first.then_some(call.name).flatten(),
                        arguments: Some(call.arguments),
                    }),
                    // dynamo-protocols 8.4 adds `custom`; this keeps it unset on 8.2 and 8.4.
                    ..Default::default()
                }]);
            }
        }
        choice
    }

    /// Convert an ordered delta run into the streaming choices it becomes.
    ///
    /// `role` / `refusal` ride on the first choice owned by the current source
    /// response; queued reasoning keeps its earlier source envelope. The terminating
    /// `finish_reason` stays after every emitted delta.
    pub(crate) fn choices_for(
        &mut self,
        original: &ChatChoiceStream,
        deltas: Vec<UnifiedParserEvent>,
        emit_tool_calls: bool,
        finish_reason: Option<FinishReason>,
    ) -> Vec<ChatChoiceStream> {
        let terminal_tool_call_delta = finish_reason_closes_tool_calls(finish_reason.as_ref());
        let recording = self.recording();
        let (mut delta_spans, upstream_reasoning) = match self.recording.as_deref_mut() {
            Some(recording) => {
                let spans = std::mem::take(&mut recording.event_spans);
                if spans.len() != deltas.len() {
                    recording.fail(format!(
                        "{} events have sources for {}",
                        deltas.len(),
                        spans.len()
                    ));
                }
                (
                    Some(spans.into_iter()),
                    std::mem::take(&mut recording.upstream_reasoning),
                )
            }
            None => (None, Vec::new()),
        };
        // Queued reasoning belongs to its earlier source response. Coalescing it
        // with parser reasoning would erase the source envelope's event boundary.
        let source_positions = std::mem::take(&mut self.qwen_reasoning_event_positions);
        self.qwen_reasoning_choice_positions.clear();
        let mut run = Deltas::new(recording);
        let mut projected = Deltas::new(recording);
        for (position, delta) in deltas.into_iter().enumerate() {
            let spans = delta_spans
                .as_mut()
                .and_then(Iterator::next)
                .unwrap_or_default();
            if !emit_tool_calls && matches!(delta, UnifiedParserEvent::ToolCall(_)) {
                continue;
            }
            if source_positions.contains(&position) {
                projected.append(coalesce_deltas(std::mem::replace(
                    &mut run,
                    Deltas::new(recording),
                )));
                self.qwen_reasoning_choice_positions
                    .insert(projected.events.len());
                projected.push(delta, spans);
            } else {
                run.push(delta, spans);
            }
        }
        projected.append(coalesce_deltas(run));
        let Deltas {
            events: deltas,
            spans: projected_spans,
        } = projected;
        let mut projected_spans = projected_spans.map(Vec::into_iter);
        let mut sources: Option<Vec<ChoiceSources>> = recording.then(Vec::new);
        let mut upstream_reasoning = Some(upstream_reasoning);
        let envelope_position = (0..deltas.len())
            .find(|position| !self.qwen_reasoning_choice_positions.contains(position));
        let mut original = original.clone();
        if let Some(original_calls) = original.delta.tool_calls.as_mut() {
            for call in original_calls {
                let tool_index = normalize_passthrough_tool_call(
                    call,
                    &mut self.passthrough_tool_indices,
                    &mut self.opened_calls,
                    &mut self.tool_ids_emitted,
                    &mut self.tool_types_emitted,
                    &mut self.next_tool_index,
                );
                append_parsed_tool_call_arguments(
                    &mut self.pending_parsed_tool_calls,
                    tool_index,
                    call,
                );
                self.tool_emitted = true;
            }
        }
        if terminal_tool_call_delta {
            resolve_complete_parsed_tool_calls_at_terminal(
                &mut self.pending_parsed_tool_calls,
                &self.family,
                finish_reason == Some(FinishReason::ToolCalls),
                self.parsers_v2_selected,
            );
        }
        let index = original.index;
        let count = deltas.len();
        let mut choices = Vec::with_capacity(count.max(1));

        for (position, delta) in deltas.into_iter().enumerate() {
            let delta_spans = projected_spans
                .as_mut()
                .and_then(Iterator::next)
                .unwrap_or_default();
            let mut choice_sources = ChoiceSources {
                index,
                ..ChoiceSources::default()
            };
            match &delta {
                UnifiedParserEvent::Text(_) => choice_sources.content = delta_spans,
                UnifiedParserEvent::Reasoning(_) => choice_sources.reasoning = delta_spans,
                _ => {}
            }
            let mut choice = self.delta_to_choice(index, delta);
            if Some(position) == envelope_position {
                choice.delta.role = original.delta.role;
                choice.delta.refusal = original.delta.refusal.clone();
                choice.delta.function_call = original.delta.function_call.clone();
                if let Some(reasoning) = original.delta.reasoning_content.as_deref() {
                    choice
                        .delta
                        .reasoning_content
                        .get_or_insert_default()
                        .insert_str(0, reasoning);
                    let mut spans = upstream_reasoning.take().unwrap_or_default();
                    append_spans(&mut spans, std::mem::take(&mut choice_sources.reasoning));
                    choice_sources.reasoning = spans;
                }
                if let Some(mut original_calls) = original.delta.tool_calls.clone() {
                    if let Some(parsed_calls) = choice.delta.tool_calls.take() {
                        original_calls.extend(parsed_calls);
                    }
                    choice.delta.tool_calls = Some(original_calls);
                }
            }
            if position + 1 == count && !self.qwen_reasoning_choice_positions.contains(&position) {
                choice.finish_reason = self.normalize_finish_reason(finish_reason);
            }
            if let Some(sources) = &mut sources {
                sources.push(choice_sources);
            }
            choices.push(choice);
        }

        if !choices.is_empty()
            && (envelope_position.is_none()
                || (finish_reason.is_some()
                    && self.qwen_reasoning_choice_positions.contains(&(count - 1))))
        {
            let mut choice = empty_choice(index);
            let mut choice_sources = ChoiceSources {
                index,
                ..ChoiceSources::default()
            };
            if envelope_position.is_none() {
                choice.delta.role = original.delta.role;
                choice.delta.refusal = original.delta.refusal.clone();
                choice.delta.reasoning_content = original.delta.reasoning_content.clone();
                choice.delta.tool_calls = original.delta.tool_calls.clone();
                choice.delta.function_call = original.delta.function_call.clone();
                if choice.delta.reasoning_content.is_some() {
                    choice_sources.reasoning = upstream_reasoning.take().unwrap_or_default();
                }
            }
            choice.finish_reason = self.normalize_finish_reason(finish_reason);
            if let Some(sources) = &mut sources {
                sources.push(choice_sources);
            }
            choices.push(choice);
        }

        // The parser produced nothing, but the chunk still carried envelope state that
        // has to reach the client (the opening role chunk, a refusal, or the terminal
        // finish_reason).
        if choices.is_empty()
            && (original.delta.role.is_some()
                || original.delta.refusal.is_some()
                || original.delta.reasoning_content.is_some()
                || original.delta.tool_calls.is_some()
                || original.delta.function_call.is_some()
                || finish_reason.is_some())
        {
            let mut choice = empty_choice(index);
            choice.delta.role = original.delta.role;
            choice.delta.refusal = original.delta.refusal.clone();
            choice.delta.reasoning_content = original.delta.reasoning_content.clone();
            choice.delta.tool_calls = original.delta.tool_calls.clone();
            choice.delta.function_call = original.delta.function_call.clone();
            if choice
                .delta
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
            {
                self.tool_emitted = true;
            }
            choice.finish_reason = self.normalize_finish_reason(finish_reason);
            if let Some(sources) = &mut sources {
                sources.push(ChoiceSources {
                    index,
                    content: Vec::new(),
                    reasoning: if choice.delta.reasoning_content.is_some() {
                        upstream_reasoning.take().unwrap_or_default()
                    } else {
                        Vec::new()
                    },
                });
            }
            choices.push(choice);
        }

        if let Some(recording) = self.recording.as_deref_mut() {
            if upstream_reasoning.is_some_and(|spans| !spans.is_empty()) {
                recording.fail("upstream reasoning sources were not placed".to_string());
            }
            recording.choice_sources = sources.unwrap_or_default();
        }
        choices
    }

    pub(crate) fn unterminated_finish_reason(&self) -> Option<FinishReason> {
        self.normalize_finish_reason(Some(FinishReason::Stop))
            .filter(|reason| *reason == FinishReason::Length || *reason == FinishReason::ToolCalls)
    }

    /// Whether this choice has emitted a tool-call delta so far.
    pub(crate) fn tool_emitted(&self) -> bool {
        self.tool_emitted
    }

    fn resolve_complete_parsed_tool_calls_at_eof(&mut self) {
        resolve_complete_parsed_tool_calls_at_eof(
            &mut self.pending_parsed_tool_calls,
            &self.family,
            self.parsers_v2_selected,
        );
    }

    fn should_preserve_reasoning_only_state(&self) -> bool {
        // A nonterminal out-of-band thought cannot finish pending native or guided input.
        !self.failed
    }

    /// Seed tool-call history from before an already-parsed detour into a freshly
    /// constructed parser instance for the same choice index.
    pub(crate) fn mark_tool_emitted(&mut self) {
        self.tool_emitted = true;
    }

    /// OpenAI streaming contract: `Stop` becomes `ToolCalls` only when every emitted
    /// call index is complete. `Length` / `ContentFilter` stay unchanged.
    fn normalize_finish_reason(&self, finish_reason: Option<FinishReason>) -> Option<FinishReason> {
        normalize_tool_finish_reason(
            finish_reason,
            self.tool_emitted,
            self.incomplete_native_call
                || !self.pending_tool_calls.is_empty()
                || !self.pending_parsed_tool_calls.is_empty()
                || has_incomplete_call_evidence(&self.parser_call_completion),
        )
    }
}

// OpenAI tool-call chunks have no per-delta completion bit. Retain every argument
// fragment so EOF can distinguish a complete JSON value from a truncated call.
fn append_parsed_tool_call_arguments(
    pending: &mut HashMap<usize, String>,
    tool_index: usize,
    call: &ChatCompletionMessageToolCallChunk,
) {
    let arguments = call
        .function
        .as_ref()
        .and_then(|function| function.arguments.as_deref())
        .unwrap_or_default();
    pending.entry(tool_index).or_default().push_str(arguments);
}

fn resolve_complete_parsed_tool_calls_at_eof(
    pending: &mut HashMap<usize, String>,
    family: &str,
    parsers_v2_selected: bool,
) {
    resolve_complete_parsed_tool_calls_at_terminal(pending, family, false, parsers_v2_selected);
}

fn resolve_complete_parsed_tool_calls_at_terminal(
    pending: &mut HashMap<usize, String>,
    family: &str,
    allow_empty_parameterless: bool,
    parsers_v2_selected: bool,
) {
    // Default-on families preserve backend arguments at terminal/EOF with the experiment off.
    // Native pending calls and explicit parser completion evidence remain separate guards.
    if matches!(family, DEEPSEEK_V41_UNIFIED_FAMILY | MUSE_UNIFIED_FAMILY) && !parsers_v2_selected {
        pending.clear();
        return;
    }
    pending.retain(|_, arguments| {
        !(allow_empty_parameterless && arguments.is_empty())
            && serde_json::from_str::<serde_json::Value>(arguments).is_err()
    });
}

fn finish_reason_closes_tool_calls(finish_reason: Option<&FinishReason>) -> bool {
    matches!(
        finish_reason,
        Some(FinishReason::Stop | FinishReason::ToolCalls)
    )
}

fn stable_tool_index(
    source_index: usize,
    indices: &mut HashMap<usize, usize>,
    opened_calls: &HashSet<usize>,
    next_tool_index: &mut usize,
) -> usize {
    if let Some(tool_index) = indices.get(&source_index) {
        return *tool_index;
    }
    let tool_index = if opened_calls.contains(&source_index) {
        let mut candidate = *next_tool_index;
        while opened_calls.contains(&candidate) {
            candidate += 1;
        }
        candidate
    } else {
        source_index
    };
    indices.insert(source_index, tool_index);
    *next_tool_index = (*next_tool_index).max(tool_index + 1);
    tool_index
}

fn normalize_passthrough_tool_call(
    call: &mut ChatCompletionMessageToolCallChunk,
    indices: &mut HashMap<usize, usize>,
    opened_calls: &mut HashSet<usize>,
    ids_emitted: &mut HashSet<usize>,
    types_emitted: &mut HashSet<usize>,
    next_tool_index: &mut usize,
) -> usize {
    let tool_index = stable_tool_index(call.index as usize, indices, opened_calls, next_tool_index);
    if call.id.is_some() && !ids_emitted.insert(tool_index) {
        call.id = None;
    }
    if call.r#type.is_some() && !types_emitted.insert(tool_index) {
        call.r#type = None;
    }
    call.index = tool_index as u32;
    opened_calls.insert(tool_index);
    tool_index
}

/// Per-choice bookkeeping that outlives any single `ChoiceState` instance for that
/// index: whether it has emitted a tool call, which indices remain provisional, and whether its terminal chunk has
/// already been sent. See the field comment at its use site in
/// `apply_stream_with_constraint` for why this can't just live inside `ChoiceState`.
struct ChoiceRecord<I: UnifiedStreamItem> {
    incomplete_native_call: bool,
    opened_calls: HashSet<usize>,
    tool_ids_emitted: HashSet<usize>,
    tool_types_emitted: HashSet<usize>,
    parser_tool_indices: HashMap<usize, usize>,
    passthrough_tool_indices: HashMap<usize, usize>,
    next_tool_index: usize,
    /// Incomplete events from the live UnifiedParser instance.
    pending_tool_calls: HashSet<usize>,
    parser_call_completion: HashMap<usize, bool>,
    /// Pass-through OpenAI call argument fragments awaiting terminal or JSON completion.
    pending_parsed_tool_calls: HashMap<usize, String>,
    /// Leading whitespace held until a Reasoning-prefill guided request reveals
    /// whether generation starts with reasoning prose or bare guided JSON.
    pending_guided_prefill_whitespace: String,
    /// Source envelopes for Qwen reasoning held until a split native opener resolves.
    pending_qwen_opener_responses: VecDeque<PendingQwenReasoningResponse<I>>,
    tool_emitted: bool,
    finished: bool,
    /// Whether an already-parsed chunk has EVER interrupted this choice. An
    /// already-parsed chunk implies the model has already emitted structured output
    /// (reasoning_content / tool_calls / function_call / `Parts` content), which per
    /// `already_parsed`'s own trigger set only happens after any reasoning phase would
    /// have concluded. So a raw run resuming after such a detour must rebuild its
    /// `ChoiceState` starting at `Response`, never at the outer request-level `prefill`
    /// (which only describes what the PROMPT opened, before this choice ever ran) — see
    /// the `Vacant` arm in the main loop below.
    detoured: bool,
}

// A derived `Default` would require `I: Default`.
impl<I: UnifiedStreamItem> Default for ChoiceRecord<I> {
    fn default() -> Self {
        Self {
            incomplete_native_call: false,
            opened_calls: HashSet::new(),
            tool_ids_emitted: HashSet::new(),
            tool_types_emitted: HashSet::new(),
            parser_tool_indices: HashMap::new(),
            passthrough_tool_indices: HashMap::new(),
            next_tool_index: 0,
            pending_tool_calls: HashSet::new(),
            parser_call_completion: HashMap::new(),
            pending_parsed_tool_calls: HashMap::new(),
            pending_guided_prefill_whitespace: String::new(),
            pending_qwen_opener_responses: VecDeque::new(),
            tool_emitted: false,
            finished: false,
            detoured: false,
        }
    }
}

struct PendingQwenReasoningResponse<I: UnifiedStreamItem> {
    text: String,
    /// Ranges of `text` when recording; empty otherwise.
    spans: Spans,
    choice: ChatChoiceStream,
    data: I::Chunk,
    annotation: I::Annotation,
}

// The source chunk is boxed so an ordinary choice, the common case, stays small.
#[allow(clippy::large_enum_variant)]
enum FinishedStreamChunk<I: UnifiedStreamItem> {
    Choice(ChatChoiceStream),
    /// A choice and its sources, only while recording.
    SourcedChoice(Box<(ChatChoiceStream, ChoiceSources)>),
    Source(Box<(I::Chunk, I::Annotation)>),
}

impl<I: UnifiedStreamItem> FinishedStreamChunk<I> {
    fn choice(choice: ChatChoiceStream, sources: Option<ChoiceSources>) -> Self {
        match sources {
            Some(sources) => Self::SourcedChoice(Box::new((choice, sources))),
            None => Self::Choice(choice),
        }
    }
}

/// Overwrite a chunk's slot, when it has one, with these choices' sources.
fn write_slot<C: UnifiedStreamChunk>(chunk: &mut C, sources: Vec<ChoiceSources>) {
    if let Some(slot) = chunk.sources_mut() {
        *slot = sources;
    }
}

/// Add a choice to the chunk being built, with its sources while recording.
fn push_emitted(
    emitted: &mut Vec<ChatChoiceStream>,
    emitted_sources: &mut Option<Vec<ChoiceSources>>,
    choice: ChatChoiceStream,
    sources: ChoiceSources,
) {
    if let Some(all) = emitted_sources {
        all.push(sources);
    }
    emitted.push(choice);
}

/// The sources of a probe's held text, in whichever field it was released into.
fn probe_choice_sources(choice: &ChatChoiceStream, spans: Spans) -> ChoiceSources {
    let mut sources = ChoiceSources {
        index: choice.index,
        ..ChoiceSources::default()
    };
    if choice.delta.content.is_some() {
        sources.content = spans;
    } else {
        sources.reasoning = spans;
    }
    sources
}

/// The text of a raw content delta and of its upstream reasoning.
fn text_len(content: Option<&ChatCompletionMessageContent>) -> usize {
    match content {
        Some(ChatCompletionMessageContent::Text(text)) => text.len(),
        _ => 0,
    }
}

fn take_qwen_reasoning_response<I: UnifiedStreamItem>(
    record: &mut ChoiceRecord<I>,
) -> Box<(I::Chunk, I::Annotation)> {
    let pending = record
        .pending_qwen_opener_responses
        .pop_front()
        .expect("queued reasoning event must have a source response");
    let mut data = pending.data;
    let mut source_reasoning = pending.choice;
    source_reasoning.delta.reasoning_content = Some(pending.text);
    source_reasoning.finish_reason = None;
    if let Some(slot) = data.sources_mut() {
        *slot = vec![ChoiceSources {
            index: source_reasoning.index,
            content: Vec::new(),
            reasoning: pending.spans,
        }];
    }
    data.inner_mut().choices = vec![source_reasoning];
    Box::new((data, pending.annotation))
}

fn qwen_chunks_for_choices<I: UnifiedStreamItem>(
    state: &ChoiceState,
    record: &mut ChoiceRecord<I>,
    choices: Vec<ChatChoiceStream>,
    sources: Option<Vec<ChoiceSources>>,
) -> Vec<FinishedStreamChunk<I>> {
    let mut sources = sources.map(Vec::into_iter);
    choices
        .into_iter()
        .enumerate()
        .map(|(position, choice)| {
            let choice_sources = sources.as_mut().map(|all| all.next().unwrap_or_default());
            if state.qwen_reasoning_choice_positions.contains(&position) {
                FinishedStreamChunk::Source(take_qwen_reasoning_response(record))
            } else {
                FinishedStreamChunk::choice(choice, choice_sources)
            }
        })
        .collect()
}

impl<I: UnifiedStreamItem> ChoiceRecord<I> {
    fn normalize_finish_reason(&self, finish_reason: Option<FinishReason>) -> Option<FinishReason> {
        normalize_tool_finish_reason(
            finish_reason,
            self.tool_emitted,
            self.incomplete_native_call
                || !self.pending_tool_calls.is_empty()
                || !self.pending_parsed_tool_calls.is_empty()
                || has_incomplete_call_evidence(&self.parser_call_completion),
        )
    }

    fn remember_state(&mut self, state: &ChoiceState) {
        self.incomplete_native_call |= state.incomplete_native_call;
        self.opened_calls.extend(state.opened_calls.iter().copied());
        self.tool_ids_emitted
            .extend(state.tool_ids_emitted.iter().copied());
        self.tool_types_emitted
            .extend(state.tool_types_emitted.iter().copied());
        self.parser_tool_indices
            .extend(state.parser_tool_indices.iter().map(|(k, v)| (*k, *v)));
        self.passthrough_tool_indices
            .extend(state.passthrough_tool_indices.iter().map(|(k, v)| (*k, *v)));
        self.next_tool_index = self.next_tool_index.max(state.next_tool_index);
        self.pending_tool_calls
            .clone_from(&state.pending_tool_calls);
        for (&index, &complete) in &state.parser_call_completion {
            self.parser_call_completion
                .entry(index)
                .and_modify(|remembered| *remembered |= complete)
                .or_insert(complete);
        }
        self.pending_parsed_tool_calls
            .clone_from(&state.pending_parsed_tool_calls);
        self.tool_emitted |= state.tool_emitted();
    }
}

/// Finish every choice that never received a terminating chunk, in index order.
///
/// A choice can reach this point two ways: still holding a live `ChoiceState` (the
/// common case — nothing special happened, it just never got an explicit terminal),
/// or "history-only" — its `ChoiceState` was removed by an already-parsed detour and
/// never rebuilt, so all that remains is its `ChoiceRecord`. The second case has
/// nothing left to flush, but a choice with a completed tool call still needs a synthesized
/// `ToolCalls` terminal so a strict client doesn't hang waiting for one; a choice that
/// never called a tool gets no synthetic chunk unless a suppressed GLM native
/// call needs a `Length` terminal to expose truncation.
fn finish_unterminated_choices<I: UnifiedStreamItem>(
    states: &mut HashMap<u32, ChoiceState>,
    records: &mut HashMap<u32, ChoiceRecord<I>>,
    family: &str,
    parsers_v2_selected: bool,
    recording: bool,
) -> Vec<FinishedStreamChunk<I>> {
    let mut indices: Vec<u32> = states
        .keys()
        .copied()
        .chain(records.keys().copied())
        .collect();
    indices.sort_unstable();
    indices.dedup();

    let mut chunks = Vec::new();
    for index in indices {
        if records.get(&index).is_some_and(|record| record.finished) {
            continue;
        }
        let record = records.entry(index).or_default();
        record.finished = true;
        let base = empty_choice(index);
        match states.get_mut(&index) {
            Some(state) => {
                let deltas = state.finish();
                // Native completion evidence remains authoritative; backend-only calls
                // use the family policy so default-on parsers retain their EOF contract.
                let mut flushed = state.choices_for(&base, deltas, true, None);
                let mut flushed_sources = state.take_choice_sources();
                state.resolve_complete_parsed_tool_calls_at_eof();
                if let Some(reason) = state.unterminated_finish_reason() {
                    if let Some(last_position) = flushed.len().checked_sub(1) {
                        if state
                            .qwen_reasoning_choice_positions
                            .contains(&last_position)
                        {
                            let mut terminal = empty_choice(index);
                            terminal.finish_reason = Some(reason);
                            flushed.push(terminal);
                            if let Some(sources) = &mut flushed_sources {
                                sources.push(ChoiceSources {
                                    index,
                                    ..ChoiceSources::default()
                                });
                            }
                        } else {
                            flushed[last_position].finish_reason = Some(reason);
                        }
                    } else {
                        flushed.extend(state.choices_for(&base, Vec::new(), true, Some(reason)));
                        if let (Some(sources), Some(more)) =
                            (&mut flushed_sources, state.take_choice_sources())
                        {
                            sources.extend(more);
                        }
                    }
                }
                chunks.extend(qwen_chunks_for_choices(
                    state,
                    record,
                    flushed,
                    flushed_sources,
                ));
            }
            None => {
                let pending_whitespace =
                    std::mem::take(&mut record.pending_guided_prefill_whitespace);
                if !pending_whitespace.is_empty() {
                    let mut choice = empty_choice(index);
                    choice.delta.reasoning_content = Some(pending_whitespace);
                    chunks.push(FinishedStreamChunk::Choice(choice));
                }
                while !record.pending_qwen_opener_responses.is_empty() {
                    chunks.push(FinishedStreamChunk::Source(take_qwen_reasoning_response(
                        record,
                    )));
                }
                resolve_complete_parsed_tool_calls_at_eof(
                    &mut record.pending_parsed_tool_calls,
                    family,
                    parsers_v2_selected,
                );
                if let Some(reason @ (FinishReason::Length | FinishReason::ToolCalls)) =
                    record.normalize_finish_reason(Some(FinishReason::Stop))
                {
                    let mut choice = base;
                    choice.finish_reason = Some(reason);
                    let sources = recording.then(|| ChoiceSources {
                        index,
                        ..ChoiceSources::default()
                    });
                    chunks.push(FinishedStreamChunk::choice(choice, sources));
                }
            }
        }
    }
    chunks
}

// Live and history-only choices must expose the same unfinished-call outcome.
fn normalize_tool_finish_reason(
    finish_reason: Option<FinishReason>,
    tool_emitted: bool,
    incomplete: bool,
) -> Option<FinishReason> {
    match finish_reason {
        Some(FinishReason::Stop | FinishReason::ToolCalls) if incomplete => {
            Some(FinishReason::Length)
        }
        Some(FinishReason::Stop) if tool_emitted => Some(FinishReason::ToolCalls),
        other => other,
    }
}

fn has_incomplete_call_evidence(completion: &HashMap<usize, bool>) -> bool {
    completion.values().any(|complete| !complete)
}

fn merge_passthrough_completion(
    completion: &mut HashMap<usize, bool>,
    indices: &HashMap<usize, usize>,
    evidence: &[ToolCallCompletion],
    choice_index: u32,
) {
    for evidence in evidence
        .iter()
        .filter(|evidence| evidence.choice_index == choice_index)
    {
        if let Some(index) = indices.get(&(evidence.tool_index as usize)) {
            completion
                .entry(*index)
                .and_modify(|complete| *complete |= evidence.complete)
                .or_insert(evidence.complete);
        }
    }
}

fn normalize_completion_evidence<I: UnifiedStreamItem>(
    evidence: &mut ToolCallCompletion,
    states: &HashMap<u32, ChoiceState>,
    records: &HashMap<u32, ChoiceRecord<I>>,
) {
    let index = evidence.tool_index as usize;
    let normalized = states
        .get(&evidence.choice_index)
        .and_then(|state| state.passthrough_tool_indices.get(&index))
        .or_else(|| {
            records
                .get(&evidence.choice_index)
                .and_then(|record| record.passthrough_tool_indices.get(&index))
        });
    if let Some(index) = normalized {
        evidence.tool_index = *index as u32;
    }
}

fn normalized_source_response<I: UnifiedStreamItem>(
    (mut data, annotation): (I::Chunk, I::Annotation),
    states: &HashMap<u32, ChoiceState>,
    records: &HashMap<u32, ChoiceRecord<I>>,
) -> I {
    for evidence in data.tool_call_completion_mut() {
        normalize_completion_evidence(evidence, states, records);
    }
    I::from_parts(data, annotation)
}

fn attach_call_completion<I: UnifiedStreamItem>(
    data: &mut I::Chunk,
    choice: &ChatChoiceStream,
    states: &HashMap<u32, ChoiceState>,
    records: &HashMap<u32, ChoiceRecord<I>>,
) {
    // Source evidence may refer to a choice with no delta in this chunk. Preserve
    // it on one child rather than filtering it to the child's choice.
    for evidence in data.tool_call_completion_mut() {
        normalize_completion_evidence(evidence, states, records);
    }
    let state = states.get(&choice.index);
    let record = records.get(&choice.index);
    let mut indices: HashSet<usize> = choice
        .delta
        .tool_calls
        .iter()
        .flatten()
        .map(|call| call.index as usize)
        .collect();
    if choice.finish_reason.is_some() {
        indices.extend(
            state
                .into_iter()
                .flat_map(|state| state.parser_call_completion.keys().copied()),
        );
        indices.extend(
            record
                .into_iter()
                .flat_map(|record| record.parser_call_completion.keys().copied()),
        );
    }
    let mut indices: Vec<_> = indices.into_iter().collect();
    indices.sort_unstable();
    for index in indices {
        let complete = state
            .and_then(|state| state.parser_call_completion.get(&index))
            .or_else(|| record.and_then(|record| record.parser_call_completion.get(&index)))
            .copied();
        if let Some(complete) = complete {
            if let Some(evidence) = data.tool_call_completion_mut().iter_mut().find(|evidence| {
                evidence.choice_index == choice.index && evidence.tool_index == index as u32
            }) {
                evidence.complete |= complete;
            } else {
                data.tool_call_completion_mut().push(ToolCallCompletion {
                    choice_index: choice.index,
                    tool_index: index as u32,
                    complete,
                });
            }
        }
    }
}

/// Wrap one rewritten choice in a response built from `template`.
///
/// Usage, nvext and metrics are cleared: they belong to the chunk that carried them,
/// and repeating them on a synthesized chunk would double-count.
fn response_with_choice<I: UnifiedStreamItem>(
    template: &I::Chunk,
    choice: ChatChoiceStream,
    sources: Option<ChoiceSources>,
    states: &HashMap<u32, ChoiceState>,
    records: &HashMap<u32, ChoiceRecord<I>>,
) -> I {
    let mut data = template.clone();
    data.scrub_synthetic();
    write_slot(&mut data, sources.into_iter().collect());
    attach_call_completion(&mut data, &choice, states, records);
    data.inner_mut().choices = vec![choice];
    I::from_parts(data, I::Annotation::default())
}

/// Whether a choice arrived already parsed by something upstream and must be passed
/// through untouched rather than re-parsed.
fn already_parsed(choice: &ChatChoiceStream) -> bool {
    let has_raw_text = matches!(
        choice.delta.content,
        Some(ChatCompletionMessageContent::Text(_))
    );
    matches!(
        choice.delta.content,
        Some(ChatCompletionMessageContent::Parts(_))
    ) || (!has_raw_text
        && (choice.delta.tool_calls.is_some()
            || choice.delta.function_call.is_some()
            || choice.delta.reasoning_content.is_some()))
}

/// Streaming path: one unified parser per response choice, replacing both the reasoning
/// parser and the tool-call jail for this request.
///
/// `uses_tool_call_structural_tag` reports that the backend was given a structural tag
/// constraining generation to the family's NATIVE grammar. When it is set the output is
/// native markup no matter what `tool_choice` says, so `tool_choice` is not consulted;
/// this path only parses guided JSON, it never builds the grammar that produces it.
#[cfg(test)]
pub(crate) fn apply_stream<S, I>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    prefill: UnifiedParserStartingState,
    family: &'static str,
) -> impl Stream<Item = I> + Send
where
    S: Stream<Item = I> + Send + 'static,
    I: UnifiedStreamItem,
{
    let guided_tool_constraint = if uses_tool_call_structural_tag {
        GuidedToolConstraint::StructuralTag
    } else {
        match tool_choice {
            Some(ChatCompletionToolChoiceOption::Named(named)) => {
                GuidedToolConstraint::GuidedJsonNamed {
                    tool_name: named.function.name,
                }
            }
            Some(ChatCompletionToolChoiceOption::Required) => {
                GuidedToolConstraint::GuidedJsonRequired
            }
            None
            | Some(ChatCompletionToolChoiceOption::Auto)
            | Some(ChatCompletionToolChoiceOption::None) => GuidedToolConstraint::None,
        }
    };
    apply_stream_with_constraint(
        stream_in,
        tool_definitions,
        guided_tool_constraint,
        prefill,
        family,
        // Streaming released, matching the production default when the rollback lever
        // is unset. Tests that exercise the rolled-back contract call
        // `apply_stream_with_constraint` directly with `false`.
        true,
    )
}

/// `guided_streaming` is the request's already-made guided tool-call streaming
/// decision (`OpenAIPreprocessor::guided_tool_streaming_release`), not a second
/// derivation of it: `true` releases each call as soon as its name and argument
/// object opener are unambiguous, `false` buffers every call to completion. It is
/// inert unless `guided_tool_constraint` installed guided JSON.
fn resolve_guided_prefix(
    index: u32,
    probe: JsonPrefillProbe,
    policy: UnifiedRequestPolicy,
    prefill: UnifiedParserStartingState,
) -> Option<ChatChoiceStream> {
    let decision = probe.decision(true);
    if probe.text.is_empty() {
        return None;
    }
    let mut choice = empty_choice(index);
    if policy.structured_response
        && (prefill == UnifiedParserStartingState::None
            || decision == Some(UnifiedParserStartingState::None))
    {
        choice.delta.content = Some(ChatCompletionMessageContent::Text(probe.text));
    } else {
        choice.delta.reasoning_content = Some(probe.text);
    }
    Some(choice)
}

fn flush_undecided_guided_prefixes(
    prefixes: &mut HashMap<u32, JsonPrefillProbe>,
    prefix_spans: &mut HashMap<u32, Spans>,
    policy: UnifiedRequestPolicy,
    prefill: UnifiedParserStartingState,
) -> Vec<(ChatChoiceStream, ChoiceSources)> {
    prefixes
        .drain()
        .filter_map(|(index, probe)| {
            let spans = prefix_spans.remove(&index).unwrap_or_default();
            resolve_guided_prefix(index, probe, policy, prefill).map(|choice| {
                let sources = probe_choice_sources(&choice, spans);
                (choice, sources)
            })
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn apply_stream_with_constraint<S, I>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    guided_tool_constraint: GuidedToolConstraint,
    prefill: impl Into<StreamPrefill>,
    family: &'static str,
    guided_streaming: bool,
) -> impl Stream<Item = I> + Send
where
    S: Stream<Item = I> + Send + 'static,
    I: UnifiedStreamItem,
{
    apply_stream_with_policy(
        stream_in,
        tool_definitions,
        guided_tool_constraint,
        prefill,
        family,
        guided_streaming,
        UnifiedRequestPolicy::default(),
    )
}

pub fn apply_stream_with_policy<S, I>(
    stream_in: S,
    tool_definitions: Option<Vec<ToolDefinition>>,
    guided_tool_constraint: GuidedToolConstraint,
    prefill: impl Into<StreamPrefill>,
    family: &'static str,
    guided_streaming: bool,
    policy: UnifiedRequestPolicy,
) -> impl Stream<Item = I> + Send
where
    S: Stream<Item = I> + Send + 'static,
    I: UnifiedStreamItem,
{
    let StreamPrefill {
        starting_state: prefill,
        prefix: prefill_prefix,
    } = prefill.into();
    let prefill = policy.starting_state(family, prefill);
    let tools = to_v2_tools(tool_definitions.as_deref());
    stream! {
        let mut undecided_guided_prefix: HashMap<u32, JsonPrefillProbe> = HashMap::new();
        // Ranges beside each probe's held text, while recording.
        let mut undecided_spans: HashMap<u32, Spans> = HashMap::new();
        // Whether the input chunks carry a source slot, so the adapter records.
        // The first data chunk decides; every later one must agree.
        let mut recording = false;
        let mut slot_presence: Option<bool> = None;
        let mut states: HashMap<u32, ChoiceState> = HashMap::new();
        // Per-choice bookkeeping that must outlive a single `ChoiceState`: a
        // `ChoiceState` is dropped whenever an already-parsed chunk interrupts a
        // choice (its buffered parser state would be wrong to keep across the
        // detour), but whether that choice ever called a tool and whether its
        // terminal has already gone out must still be known afterward — by a
        // resumed raw run, an already-parsed terminal, or the EOF backstop below,
        // even when no `ChoiceState` for that index exists any more (or never did:
        // an index whose first-ever chunk is already-parsed never gets one either).
        let mut records: HashMap<u32, ChoiceRecord<I>> = HashMap::new();
        // Last data response with its choices cleared, so an end-of-stream flush has an
        // envelope (id, model, created) to attach synthesized chunks to.
        let mut template: Option<I::Chunk> = None;

        tokio::pin!(stream_in);

        while let Some(response) = stream_in.next().await {
            if response.is_error() {
                yield response;
                return;
            }
            let (mut chat, mut annotation) = match response.into_parts() {
                Ok(parts) => parts,
                Err(response) => {
                    // Non-data annotations (errors, comments) pass through untouched.
                    yield response;
                    continue;
                }
            };

            // The input slot dates the input text. Take it out, so no chunk built
            // from this one repeats it.
            let has_slot = chat.sources_mut().map(std::mem::take);
            if *slot_presence.get_or_insert(has_slot.is_some()) != has_slot.is_some() {
                // A parser built without recording cannot date its held text.
                yield I::from_error("the input chunks change whether they carry a source slot".to_string());
                return;
            }
            recording = has_slot.is_some();
            let mut input_sources = has_slot.unwrap_or_default();

            {
                let mut next = chat.clone();
                next.scrub_synthetic();
                next.inner_mut().choices.clear();
                template = Some(next);
            }

            for evidence in chat.tool_call_completion() {
                if let Some(state) = states.get_mut(&evidence.choice_index) {
                    merge_passthrough_completion(&mut state.parser_call_completion, &state.passthrough_tool_indices, std::slice::from_ref(evidence), evidence.choice_index);
                }
                if let Some(record) = records.get_mut(&evidence.choice_index) {
                    merge_passthrough_completion(&mut record.parser_call_completion, &record.passthrough_tool_indices, std::slice::from_ref(evidence), evidence.choice_index);
                }
            }

            // A finished choice cannot reopen its parser; preserve envelope metadata below.
            chat.inner_mut().choices.retain(|choice| {
                !records.get(&choice.index).is_some_and(|record| record.finished)
            });

            if chat.inner().choices.is_empty() {
                // A metadata-only chunk. OpenAI stream ordering requires every choice's
                // terminal finish_reason to precede it, so flush first.
                if chat.inner().usage.is_some()
                    && let Some(template) = &template {
                    for (choice, sources) in flush_undecided_guided_prefixes(&mut undecided_guided_prefix, &mut undecided_spans, policy, prefill) {
                        yield response_with_choice(template, choice, recording.then_some(sources), &states, &records);
                    }
                    let finished = finish_unterminated_choices(&mut states, &mut records, family, policy.parsers_v2_selected, recording);
                    // Check before emitting: a failed choice's flushed text is undated.
                    if let Some(error) = states.values_mut().find_map(ChoiceState::take_recording_error) {
                        yield I::from_error(error);
                        return;
                    }
                    for output in finished {
                        match output {
                            FinishedStreamChunk::Choice(choice) => {
                                yield response_with_choice(template, choice, None, &states, &records);
                            }
                            FinishedStreamChunk::SourcedChoice(sourced) => {
                                let (choice, sources) = *sourced;
                                yield response_with_choice(template, choice, Some(sources), &states, &records);
                            }
                            FinishedStreamChunk::Source(response) => yield normalized_source_response(*response, &states, &records),
                        }
                    }
                }
                for evidence in chat.tool_call_completion_mut() {
                    normalize_completion_evidence(evidence, &states, &records);
                }
                write_slot(&mut chat, Vec::new());
                yield I::from_parts(chat, annotation);
                continue;
            }

            let originals = std::mem::take(&mut chat.inner_mut().choices);
            let mut emitted: Vec<ChatChoiceStream> = Vec::new();
            let mut emitted_sources: Option<Vec<ChoiceSources>> = recording.then(Vec::new);
            let mut deferred_qwen_responses = Vec::new();
            let mut qwen_source_metadata_deferred = false;
            for mut original in originals {
                // This choice's input sources, checked against its text.
                let mut input = match input_sources.iter().position(|sources| sources.index == original.index) {
                    Some(position) => input_sources.swap_remove(position),
                    None => ChoiceSources { index: original.index, ..ChoiceSources::default() },
                };
                if recording
                    && (spans_len(&input.content) != text_len(original.delta.content.as_ref())
                        || spans_len(&input.reasoning)
                            != original.delta.reasoning_content.as_ref().map_or(0, String::len))
                {
                    yield I::from_error(format!(
                        "the sources of choice {} do not cover its text",
                        original.index
                    ));
                    return;
                }
                if already_parsed(&original) {
                    if let Some(prefix) = undecided_guided_prefix.remove(&original.index)
                        && let Some(choice) = resolve_guided_prefix(original.index, prefix, policy, prefill) {
                        let spans = undecided_spans.remove(&original.index).unwrap_or_default();
                        let sources = probe_choice_sources(&choice, spans);
                        push_emitted(&mut emitted, &mut emitted_sources, choice, sources);
                    }
                    // Record this index exists even if no `ChoiceState` is ever
                    // built for it (e.g. its first-ever chunk is already-parsed),
                    // so it is not invisible to `finish_unterminated_choices`.
                    let record = records.entry(original.index).or_default();
                    let pending_whitespace =
                        std::mem::take(&mut record.pending_guided_prefill_whitespace);
                    if !pending_whitespace.is_empty() {
                        if let Some(reasoning) = original.delta.reasoning_content.as_mut() {
                            reasoning.insert_str(0, &pending_whitespace);
                        } else {
                            let mut deferred_reasoning = empty_choice(original.index);
                            deferred_reasoning.delta.reasoning_content =
                                Some(pending_whitespace);
                            let sources = ChoiceSources { index: original.index, ..ChoiceSources::default() };
                            push_emitted(&mut emitted, &mut emitted_sources, deferred_reasoning, sources);
                        }
                    }
                    // Any raw run resuming this choice after this point must rebuild
                    // starting at `Response`, not the outer request-level `prefill` —
                    // see the `Vacant` arm below and the field doc on `ChoiceRecord`.
                    record.detoured = true;
                    if let Some(state) = states.get(&original.index) {
                        record.remember_state(state);
                    }
                    let parsed_calls_are_terminal =
                        finish_reason_closes_tool_calls(original.finish_reason.as_ref());
                    // An already-parsed chunk carries its tool calls verbatim in its
                    // own delta rather than through a `ChoiceState`, so that history
                    // has to be observed here directly — a state may never exist for
                    // this index at all (its very first chunk can be already-parsed).
                    if let Some(calls) = original
                        .delta
                        .tool_calls
                        .as_mut()
                        .filter(|calls| !calls.is_empty())
                    {
                        record.tool_emitted = true;
                        for call in calls {
                            let tool_index = normalize_passthrough_tool_call(
                                call,
                                &mut record.passthrough_tool_indices,
                                &mut record.opened_calls,
                                &mut record.tool_ids_emitted,
                                &mut record.tool_types_emitted,
                                &mut record.next_tool_index,
                            );
                            if parsed_calls_are_terminal {
                                record.pending_tool_calls.remove(&tool_index);
                            }
                            append_parsed_tool_call_arguments(
                                &mut record.pending_parsed_tool_calls,
                                tool_index,
                                call,
                            );
                        }
                    }
                    merge_passthrough_completion(
                        &mut record.parser_call_completion,
                        &record.passthrough_tool_indices,
                        chat.tool_call_completion(),
                        original.index,
                    );
                    if parsed_calls_are_terminal {
                        resolve_complete_parsed_tool_calls_at_terminal(
                            &mut record.pending_parsed_tool_calls,
                            family,
                            original.finish_reason == Some(FinishReason::ToolCalls),
                            policy.parsers_v2_selected,
                        );
                    }
                    let reported_tool_indices: HashSet<usize> = if parsed_calls_are_terminal {
                        original
                            .delta
                            .tool_calls
                            .as_ref()
                            .into_iter()
                            .flatten()
                            .map(|call| call.index as usize)
                            .collect()
                    } else {
                        HashSet::new()
                    };
                    let preserve_reasoning_only_state = original.finish_reason.is_none()
                        && original.delta.reasoning_content.is_some()
                        && original.delta.content.is_none()
                        && original.delta.tool_calls.is_none()
                        && original.delta.function_call.is_none()
                        && original.delta.refusal.is_none()
                        && states
                            .get(&original.index)
                            .is_some_and(ChoiceState::should_preserve_reasoning_only_state);
                    let queue_qwen_opener_reasoning = preserve_reasoning_only_state
                        && original
                            .delta
                            .reasoning_content
                            .as_ref()
                            .is_some_and(|reasoning| !reasoning.is_empty())
                        && family == QWEN3_UNIFIED_FAMILY
                        && states.get(&original.index).is_some_and(|state| {
                            state.qwen_native_lexical_state.mode
                                == QwenNativeLexicalMode::OutsideBlock
                                && !state.pending_qwen_opener_text.is_empty()
                        });
                    if preserve_reasoning_only_state {
                        // Reasoning is an ordered event between generated bytes, not a
                        // replacement parser input. Keep the parser state alive so a
                        // later raw chunk can finish the same native call or guided
                        // argument object. Kimi may still be buffering an open JSON
                        // string before its first tool-call event.
                        if queue_qwen_opener_reasoning {
                            let reasoning = original
                                .delta
                                .reasoning_content
                                .take()
                                .expect("reasoning-only chunk has reasoning content");
                            let reasoning_spans = std::mem::take(&mut input.reasoning);
                            states
                                .get_mut(&original.index)
                                .expect("preserved parser state must still exist")
                                .queue_qwen_opener_reasoning_sourced(reasoning.clone(), reasoning_spans.clone());
                            let mut source_data = chat.clone();
                            source_data.inner_mut().choices.clear();
                            let keep_source_metadata = !qwen_source_metadata_deferred;
                            qwen_source_metadata_deferred = true;
                            if !keep_source_metadata {
                                source_data.tool_call_completion_mut().clear();
                                source_data.inner_mut().usage = None;
                                source_data.clear_nvext();
                                source_data.clear_llm_metrics();
                            }
                            record.pending_qwen_opener_responses.push_back(
                                PendingQwenReasoningResponse {
                                    text: reasoning,
                                    spans: reasoning_spans,
                                    choice: original.clone(),
                                    data: source_data,
                                    annotation: if keep_source_metadata {
                                        annotation.clone()
                                    } else {
                                        I::Annotation::default()
                                    },
                                },
                            );
                        }
                    } else if let Some(mut state) = states.remove(&original.index) {
                        // A prior raw chunk on this choice may have emitted a tool
                        // call before this already-parsed terminal replaced it; that
                        // history lives only in the state being discarded here, so
                        // fold it into the record before it's gone (also needed by a
                        // later chunk or the EOF backstop after this `ChoiceState` is
                        // gone). Normalize against `record.tool_emitted`, not the
                        // state's own flag: this chunk's own `delta.tool_calls` (set
                        // into `record` above) can carry a tool call the state itself
                        // never saw, e.g. an already-parsed terminal chunk that is
                        // itself the first and only tool-call signal for this choice.
                        if state.tool_emitted() {
                            record.tool_emitted = true;
                        }
                        record
                            .pending_tool_calls
                            .extend(
                                state
                                    .pending_tool_calls
                                    .iter()
                                    .filter(|index| !reported_tool_indices.contains(*index))
                                    .copied(),
                            );
                        record.opened_calls.extend(state.opened_calls.iter().copied());
                        record
                            .tool_ids_emitted
                            .extend(state.tool_ids_emitted.iter().copied());
                        record
                            .tool_types_emitted
                            .extend(state.tool_types_emitted.iter().copied());
                        record
                            .parser_tool_indices
                            .extend(state.parser_tool_indices.iter().map(|(k, v)| (*k, *v)));
                        record.passthrough_tool_indices.extend(
                            state
                                .passthrough_tool_indices
                                .iter()
                                .map(|(k, v)| (*k, *v)),
                        );
                        record.next_tool_index =
                            record.next_tool_index.max(state.next_tool_index);
                        state
                            .pending_tool_calls
                            .retain(|index| !reported_tool_indices.contains(index));
                        state
                            .pending_tool_calls
                            .extend(record.pending_tool_calls.iter().copied());
                        state
                            .pending_parsed_tool_calls
                            .clone_from(&record.pending_parsed_tool_calls);
                        let deltas = state.finish();
                        let flushed = state.choices_for(
                            &empty_choice(original.index),
                            deltas,
                            true,
                            None,
                        );
                        let flushed_sources = state.take_choice_sources();
                        if let Some(error) = state.take_recording_error() {
                            yield I::from_error(error);
                            return;
                        }
                        for output in qwen_chunks_for_choices(&state, record, flushed, flushed_sources) {
                            match output {
                                FinishedStreamChunk::Source(response) => {
                                    deferred_qwen_responses.push((emitted.len(), response));
                                }
                                FinishedStreamChunk::Choice(choice) => emitted.push(choice),
                                FinishedStreamChunk::SourcedChoice(sourced) => {
                                    let (choice, sources) = *sourced;
                                    push_emitted(&mut emitted, &mut emitted_sources, choice, sources);
                                }
                            }
                        }
                        record.remember_state(&state);
                    }
                    original.finish_reason = record.normalize_finish_reason(original.finish_reason);
                    if original.finish_reason.is_some() {
                        record.finished = true;
                    }
                    if !queue_qwen_opener_reasoning {
                        let sources = ChoiceSources {
                            index: original.index,
                            content: Vec::new(),
                            reasoning: input.reasoning,
                        };
                        push_emitted(&mut emitted, &mut emitted_sources, original, sources);
                    }
                    continue;
                }

                // Keep a provisional native call's parser across a reasoning-only
                // detour so its argument tail can finish. If the resumed bytes start
                // a fresh call instead, finish the old parser first; the record keeps
                // its emitted index pending while the next parser gets a new index.
                // Qwen's lexical state distinguishes a new block marker from the same
                // text inside a parameter value; JSON grammars use their own quoted
                // strings and may legally contain another parser marker there.
                let new_native_call_after_detour = records
                    .get(&original.index)
                    .is_some_and(|record| record.detoured)
                    && family == QWEN3_UNIFIED_FAMILY
                    && states
                        .get(&original.index)
                        .is_some_and(|state| {
                            matches!(&state.tool_output_mode, UnifiedToolOutputMode::Native)
                                && !state.pending_tool_calls.is_empty()
                                && state.qwen_resumed_marker_can_start_call()
                        })
                    && match original.delta.content.as_ref() {
                        Some(ChatCompletionMessageContent::Text(text)) => {
                            first_unquoted_structural_tool_call_marker(
                                text,
                                QWEN3_TOOL_CALL_PARSER,
                            )
                                .is_some_and(|marker| text[..marker].trim().is_empty())
                        }
                        _ => false,
                    };
                if new_native_call_after_detour
                    && let Some(mut state) = states.remove(&original.index)
                {
                        let deltas = state.finish();
                        let flushed = state.choices_for(
                            &empty_choice(original.index), deltas, true, None,
                        );
                        let flushed_sources = state.take_choice_sources();
                        if let Some(error) = state.take_recording_error() {
                            yield I::from_error(error);
                            return;
                        }
                        let record = records.entry(original.index).or_default();
                        for output in qwen_chunks_for_choices(&state, record, flushed, flushed_sources) {
                            match output {
                                FinishedStreamChunk::Source(response) => {
                                    deferred_qwen_responses.push((emitted.len(), response));
                                }
                                FinishedStreamChunk::Choice(choice) => emitted.push(choice),
                                FinishedStreamChunk::SourcedChoice(sourced) => {
                                    let (choice, sources) = *sourced;
                                    push_emitted(&mut emitted, &mut emitted_sources, choice, sources);
                                }
                            }
                        }
                        record.remember_state(&state);
                }

                let mut detected_prefill = prefill;
                if !states.contains_key(&original.index)
                    && (prefill == UnifiedParserStartingState::Reasoning
                        || policy.structured_response && prefill == UnifiedParserStartingState::None)
                    && policy.inspects_json_prefill(&guided_tool_constraint)
                    && !records.get(&original.index).is_some_and(|record| record.detoured)
                {
                    let text = match original.delta.content.as_ref() {
                        Some(ChatCompletionMessageContent::Text(text)) => text.as_str(),
                        _ => "",
                    };
                    let mut probe = undecided_guided_prefix.remove(&original.index).unwrap_or_default();
                    probe.push(text);
                    let mut probe_spans = undecided_spans.remove(&original.index).unwrap_or_default();
                    append_spans(&mut probe_spans, std::mem::take(&mut input.content));
                    let probe_decision = if prefill == UnifiedParserStartingState::None {
                        // Without a prompt-opened channel, non-JSON may contain an explicit
                        // reasoning opener. Let the native parser recognize it.
                        probe.unopened_channel_decision(original.finish_reason.is_some())
                    } else {
                        match probe.decision_for_family(
                            family,
                            original.finish_reason.is_some(),
                            text,
                        ) {
                            Ok(decision) => decision,
                            Err(error) => {
                                tracing::warn!(
                                    error = %error,
                                    family,
                                    choice = original.index,
                                    "structured response prefill detection failed; treating buffered output as reasoning"
                                );
                                Some(UnifiedParserStartingState::Reasoning)
                            }
                        }
                    };
                    let awaiting_reasoning_boundary = prefill == UnifiedParserStartingState::Reasoning
                        && policy.structured_response
                        && !policy.reasoning_disabled
                        && probe_decision == Some(UnifiedParserStartingState::None)
                        && original.finish_reason.is_none();
                    if (probe.text.trim().is_empty() && original.finish_reason.is_none())
                        || policy.structured_response
                            && (probe_decision.is_none() || awaiting_reasoning_boundary)
                    {
                        undecided_guided_prefix.insert(original.index, probe);
                        if recording {
                            undecided_spans.insert(original.index, probe_spans);
                        }
                        original.delta.content = None;
                        let sources = ChoiceSources {
                            index: original.index,
                            content: Vec::new(),
                            reasoning: input.reasoning,
                        };
                        push_emitted(&mut emitted, &mut emitted_sources, original, sources);
                        continue;
                    }
                    if prefill == UnifiedParserStartingState::None
                        && probe_decision == Some(UnifiedParserStartingState::Response)
                    {
                        detected_prefill = UnifiedParserStartingState::Response;
                    }
                    original.delta.content = Some(ChatCompletionMessageContent::Text(probe.text));
                    input.content = probe_spans;
                }

                let state = match states.entry(original.index) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let mode = tool_output_mode(&guided_tool_constraint);
                        // `Vacant` covers both a brand-new choice AND one whose
                        // `ChoiceState` was just dropped by an already-parsed detour.
                        // Only the former may start at the request-level `prefill`
                        // (what the PROMPT opened); a choice that has already been
                        // through a detour has, by definition, already emitted
                        // structured output, so its reasoning phase (if any) already
                        // concluded and a resumed raw run must start at `Response`.
                        let choice_prefill = if records
                            .get(&original.index)
                            .is_some_and(|record| record.detoured)
                        {
                            UnifiedParserStartingState::Response
                        } else if prefill == UnifiedParserStartingState::Reasoning
                            && policy.inspects_json_prefill(&guided_tool_constraint)
                        {
                            if policy.structured_response {
                                match original.delta.content.as_ref() {
                                    Some(ChatCompletionMessageContent::Text(text)) => match structured_json_response_prefill(family, text, original.finish_reason.is_some()) {
                                        Ok(Some(state)) => state,
                                        Ok(None) => UnifiedParserStartingState::Reasoning,
                                        Err(error) => {
                                            tracing::warn!(
                                                error = %error,
                                                family,
                                                choice = original.index,
                                                "structured response prefill detection failed; treating output as reasoning"
                                            );
                                            UnifiedParserStartingState::Reasoning
                                        }
                                    },
                                    _ => UnifiedParserStartingState::Reasoning,
                                }
                            } else {
                                bare_guided_json_prefill(original.delta.content.as_ref())
                            }
                        } else {
                            detected_prefill
                        };
                        let pending_whitespace = records
                            .get_mut(&original.index)
                            .map(|record| {
                                std::mem::take(
                                    &mut record.pending_guided_prefill_whitespace,
                                )
                            })
                            .unwrap_or_default();
                        if choice_prefill == UnifiedParserStartingState::Reasoning
                            && !pending_whitespace.is_empty()
                        {
                            match original.delta.content.as_mut() {
                                Some(ChatCompletionMessageContent::Text(text)) => {
                                    let mut combined = pending_whitespace;
                                    combined.push_str(text);
                                    *text = combined;
                                }
                                None => {
                                    original.delta.content = Some(
                                        ChatCompletionMessageContent::Text(pending_whitespace),
                                    );
                                }
                                Some(ChatCompletionMessageContent::Parts(_)) => {
                                    unreachable!("already parsed parts bypass raw parser setup")
                                }
                            }
                        }
                        let choice_prefill = StreamPrefill {
                            starting_state: choice_prefill,
                            prefix: (choice_prefill == UnifiedParserStartingState::Reasoning).then_some(prefill_prefix).flatten(),
                        };
                        match ChoiceState::build(family, &tools, choice_prefill, mode, guided_streaming, policy.parsers_v2_selected, recording) {
                            Ok(mut state) => {
                                // A raw run resuming after an already-parsed detour
                                // gets a brand-new parser instance; seed it with any
                                // tool-call history from before the gap so a Stop at
                                // the end of this run still normalizes correctly.
                                if records
                                    .get(&original.index)
                                    .is_some_and(|record| record.tool_emitted)
                                {
                                    state.mark_tool_emitted();
                                }
                                if let Some(record) = records.get(&original.index) {
                                    state.incomplete_native_call = record.incomplete_native_call;
                                    state.tool_index_offset = record.next_tool_index;
                                    state.next_tool_index = record.next_tool_index;
                                    state.opened_calls.extend(record.opened_calls.iter().copied());
                                    state
                                        .tool_ids_emitted
                                        .extend(record.tool_ids_emitted.iter().copied());
                                    state
                                        .tool_types_emitted
                                        .extend(record.tool_types_emitted.iter().copied());
                                    state
                                        .parser_tool_indices
                                        .extend(record.parser_tool_indices.iter().map(|(k, v)| (*k, *v)));
                                    state.passthrough_tool_indices.extend(
                                        record
                                            .passthrough_tool_indices
                                            .iter()
                                            .map(|(k, v)| (*k, *v)),
                                    );
                                    state.parser_call_completion.clone_from(&record.parser_call_completion);
                                    state
                                        .pending_tool_calls
                                        .extend(record.pending_tool_calls.iter().copied());
                                    state
                                        .pending_parsed_tool_calls
                                        .clone_from(&record.pending_parsed_tool_calls);
                                }
                                entry.insert(state)
                            }
                            Err(ChoiceStateError::Recording(message)) => {
                                // A choice that cannot record must not pass through
                                // with undated text.
                                yield I::from_error(message);
                                return;
                            }
                            Err(ChoiceStateError::Parser(error)) => {
                                tracing::warn!(
                                    error = %error,
                                    family,
                                    choice = original.index,
                                    "unified parser construction failed; passing choice through"
                                );
                                push_emitted(&mut emitted, &mut emitted_sources, original, input);
                                continue;
                            }
                        }
                    }
                };

                let mut deltas = Vec::new();
                if let Some(ChatCompletionMessageContent::Text(text)) =
                    original.delta.content.as_ref()
                {
                    deltas.extend(state.push_sourced(text, std::mem::take(&mut input.content)));
                }
                let terminal = original.finish_reason;
                let already_finished = records
                    .get(&original.index)
                    .is_some_and(|record| record.finished);
                if terminal.is_some() {
                    if !already_finished {
                        deltas.extend(state.finish());
                    }
                    records.entry(original.index).or_default().finished = true;
                }

                state.set_upstream_reasoning(std::mem::take(&mut input.reasoning));
                let mut choices = state.choices_for(&original, deltas, true, terminal);
                let choice_sources = state.take_choice_sources();
                if let Some(error) = state.take_recording_error() {
                    yield I::from_error(error);
                    return;
                }
                merge_passthrough_completion(
                    &mut state.parser_call_completion,
                    &state.passthrough_tool_indices,
                    chat.tool_call_completion(),
                    original.index,
                );
                // Pass-through indices are assigned by choices_for, so completion
                // evidence must be remapped before accepting its terminal decision.
                for choice in &mut choices {
                    choice.finish_reason = state.normalize_finish_reason(choice.finish_reason);
                }
                let mut parsed = Vec::new();
                let mut parsed_sources: Option<Vec<ChoiceSources>> = recording.then(Vec::new);
                let record = records.entry(original.index).or_default();
                for output in qwen_chunks_for_choices(state, record, choices, choice_sources) {
                    match output {
                        FinishedStreamChunk::Source(response) => {
                            deferred_qwen_responses.push((emitted.len() + parsed.len(), response));
                        }
                        FinishedStreamChunk::Choice(choice) => parsed.push(choice),
                        FinishedStreamChunk::SourcedChoice(sourced) => {
                            let (choice, sources) = *sourced;
                            push_emitted(&mut parsed, &mut parsed_sources, choice, sources);
                        }
                    }
                }
                if parsed.is_empty() {
                    // A marker-only chunk produced no deltas. Keep it as an empty
                    // choice so the typed llm_metrics and annotation metadata it
                    // carries still reach the client.
                    let sources = ChoiceSources { index: original.index, ..ChoiceSources::default() };
                    push_emitted(&mut parsed, &mut parsed_sources, empty_choice(original.index), sources);
                }
                emitted.extend(parsed);
                if let (Some(all), Some(more)) = (&mut emitted_sources, parsed_sources) {
                    all.extend(more);
                }
            }

            if qwen_source_metadata_deferred {
                chat.tool_call_completion_mut().clear();
                chat.inner_mut().usage = None;
                chat.clear_nvext();
                chat.clear_llm_metrics();
                annotation = I::Annotation::default();
            }

            if emitted.is_empty() && deferred_qwen_responses.is_empty() {
                continue;
            }

            // One upstream chunk can fan out into several. Usage, nvext and annotation
            // fields stay on the last child. Token metrics stay on one reasoning child
            // when present so the downstream reasoning-usage estimator preserves the
            // source chunk's classification without counting it twice.
            if let Some(sources) = &emitted_sources {
                debug_assert_eq!(sources.len(), emitted.len(), "each emitted choice has sources");
            }
            let mut emitted_sources = emitted_sources.map(Vec::into_iter);
            let last = emitted.len().checked_sub(1);
            let llm_metrics_position = fanout_llm_metrics_position(&emitted);
            let mut deferred_qwen_responses = deferred_qwen_responses.into_iter().peekable();
            for position in 0..=emitted.len() {
                while deferred_qwen_responses
                    .peek()
                    .is_some_and(|(deferred_position, _)| *deferred_position == position)
                {
                    let (_, response) = deferred_qwen_responses
                        .next()
                        .expect("peeked Qwen response should be present");
                    yield normalized_source_response(*response, &states, &records);
                }
                let Some(choice) = emitted.get(position).cloned() else {
                    continue;
                };
                let choice_sources = emitted_sources.as_mut().and_then(Iterator::next);
                let is_last = Some(position) == last;
                let mut data = chat.clone();
                if !is_last {
                    data.tool_call_completion_mut().clear();
                }
                attach_call_completion(&mut data, &choice, &states, &records);
                data.inner_mut().choices = vec![choice];
                write_slot(&mut data, choice_sources.into_iter().collect());
                if !is_last {
                    data.inner_mut().usage = None;
                    data.clear_nvext();
                }
                if Some(position) != llm_metrics_position {
                    data.clear_llm_metrics();
                }
                yield I::from_parts(
                    data,
                    if is_last {
                        std::mem::take(&mut annotation)
                    } else {
                        I::Annotation::default()
                    },
                );
            }
        }

        // Backstop: the stream ended without a terminating chunk for some choice.
        if let Some(template) = &template {
            for (choice, sources) in flush_undecided_guided_prefixes(&mut undecided_guided_prefix, &mut undecided_spans, policy, prefill) {
                yield response_with_choice(template, choice, recording.then_some(sources), &states, &records);
            }
            let finished = finish_unterminated_choices(&mut states, &mut records, family, policy.parsers_v2_selected, recording);
            if let Some(error) = states.values_mut().find_map(ChoiceState::take_recording_error) {
                yield I::from_error(error);
                return;
            }
            for output in finished {
                match output {
                    FinishedStreamChunk::Choice(choice) => {
                        yield response_with_choice(template, choice, None, &states, &records);
                    }
                    FinishedStreamChunk::SourcedChoice(sourced) => {
                        let (choice, sources) = *sourced;
                        yield response_with_choice(template, choice, Some(sources), &states, &records);
                    }
                    FinishedStreamChunk::Source(response) => yield normalized_source_response(*response, &states, &records),
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "unified_stream_tests.rs"]
mod tests;
