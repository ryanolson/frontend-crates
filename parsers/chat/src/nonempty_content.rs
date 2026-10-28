// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `force_nonempty_content` contract on a stream, and the keep-alive
//! predicate that follows from it.
//!
//! A request with `chat_template_args.force_nonempty_content: true` promises
//! the client a non-empty `content`. When the parser classifies the whole turn
//! as reasoning, the stream holds the reasoning back and sends it as `content`
//! at the end. While it holds, the stream can send no data frames at all, so
//! the HTTP layer turns SSE keep-alive frames on for it.

use std::collections::HashMap;

use std::ops::Range;

use dynamo_parsers::chat_stream::{ChatStreamItem, choice_text, into_entries, match_chunk_sources};
use dynamo_parsers::sources::ChoiceSources;
use dynamo_protocols::types::{ChatChoiceStream, ChatCompletionMessageContent};
use dynamo_renderer::request_rules::ChatTemplateArgs;
use futures::{Stream, StreamExt};

use crate::chat_stream::PolicyStreamItem;
use crate::families::selected_family;
use crate::reasoning_off::is_reasoning_disabled_by_request;
use crate::version::ParserVersion;

/// The input ranges of a choice's held text, kept only while recording.
#[derive(Default)]
struct HeldSources {
    reasoning: Vec<Range<usize>>,
    content: Vec<Range<usize>>,
}

#[derive(Default)]
struct DeferredUnifiedChoice {
    pending_reasoning: String,
    pending_content: String,
    saw_visible_output: bool,
    drained: bool,
    // Text of this choice arrived without ranges, so its output gets no entry.
    unsourced: bool,
    // While recording, the input ranges of the held text. They sit behind a
    // box that the stock path never allocates, so the state and what
    // `release_split` and `drain` return stay the size they were before
    // recording: the wider shapes cost the stock deferral 4.7% more
    // instructions per chunk on brev.
    held: Option<Box<HeldSources>>,
}

impl DeferredUnifiedChoice {
    fn release_split(&mut self) -> (Option<String>, Option<String>) {
        let content = std::mem::take(&mut self.pending_content);
        let reasoning = std::mem::take(&mut self.pending_reasoning);
        (
            (!content.is_empty()).then_some(content),
            (!reasoning.is_empty()).then_some(reasoning),
        )
    }

    /// Releases the held text at a finish. Held ranges are rearranged to
    /// match: when reasoning surfaces as content, its ranges become the
    /// content's.
    fn drain(&mut self) -> (Option<String>, Option<String>) {
        if self.drained {
            return (None, None);
        }
        self.drained = true;
        if self.saw_visible_output {
            return self.release_split();
        }
        let reasoning = std::mem::take(&mut self.pending_reasoning);
        let content = std::mem::take(&mut self.pending_content);
        let held = self.held.as_deref_mut();
        if !reasoning.trim().is_empty() {
            if let Some(held) = held {
                held.content = std::mem::take(&mut held.reasoning);
            }
            (Some(reasoning), None)
        } else {
            if let Some(held) = held {
                held.reasoning.clear();
            }
            ((!content.is_empty()).then_some(content), None)
        }
    }

    /// The held ranges, to add to while recording.
    fn held(&mut self) -> &mut HeldSources {
        self.held.get_or_insert_default()
    }

    /// The held ranges of text just released, when it released any.
    fn take_held(&mut self, released: bool) -> HeldSources {
        match self.held.as_deref_mut() {
            Some(held) if released => std::mem::take(held),
            _ => HeldSources::default(),
        }
    }
}

/// `front`'s ranges, then `back`'s. When one list is empty the other is the
/// result, so text that only one side holds needs no new list: joining into an
/// empty list allocated once per chunk.
fn join_ranges(mut front: Vec<Range<usize>>, mut back: Vec<Range<usize>>) -> Vec<Range<usize>> {
    if front.is_empty() {
        return back;
    }
    front.append(&mut back);
    front
}

/// The source entry of a choice of one output chunk, or none when its text
/// has no source.
fn entry(
    recording: bool,
    unsourced: bool,
    index: u32,
    content: Vec<Range<usize>>,
    reasoning: Vec<Range<usize>>,
) -> Option<ChoiceSources> {
    (recording && !unsourced).then_some(ChoiceSources {
        index,
        content,
        reasoning,
    })
}

/// Whether a request should surface parsed `reasoning_content` as `content`
/// when no content was generated: any request carrying
/// `force_nonempty_content=true`.
///
/// Deliberately NOT keyed on the model or its reasoning parser. The flag is a
/// request-level contract — "this response will have non-empty content" — so
/// it is honored generically after parsing and before the response is sent,
/// rather than being reimplemented inside each model-specific parser. Keying
/// it on a parser allow-list means every new alias silently loses the
/// behavior until someone edits the list.
///
/// Note it rides in on `chat_template_args` but is NOT consumed by the chat
/// template: the Nemotron template never reads `force_nonempty_content`, and
/// rendering with it set produces a byte-identical prompt. It is a
/// serving-layer flag that upstream happens to transport through the template
/// kwargs, which is why the check belongs in postprocessing and not in a
/// parser. So a client can set it on any model, and doing so is an explicit
/// request for non-empty content — honoring it generically is the intent, not
/// an accident of where the flag is declared.
///
/// Drives both paths. The non-streaming aggregator takes it as
/// `move_reasoning_to_content_when_empty`, and the stream path holds reasoning
/// back so it reaches the same answer at the terminal chunk. Both must use
/// this one predicate or the two paths would disagree on the same input.
///
/// `enable_thinking` is intentionally not consulted here: when thinking is
/// off, reasoning parsing is disabled, so `reasoning_content` is always empty
/// and the move is vacuous rather than suppressed.
pub fn wants_reasoning_as_content_when_empty(
    chat_template_args: Option<&ChatTemplateArgs>,
) -> bool {
    chat_template_args.is_some_and(|args| {
        args.get("force_nonempty_content") == Some(&serde_json::Value::Bool(true))
    })
}

/// Whether this request's stream can withhold every data frame while it
/// buffers, which is the only reason to force SSE keep-alive frames on.
///
/// The buffering runs only when the stream parses reasoning, so a
/// `force_nonempty_content=true` request with reasoning disabled
/// (`enable_thinking=false`) defers nothing and gets no heartbeat. The
/// configured interval is opt-in because some OpenAI-compatible clients do
/// not ignore SSE comment frames.
///
/// Known gap: a guided-JSON bypass also suppresses the deferral, but it is
/// derived from guided-output inspection that is not available at the HTTP
/// layer, so a guided-JSON bypass still enables the heartbeat when nothing is
/// withheld. That direction is conservative — extra comment frames on an
/// opt-in path rather than a silent stream.
///
/// `version` is the deployment's parser version (in Dynamo,
/// `DYN_PARSER_VERSION`).
pub fn stream_can_defer_all_output(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    chat_template_args: Option<&ChatTemplateArgs>,
    version: ParserVersion,
) -> bool {
    // Unified families share reasoning-only output deferral with the legacy path.
    let has_reasoning_decoder = reasoning_parser.is_some()
        || selected_family(tool_call_parser, reasoning_parser, version).is_some();
    has_reasoning_decoder
        && wants_reasoning_as_content_when_empty(chat_template_args)
        && !is_reasoning_disabled_by_request(reasoning_parser, chat_template_args)
}

/// The deferral of one chunk: it edits `response` in place, writes the prefix
/// choices to emit before it into `prefix_choices`, and returns `false` when
/// the chunk has no data. It is a plain function so that its locals stay out
/// of the stream's state, which is held across every yield, and it writes
/// into the caller's buffer rather than returning one, which cost a copy per
/// chunk.
fn defer_chunk<I: PolicyStreamItem>(
    states: &mut HashMap<u32, DeferredUnifiedChoice>,
    last_response: &mut Option<I>,
    response: &mut I,
    prefix_choices: &mut Vec<(ChatChoiceStream, Option<ChoiceSources>)>,
) -> bool {
    // While recording, the input slot gives each choice's ranges and
    // the output slot gets the ranges of the text this stage emits.
    let entries = response.sources_mut().map(std::mem::take);
    let recording = entries.is_some();
    let mut matched = entries.map(|entries| {
        let choices = response.chunk().map_or(&[][..], |data| &data.choices[..]);
        match_chunk_sources(entries, choices)
    });
    let Some(data) = response.chunk_mut() else {
        return false;
    };

    for (position, choice) in data.choices.iter_mut().enumerate() {
        let state = states.entry(choice.index).or_default();
        if choice.delta.content.is_some()
            || choice.delta.reasoning_content.is_some()
            || choice.delta.tool_calls.is_some()
        {
            state.drained = false;
        }
        let given = matched
            .as_mut()
            .and_then(|matched| matched.get_mut(position)?.take());
        if recording && given.is_none() && {
            let text = choice_text(choice);
            text.content_len + text.reasoning_len > 0
        } {
            state.unsourced = true;
        }
        let (mut content_sources, mut reasoning_sources) = given
            .map(|given| (given.content, given.reasoning))
            .unwrap_or_default();

        if !state.saw_visible_output
            && let Some(reasoning) = choice.delta.reasoning_content.take()
        {
            state.pending_reasoning.push_str(&reasoning);
            if recording {
                state.held().reasoning.append(&mut reasoning_sources);
            }
        }

        let carries_parts = matches!(
            choice.delta.content,
            Some(ChatCompletionMessageContent::Parts(_))
        );
        match choice.delta.content.take() {
            Some(ChatCompletionMessageContent::Text(text)) => {
                if !state.saw_visible_output && text.trim().is_empty() {
                    state.pending_content.push_str(&text);
                    if recording {
                        state.held().content.append(&mut content_sources);
                    }
                } else {
                    if !text.trim().is_empty() {
                        state.saw_visible_output = true;
                    }
                    let mut content = std::mem::take(&mut state.pending_content);
                    content.push_str(&text);
                    choice.delta.content = Some(ChatCompletionMessageContent::Text(content));
                    if let Some(held) = state.held.as_deref_mut() {
                        let front = std::mem::take(&mut held.content);
                        content_sources = join_ranges(front, content_sources);
                    }
                }
            }
            Some(parts @ ChatCompletionMessageContent::Parts(_)) => {
                choice.delta.content = Some(parts);
            }
            None => {}
        }

        let has_tool_calls = choice
            .delta
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty());
        if carries_parts || has_tool_calls {
            state.saw_visible_output = true;
        }
        if carries_parts {
            state.pending_content.clear();
            if let Some(held) = state.held.as_deref_mut() {
                held.content.clear();
            }
        }

        if state.saw_visible_output {
            let (content, reasoning) = state.release_split();
            let mut held = state.take_held(recording && (content.is_some() || reasoning.is_some()));
            if let Some(reasoning) = reasoning {
                let mut prefix = choice.clone();
                prefix.delta.role = None;
                prefix.delta.content = None;
                prefix.delta.tool_calls = None;
                prefix.delta.function_call = None;
                prefix.delta.refusal = None;
                prefix.delta.reasoning_content = Some(reasoning);
                prefix.finish_reason = None;
                prefix.logprobs = None;
                let sources = entry(
                    recording,
                    state.unsourced,
                    choice.index,
                    Vec::new(),
                    std::mem::take(&mut held.reasoning),
                );
                prefix_choices.push((prefix, sources));
            }
            if let Some(content) = content {
                match choice.delta.content.as_mut() {
                    Some(ChatCompletionMessageContent::Text(existing)) => {
                        existing.insert_str(0, &content);
                        let front = std::mem::take(&mut held.content);
                        content_sources = join_ranges(front, content_sources);
                    }
                    Some(ChatCompletionMessageContent::Parts(_)) | None => {
                        let mut prefix = choice.clone();
                        prefix.delta.role = None;
                        prefix.delta.content = Some(ChatCompletionMessageContent::Text(content));
                        prefix.delta.tool_calls = None;
                        prefix.delta.function_call = None;
                        prefix.delta.refusal = None;
                        prefix.delta.reasoning_content = None;
                        prefix.finish_reason = None;
                        prefix.logprobs = None;
                        let sources = entry(
                            recording,
                            state.unsourced,
                            choice.index,
                            std::mem::take(&mut held.content),
                            Vec::new(),
                        );
                        prefix_choices.push((prefix, sources));
                    }
                }
            }
        }

        if choice.finish_reason.is_some() {
            let (content, reasoning) = state.drain();
            let mut held = state.take_held(recording && (content.is_some() || reasoning.is_some()));
            if let Some(content) = content {
                match choice.delta.content.as_mut() {
                    Some(ChatCompletionMessageContent::Text(existing)) => {
                        existing.push_str(&content);
                        let back = std::mem::take(&mut held.content);
                        content_sources = join_ranges(content_sources, back);
                    }
                    Some(ChatCompletionMessageContent::Parts(_)) => {}
                    None => {
                        choice.delta.content = Some(ChatCompletionMessageContent::Text(content));
                        let back = std::mem::take(&mut held.content);
                        content_sources = join_ranges(content_sources, back);
                    }
                }
            }
            if let Some(reasoning) = reasoning {
                choice
                    .delta
                    .reasoning_content
                    .get_or_insert_default()
                    .push_str(&reasoning);
                let back = std::mem::take(&mut held.reasoning);
                reasoning_sources = join_ranges(reasoning_sources, back);
            }
        }

        if recording {
            // Held text took its ranges with it; what stays on the
            // choice keeps its own.
            if choice.delta.reasoning_content.is_none() {
                reasoning_sources.clear();
            }
            if !matches!(
                choice.delta.content,
                Some(ChatCompletionMessageContent::Text(_))
            ) {
                content_sources.clear();
            }
            // Back at its position, so the output list reuses the input's,
            // in choice order.
            if let Some(matched) = matched.as_mut() {
                matched[position] = entry(
                    recording,
                    state.unsourced,
                    choice.index,
                    content_sources,
                    reasoning_sources,
                );
            }
        }
    }

    let has_choices = !data.choices.is_empty();
    if recording && let Some(slot) = response.sources_mut() {
        *slot = matched.map_or_else(Vec::new, into_entries);
    }
    if has_choices {
        // The copy is only the envelope of an end-of-stream flush, which
        // writes its own entries, so it leaves the slot behind: copying it
        // cost recording two allocations per chunk.
        let slot = response.sources_mut().map(std::mem::take);
        let envelope = response.clone();
        if let (Some(slot), Some(kept)) = (slot, response.sources_mut()) {
            *kept = slot;
        }
        *last_response = Some(envelope);
    }
    true
}

/// Apply the request-level non-empty-content contract after a unified parser has
/// already split reasoning, content, and tool calls.
///
/// While the items carry a source slot, each held span keeps its input ranges
/// and they move with it: into the reasoning or content of a later chunk, a
/// split-off prefix chunk, or the end-of-stream flush.
pub(crate) fn defer_unified_reasoning_for_nonempty_content<I, S>(
    stream_in: S,
) -> impl Stream<Item = I> + Send
where
    I: PolicyStreamItem,
    S: Stream<Item = I> + Send + 'static,
{
    async_stream::stream! {
        let mut states: HashMap<u32, DeferredUnifiedChoice> = HashMap::new();
        let mut last_response: Option<I> = None;
        let mut prefix_choices = Vec::new();
        tokio::pin!(stream_in);

        while let Some(mut response) = stream_in.next().await {
            if ChatStreamItem::is_error(&response) {
                yield response;
                return;
            }
            if !defer_chunk(&mut states, &mut last_response, &mut response, &mut prefix_choices) {
                yield response;
                continue;
            }
            for (prefix_choice, sources) in prefix_choices.drain(..) {
                let mut prefix_response = response.clone();
                prefix_response.scrub_synthetic();
                if let Some(prefix_data) = prefix_response.chunk_mut() {
                    prefix_data.choices = vec![prefix_choice];
                }
                if let Some(slot) = prefix_response.sources_mut() {
                    *slot = sources.into_iter().collect();
                }
                prefix_response.clear_id();
                yield prefix_response;
            }
            yield response;
        }

        let mut indices: Vec<_> = states.keys().copied().collect();
        indices.sort_unstable();
        let flushed: Vec<_> = indices
            .into_iter()
            .filter_map(|index| {
                let state = states.get_mut(&index)?;
                let unsourced = state.unsourced;
                let (content, reasoning) = state.drain();
                let released = content.is_some() || reasoning.is_some();
                let held = state.take_held(released);
                released.then_some((index, (content, reasoning), held, unsourced))
            })
            .collect();
        if !flushed.is_empty()
            && let Some(mut response) = last_response
            && {
                response.scrub_synthetic();
                response.chunk().is_some_and(|data| !data.choices.is_empty())
            }
        {
            let recording = response.sources().is_some();
            let mut flushed = flushed;
            let sources: Vec<ChoiceSources> = flushed
                .iter_mut()
                .filter_map(|(index, _, held, unsourced)| {
                    entry(
                        recording,
                        *unsourced,
                        *index,
                        std::mem::take(&mut held.content),
                        std::mem::take(&mut held.reasoning),
                    )
                })
                .collect();
            if let Some(slot) = response.sources_mut() {
                *slot = sources;
            }
            if let Some(data) = response.chunk_mut()
                && let Some(template) = data.choices.first().cloned()
            {
                data.choices = flushed
                    .into_iter()
                    .map(|(index, (content, reasoning), _, _)| {
                        let mut choice = template.clone();
                        choice.index = index;
                        choice.delta.role = None;
                        choice.delta.content = content.map(ChatCompletionMessageContent::Text);
                        choice.delta.tool_calls = None;
                        choice.delta.function_call = None;
                        choice.delta.refusal = None;
                        choice.delta.reasoning_content = reasoning;
                        choice.finish_reason = None;
                        choice.logprobs = None;
                        choice
                    })
                    .collect();
                yield response;
            }
        }
    }
}

/// Hold the trailing usage-only chunk until every parser recovery chunk has
/// been emitted. A truncated upstream stream can end without
/// `finish_reason`, so reasoning recovery happens at EOF; forwarding usage
/// immediately would put that recovered content after the chunk clients
/// treat as the stream trailer. A transport error discards the pending
/// trailer because the response did not complete successfully.
///
/// A transport error here is an item whose annotation carries an error
/// ([`PolicyStreamItem::has_error`]), whatever its event.
pub(crate) fn hold_usage_until_stream_end<I, S>(stream: S) -> impl Stream<Item = I> + Send
where
    I: PolicyStreamItem,
    S: Stream<Item = I> + Send + 'static,
{
    async_stream::stream! {
        tokio::pin!(stream);
        let mut pending_usage = None;
        let mut transport_failed = false;
        while let Some(response) = stream.next().await {
            if response.has_error() {
                transport_failed = true;
                pending_usage = None;
                yield response;
                continue;
            }
            let is_usage_only = response
                .chunk()
                .is_some_and(|data| data.choices.is_empty() && data.usage.is_some());
            if is_usage_only {
                if let Some(previous) = pending_usage.replace(response) {
                    yield previous;
                }
            } else {
                yield response;
            }
        }
        if !transport_failed && let Some(usage) = pending_usage {
            yield usage;
        }
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;

    /// A choice's deferral state is its fields before source recording plus
    /// one pointer: held ranges sit behind a box the stock path never fills.
    /// With them inline, the state grew from 56 to 104 bytes and the stock
    /// deferral ran 4.7% more instructions per chunk on brev.
    #[test]
    fn the_choice_state_holds_ranges_out_of_line() {
        assert!(
            std::mem::size_of::<DeferredUnifiedChoice>() <= 64,
            "the deferral state is {} bytes",
            std::mem::size_of::<DeferredUnifiedChoice>()
        );
    }
}
