// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whether the rendered prompt already opened a reasoning block.
//!
//! Some chat templates end the prompt with the reasoning opener (`<think>`, for
//! example), so the model's first token is already inside reasoning. The parser must
//! then start in reasoning, or it sends the reasoning as content.

use crate::families::configured_family;
use crate::route::ToolProcessingRoute;

/// Prompt-opened reasoning for the parser route selected after guided decoding.
///
/// Native Muse channels and the legacy Basic reasoning parser use different
/// delimiters. Keep both facts until the request's actual route is known.
#[derive(Clone, Copy, Debug)]
pub struct PromptReasoningPrefill {
    legacy: bool,
    unified: bool,
    prefix: Option<&'static str>,
    response: bool,
}

impl From<bool> for PromptReasoningPrefill {
    /// An explicit boolean describes the caller's already-selected parser.
    fn from(value: bool) -> Self {
        Self {
            legacy: value,
            unified: value,
            prefix: None,
            response: false,
        }
    }
}

impl PromptReasoningPrefill {
    /// Whether the parser on `route` starts inside reasoning.
    pub fn for_route(self, route: &ToolProcessingRoute) -> bool {
        match route {
            ToolProcessingRoute::Unified(_) => self.unified,
            _ => self.legacy,
        }
    }

    /// Whether the v1 reasoning parser starts inside reasoning.
    pub fn legacy(self) -> bool {
        self.legacy
    }

    /// Whether the unified v2 parser starts inside reasoning.
    pub fn unified(self) -> bool {
        self.unified
    }

    /// The part of a Gemma 4 reasoning header that the prompt already holds, if any.
    pub fn prefix(self) -> Option<&'static str> {
        self.prefix
    }

    /// Whether the unified v2 parser starts inside the answer: the prompt
    /// ended inside a GPT-OSS `final` or `commentary` message, as a prompt that
    /// continues its final message (`continue_final_message`) does, or after a
    /// last header that is not an assistant's.
    pub fn response(self) -> bool {
        self.response
    }
}

/// The Harmony message headers a continued GPT-OSS message can open with.
const GPT_OSS_MESSAGE_HEADERS: [&str; 3] = [
    "<|channel|>final<|message|>",
    "<|channel|>analysis<|message|>",
    "<|channel|>commentary<|message|>",
];

/// Where a GPT-OSS prompt ends, for the unified parser's start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GptOssPromptEnd {
    /// Inside the assistant message with this header: the model continues it.
    Open(&'static str),
    /// After a last header that is not an assistant's, as a continued user
    /// message, or a prompt with no generation prompt, ends. The model
    /// writes no header of its own there, so its reply is the answer.
    NotAssistant,
}

/// Where `prompt` ends ([`GptOssPromptEnd`]), read from its last
/// `<|start|>` header as dynamo reads it: the role is the text after
/// `<|start|>`, up to whitespace or `<`. A generation prompt ends at
/// `<|start|>assistant`, with no channel open, so it gives `None`, as does an
/// assistant message that ended or a header the list does not hold.
fn gpt_oss_prompt_end(prompt: &str) -> Option<GptOssPromptEnd> {
    let (_, header) = prompt.rsplit_once("<|start|>")?;
    let role_end = header
        .find(|c: char| c.is_whitespace() || c == '<')
        .unwrap_or(header.len());
    if &header[..role_end] != "assistant" {
        return Some(GptOssPromptEnd::NotAssistant);
    }
    gpt_oss_open_message(&header[role_end..]).map(GptOssPromptEnd::Open)
}

/// The header of the assistant message that `message` (the text after
/// `<|start|>assistant`) opens, if the message is still open: it starts with
/// one of [`GPT_OSS_MESSAGE_HEADERS`], and nothing after the header ends it.
fn gpt_oss_open_message(message: &str) -> Option<&'static str> {
    let header = GPT_OSS_MESSAGE_HEADERS
        .into_iter()
        .find(|header| message.starts_with(header))?;
    let body = &message[header.len()..];
    let ended = ["<|end|>", "<|return|>", "<|call|>", "<|start|>"]
        .iter()
        .any(|marker| body.contains(marker));
    (!ended).then_some(header)
}

/// Whether `formatted_prompt`, with trailing whitespace removed, ends in the reasoning
/// opener of `reasoning_parser`. Unknown names use `<think>`.
pub fn prompt_injected_reasoning_start(
    reasoning_parser: Option<&str>,
    formatted_prompt: Option<&str>,
) -> bool {
    let Some(prompt) = formatted_prompt.map(str::trim_end) else {
        return false;
    };

    match reasoning_parser {
        Some("minimax_m3") | Some("minimax-m3") => prompt.ends_with("<mm:think>"),
        Some("kimi_k3") | Some("kimi-k3") => {
            let mut suffix = prompt.chars().rev().filter(|c| !c.is_whitespace());
            "<|open|>think<|sep|>"
                .chars()
                .rev()
                .all(|c| suffix.next() == Some(c))
        }
        Some("gemma4") => prompt.ends_with("<|channel>thought") || prompt.ends_with("<|channel>"),
        Some("muse" | "muse_glimmer") => prompt.ends_with("assistant to=self<|message|>"),
        _ => prompt.ends_with("<think>"),
    }
}

/// [`prompt_injected_reasoning_start`] for a parser pair: the pair's unified family
/// chooses the opener when it has one, and the reasoning parser otherwise.
pub fn prompt_injected_reasoning_start_for_parsers(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    formatted_prompt: Option<&str>,
) -> bool {
    let parser = configured_family(tool_call_parser, reasoning_parser).or(reasoning_parser);
    prompt_injected_reasoning_start(parser, formatted_prompt)
}

/// The prompt prefill for both routes the parser pair can take.
pub fn prompt_reasoning_prefill_for_parsers(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    formatted_prompt: Option<&str>,
) -> PromptReasoningPrefill {
    let unified = prompt_injected_reasoning_start_for_parsers(
        tool_call_parser,
        reasoning_parser,
        formatted_prompt,
    );
    let legacy = if matches!(reasoning_parser, Some("muse" | "muse_glimmer")) {
        // The legacy crate falls back to Basic for Muse, so only its generic
        // opener may seed reasoning. Native Muse prefill would swallow tool JSON.
        prompt_injected_reasoning_start(None, formatted_prompt)
    } else {
        unified
    };
    let prefix = if configured_family(tool_call_parser, reasoning_parser) == Some("gemma4") {
        const HEADER: &str = "<|channel>thought\n";
        formatted_prompt.and_then(|raw| {
            ("<|channel>".len()..HEADER.len())
                .rev()
                .map(|end| &HEADER[..end])
                .find(|prefix| raw.ends_with(prefix))
        })
    } else {
        None
    };
    // A continued GPT-OSS message leaves its channel open: an analysis
    // message starts the parser in reasoning, and a final or commentary
    // message in the answer. After a header that is not an assistant's, the
    // reply is the answer too, as dynamo starts it.
    let end = (configured_family(tool_call_parser, reasoning_parser) == Some("gpt_oss"))
        .then(|| formatted_prompt.and_then(gpt_oss_prompt_end))
        .flatten();
    let open_analysis = end == Some(GptOssPromptEnd::Open(GPT_OSS_MESSAGE_HEADERS[1]));
    PromptReasoningPrefill {
        legacy,
        unified: unified || prefix.is_some() || open_analysis,
        prefix,
        response: end.is_some() && !open_analysis,
    }
}
