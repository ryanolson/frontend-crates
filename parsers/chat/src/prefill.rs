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
    /// continues its final message (`continue_final_message`) does.
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

/// The header of the assistant message that `prompt` ends inside, if it ends
/// inside one: the last `<|start|>assistant` opens one of
/// [`GPT_OSS_MESSAGE_HEADERS`], and nothing after it ends the message. A
/// generation prompt ends at `<|start|>assistant`, with no header, so it
/// opens none.
fn gpt_oss_open_message(prompt: &str) -> Option<&'static str> {
    let (_, message) = prompt.rsplit_once("<|start|>assistant")?;
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
    // message in the answer.
    let open_message = (configured_family(tool_call_parser, reasoning_parser) == Some("gpt_oss"))
        .then(|| formatted_prompt.and_then(gpt_oss_open_message))
        .flatten();
    let open_analysis = open_message == Some(GPT_OSS_MESSAGE_HEADERS[1]);
    PromptReasoningPrefill {
        legacy,
        unified: unified || prefix.is_some() || open_analysis,
        prefix,
        response: open_message.is_some() && !open_analysis,
    }
}
