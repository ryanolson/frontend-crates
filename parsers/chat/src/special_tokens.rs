// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `skip_special_tokens` default for each parser.
//!
//! Some parsers match markers that the tokenizer counts as special tokens. The OpenAI
//! default, `skip_special_tokens = true`, strips them in the decode, so the parser sees
//! no markers and the markup leaks into `content`. For those parsers the default turns
//! to `false`, unless the client set the field.

use anyhow::Context;

use crate::families::selected_family;
use crate::version::ParserVersion;

/// Whether the selected tool-call or reasoning parser depends on the
/// engine emitting special tokens (e.g. Gemma 4's `<|tool_call>` /
/// `<|channel>`). Mirrors upstream vLLM's per-parser `adjust_request`
/// hooks. Used to flip the request default for `skip_special_tokens`
/// from `true` to `false` so the parsers actually see the markers
/// they're matching on.
///
/// # Errors
///
/// When the selected unified family's parser cannot be built.
pub fn parser_requires_special_tokens(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    version: ParserVersion,
) -> anyhow::Result<bool> {
    if let Some(family) = selected_family(tool_call_parser, reasoning_parser, version) {
        // The unified grammar owns token preservation, including native tool markers.
        return dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
            .with_context(|| format!("Failed to resolve special-token requirements for {family}"))
            .map(|parser| parser.preserve_special_tokens());
    }
    // Parsers in this allow-list match against special tokens that the
    // tokenizer would otherwise strip when `skip_special_tokens=true`
    // (the OpenAI-API default). Without the tokens preserved through
    // decode the parsers silently produce empty reasoning_content /
    // tool_calls.
    //
    // - gemma4: `<|think|>` prompt trigger plus parser-visible
    //   `<|channel>` / `<channel|>` reasoning markers and tool-call markers.
    // - harmony / gpt_oss: `<|channel|>analysis<|message|>...<|end|>`.
    // - kimi_k2: `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>`.
    // - kimi_k25: `</think>` (special token id 163607).
    // - kimi_k3: `<|open|>` / `<|close|>` / `<|sep|>` XTML markers.
    // - mistral: `[THINK]` / `[/THINK]` reasoning markers.
    // - minimax_m3: `]<]minimax[>[` tool-call namespace tokens and
    //   `<mm:think>` reasoning markers.
    // - inkling: `<|message_model|>` / `<|content_thinking|>` /
    //   `<|content_text|>` / `<|content_invoke_tool_json|>` / `<|end_message|>`
    //   channel markers, consumed by both the tool-call and reasoning parsers.
    // - muse_glimmer: `<|start|>` / `<|message|>` / `<|eom|>` / `<|eot|>`
    //   channel markers, consumed by the unified parser (reasoning + content +
    //   tool calls); matched on either parser name since the card may set only
    //   the reasoning name.
    Ok(matches!(
        tool_call_parser,
        Some("gemma4")
            | Some("gemma-4")
            | Some("harmony")
            | Some("kimi_k2")
            | Some("kimi_k3")
            | Some("kimi-k3")
            | Some("minimax_m3")
            | Some("minimax-m3")
            | Some("minimax_m3_nom")
            | Some("minimax-m3-nom")
            | Some("inkling")
            | Some("muse_glimmer")
            | Some("muse")
            | Some("deepseek_v41")
    ) || matches!(
        reasoning_parser,
        Some("gemma4")
            | Some("gemma-4")
            | Some("gpt_oss")
            | Some("kimi_k25")
            | Some("kimi_k3")
            | Some("kimi-k3")
            | Some("mistral")
            | Some("minimax_m3")
            | Some("minimax-m3")
            | Some("inkling")
            | Some("muse_glimmer")
            | Some("muse")
            | Some("deepseek_v41")
    ))
}

/// Whether the resolved `skip_special_tokens` will strip the special-token
/// markers an active parser depends on — guaranteeing the parser silently
/// emits empty tool_calls / reasoning_content and the markup leaks into
/// `content`. Only true when the caller explicitly forced
/// `skip_special_tokens=true` while a special-token-dependent parser is
/// active; otherwise the default is flipped to false before this check.
pub fn special_tokens_will_be_stripped(
    skip_special_tokens: Option<bool>,
    parser_requires_special_tokens: bool,
) -> bool {
    skip_special_tokens == Some(true) && parser_requires_special_tokens
}

/// The request's `skip_special_tokens` after the parser default.
///
/// `requested` is the client's value. `parser_requires_special_tokens` comes from
/// [`parser_requires_special_tokens`]. The parser names are only for the warning.
pub fn resolve_skip_special_tokens(
    requested: Option<bool>,
    parser_requires_special_tokens: bool,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<bool> {
    // Some parsers rely on `<|tool_call>`, `<|channel>`, etc. being
    // visible in the decoded text. The default `skip_special_tokens=true`
    // strips them and silently bypasses parsing. Mirror upstream's
    // per-parser `adjust_request` hook by flipping the default to false
    // for parsers that need special tokens preserved, unless the caller
    // has explicitly set `skip_special_tokens`.
    if requested.is_none() && parser_requires_special_tokens {
        return Some(false);
    }
    if special_tokens_will_be_stripped(requested, parser_requires_special_tokens) {
        // Caller forced `skip_special_tokens=true` while a special-token-
        // dependent parser is active. The engine's markers (e.g. harmony
        // `<|channel|>` / `<|message|>`) get stripped before the parser
        // runs, so tool_calls / reasoning_content come back empty and the
        // markup leaks into `content`. Warn once: this is a silent,
        // deterministic correctness loss that no parser fixture can catch.
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                ?tool_call_parser,
                ?reasoning_parser,
                "skip_special_tokens=true requested while a special-token-dependent parser is active; the engine's special tokens will be stripped before parsing, so tool_calls/reasoning_content will be empty and the markup will leak into content. Unset skip_special_tokens (Dynamo defaults it to false for these parsers) or set it to false."
            );
        });
    }
    requested
}
