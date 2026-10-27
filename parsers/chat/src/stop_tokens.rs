// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A tool-call end token that is also a hidden EOS becomes a visible stop.
//!
//! Some tool-call parsers end a call on a token that is also a model EOS, for example
//! Harmony's `<|call|>` for GPT-OSS. As a hidden stop, the engine stops and drops the
//! token, so the parser sees a call with no end and drops it. As a visible stop, the
//! engine still stops, and the parser gets the token.

use anyhow::Context;
use dynamo_renderer::OAIChatLikeRequest;

/// Move every tool-call end token of `tool_call_parser` that is a hidden stop into the
/// visible stops, when the request permits tool calls.
///
/// `encode` turns a marker into token IDs; only a marker that is one token can be a
/// stop. Neither list is created when nothing moves.
///
/// # Errors
///
/// When `encode` fails for an end token.
pub fn move_tool_parser_end_tokens_to_visible_stops<R, E>(
    tool_call_parser: Option<&str>,
    request: &R,
    stop_token_ids_hidden: &mut Option<Vec<u32>>,
    stop_token_ids_visible: &mut Option<Vec<u32>>,
    encode: E,
) -> anyhow::Result<()>
where
    R: OAIChatLikeRequest,
    E: FnMut(&str) -> anyhow::Result<Vec<u32>>,
{
    let mut visible_tool_parser_end_token_ids = Vec::new();
    if let Some(stop_tokens) = stop_token_ids_hidden {
        visible_tool_parser_end_token_ids = remove_tool_parser_end_tokens_from_hidden_stops(
            tool_call_parser,
            request,
            stop_tokens,
            encode,
        )?;
    }
    if !visible_tool_parser_end_token_ids.is_empty() {
        let visible_stops = stop_token_ids_visible.get_or_insert_with(Vec::new);
        for token_id in visible_tool_parser_end_token_ids {
            if !visible_stops.contains(&token_id) {
                visible_stops.push(token_id);
            }
        }
    }
    Ok(())
}

/// Remove the parser's single-token end markers from `hidden_stop_token_ids`, and
/// return the ones it removed, in marker order and without repeats.
///
/// # Errors
///
/// When `encode` fails for an end token.
pub fn remove_tool_parser_end_tokens_from_hidden_stops<R, E>(
    tool_call_parser: Option<&str>,
    request: &R,
    hidden_stop_token_ids: &mut Vec<u32>,
    mut encode: E,
) -> anyhow::Result<Vec<u32>>
where
    R: OAIChatLikeRequest,
    E: FnMut(&str) -> anyhow::Result<Vec<u32>>,
{
    let has_tools = request_has_effective_tools(request);
    let tool_choice_none = request
        .tool_choice()
        .as_ref()
        .and_then(|tool_choice| tool_choice.as_str())
        == Some("none");

    if !should_keep_tool_parser_end_tokens_visible(has_tools, tool_choice_none) {
        return Ok(Vec::new());
    }

    let Some(tool_call_parser) = tool_call_parser.filter(|p| !p.is_empty()) else {
        return Ok(Vec::new());
    };
    let Some(tool_call_config) =
        dynamo_parsers::tool_calling::parsers::get_tool_parser_map().get(tool_call_parser)
    else {
        return Ok(Vec::new());
    };

    let mut visible_stop_token_ids = Vec::new();
    for end_token in tool_call_config.parser_config.tool_call_end_tokens() {
        if end_token.is_empty() {
            continue;
        }
        let encoded = encode(&end_token).with_context(|| {
            format!(
                "Failed to encode tool-call end token {end_token:?} for parser {tool_call_parser:?}"
            )
        })?;
        let was_hidden_eos = remove_single_token_marker(hidden_stop_token_ids, &encoded);
        if !was_hidden_eos {
            tracing::debug!(
                token_ids = ?encoded,
                end_token,
                parser = tool_call_parser,
                "Tool-call end token was not a single hidden EOS token"
            );
            continue;
        }
        if let [token_id] = encoded.as_slice()
            && !visible_stop_token_ids.contains(token_id)
        {
            visible_stop_token_ids.push(*token_id);
        }
    }

    Ok(visible_stop_token_ids)
}

/// Whether the request declares tools, at the top level or in a system message.
///
/// `OAIChatLikeRequest` has no equivalent method, so this generic path mirrors
/// `CreateChatCompletionRequest::has_effective_tools`.
pub fn request_has_effective_tools<R: OAIChatLikeRequest>(request: &R) -> bool {
    request
        .tools()
        .as_ref()
        .and_then(|tools| tools.len())
        .is_some_and(|len| len > 0)
        || request.typed_messages().is_some_and(|messages| {
            messages.iter().any(|message| {
                matches!(
                    message,
                    dynamo_protocols::types::ChatCompletionRequestMessage::System(system)
                        if system.tools.as_ref().is_some_and(|tools| !tools.is_empty())
                )
            })
        })
}

fn should_keep_tool_parser_end_tokens_visible(has_tools: bool, tool_choice_none: bool) -> bool {
    has_tools && !tool_choice_none
}

fn remove_single_token_marker(
    hidden_eos_token_ids: &mut Vec<u32>,
    marker_token_ids: &[u32],
) -> bool {
    let [marker_token_id] = marker_token_ids else {
        return false;
    };
    let before = hidden_eos_token_ids.len();
    hidden_eos_token_ids.retain(|token_id| token_id != marker_token_id);
    hidden_eos_token_ids.len() != before
}
