// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rules that the Dynamo frontend's HTTP handlers apply to a chat request
//! before the preprocessor runs: the reasoning controls of Chat Completions
//! and Responses, and the thinking default of Messages.
//!
//! They change the template arguments, so a frontend that skips them
//! renders another prompt than Dynamo. [`apply_handler_rules`] runs the
//! rule of the API a request came in on, as that API's handler does.

use dynamo_protocols::types::CreateChatCompletionRequest;
use serde_json::Value;

use super::thinking::ChatTemplateArgs;
use crate::PromptRenderError;

/// The public API a chat request came in on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestApi {
    /// OpenAI Chat Completions.
    ChatCompletions,
    /// OpenAI Responses, after its conversion to a chat request.
    Responses,
    /// Anthropic Messages, after its conversion to a chat request.
    Messages,
}

/// Apply the rule that the handler of `api` runs before the preprocessor.
///
/// - Chat Completions and Responses: [`normalize_reasoning_template_args`]
///   (Dynamo `lib/llm/src/http/service/openai.rs`, `normalize_chat_reasoning_template_args`).
///   On success, the caller drops the request's top-level `thinking`, which
///   is then folded into `chat_template_args`.
/// - Messages: [`apply_messages_thinking_defaults`]
///   (`lib/llm/src/http/service/anthropic.rs`, `handler_messages`).
///
/// `thinking` is the request's top-level `thinking`, and
/// `thinking_explicitly_disabled` is whether a Messages request set
/// `thinking.type` to `disabled`.
///
/// # Errors
///
/// [`PromptRenderError::InvalidRequest`] for a `thinking` value that is not a
/// boolean or an object with `type` `enabled`, `disabled`, or `adaptive`.
pub fn apply_handler_rules(
    api: RequestApi,
    request: &CreateChatCompletionRequest,
    thinking: Option<&Value>,
    has_reasoning_parser: bool,
    thinking_explicitly_disabled: bool,
    chat_template_args: &mut Option<ChatTemplateArgs>,
) -> Result<(), PromptRenderError> {
    match api {
        RequestApi::ChatCompletions | RequestApi::Responses => {
            normalize_reasoning_template_args(request, thinking, chat_template_args)
        }
        RequestApi::Messages => {
            apply_messages_thinking_defaults(
                has_reasoning_parser,
                thinking_explicitly_disabled,
                chat_template_args,
            );
            Ok(())
        }
    }
}

/// Resolve a request's reasoning controls into `chat_template_args`: the
/// top-level `thinking` and `reasoning_effort`, and the thinking keys and
/// `reasoning_effort` already in the map.
///
/// The highest control wins, then it is stated in every dialect a template
/// may read: `thinking`, `enable_thinking`, and `thinking_mode`. An explicit
/// top-level `reasoning_effort` wins over the nested one, as in vLLM, and is
/// written to the map. The caller then drops the top-level `thinking`, so it
/// is not read again with another precedence.
///
/// # Errors
///
/// [`PromptRenderError::InvalidRequest`] for a `thinking` value that is not a
/// boolean or an object with `type` `enabled`, `disabled`, or `adaptive`.
pub fn normalize_reasoning_template_args(
    request: &CreateChatCompletionRequest,
    thinking: Option<&Value>,
    chat_template_args: &mut Option<ChatTemplateArgs>,
) -> Result<(), PromptRenderError> {
    let thinking_mode = thinking.map(openai_thinking_mode).transpose()?.flatten();
    // vLLM's order: an explicit top-level grade wins, else the nested one.
    let reasoning_effort = request
        .reasoning_effort
        .as_ref()
        .and_then(|effort| serde_json::to_value(effort).ok())
        .or_else(|| {
            chat_template_args
                .as_ref()
                .and_then(|args| args.get("reasoning_effort"))
                // `null` means the client set nothing.
                .filter(|effort| !effort.is_null())
                .cloned()
        });

    let has_template_control = chat_template_args
        .as_ref()
        .is_some_and(|args| THINKING_KEYS.iter().any(|key| args.contains_key(*key)));
    if thinking_mode.is_none() && reasoning_effort.is_none() && !has_template_control {
        return Ok(());
    }

    let args = chat_template_args.get_or_insert_with(ChatTemplateArgs::new);
    if let Some(mode) = thinking_mode {
        match mode {
            OpenAiThinkingMode::Enabled => set_thinking(args, true),
            OpenAiThinkingMode::Disabled => set_thinking(args, false),
            OpenAiThinkingMode::Adaptive => {
                // `adaptive` defers to the model, so no toggle may survive.
                args.remove("thinking");
                args.remove("enable_thinking");
                args.insert(
                    "thinking_mode".to_string(),
                    Value::String("adaptive".to_string()),
                );
            }
        }
    }
    // Downstream readers compare modes case-insensitively; match them.
    let mode = args
        .get("thinking_mode")
        .and_then(|v| v.as_str())
        .map(str::to_ascii_lowercase);
    // Highest wins, then it is stated in every dialect. A bool here is the
    // client's own, because the root `adaptive` param cleared both above, so
    // it outranks a nested mode.
    let decided = THINKING_TOGGLES
        .iter()
        .find_map(|key| args.get(*key).and_then(|v| v.as_bool()))
        .or_else(|| match mode.as_deref() {
            // The model decides, so nothing is written.
            Some("adaptive") => None,
            Some("enabled") => Some(true),
            Some("disabled") => Some(false),
            // Only a string is a grade; anything else decides nothing.
            _ => reasoning_effort
                .as_ref()
                .and_then(|effort| effort.as_str())
                .map(|effort| effort != "none"),
        });
    if let Some(on) = decided {
        set_thinking(args, on);
    }
    if let Some(effort) = reasoning_effort {
        args.insert("reasoning_effort".to_string(), effort);
    }
    Ok(())
}

/// The Messages handler's thinking default: with a reasoning parser, and
/// thinking not disabled by the client, the template is asked to open a
/// reasoning block (`enable_thinking`) and to keep earlier turns' reasoning
/// (`truncate_history_thinking: false`), unless the request set either.
pub fn apply_messages_thinking_defaults(
    has_reasoning_parser: bool,
    thinking_explicitly_disabled: bool,
    chat_template_args: &mut Option<ChatTemplateArgs>,
) {
    if !has_reasoning_parser || thinking_explicitly_disabled {
        return;
    }
    let args = chat_template_args.get_or_insert_with(ChatTemplateArgs::new);
    args.entry("enable_thinking".to_string())
        .or_insert(Value::Bool(true));
    // Agentic flows need the reasoning of earlier turns, which some
    // templates (Nemotron) strip by default.
    args.entry("truncate_history_thinking".to_string())
        .or_insert(Value::Bool(false));
}

/// The two boolean dialects, in precedence order. `thinking_mode` carries the
/// same decision as a string; families split over which one they read.
const THINKING_TOGGLES: [&str; 2] = ["thinking", "enable_thinking"];
const THINKING_KEYS: [&str; 3] = ["thinking", "enable_thinking", "thinking_mode"];

fn set_thinking(args: &mut ChatTemplateArgs, on: bool) {
    args.insert("thinking".to_string(), Value::Bool(on));
    args.insert("enable_thinking".to_string(), Value::Bool(on));
    args.insert(
        "thinking_mode".to_string(),
        Value::String(if on { "enabled" } else { "disabled" }.to_string()),
    );
}

enum OpenAiThinkingMode {
    Enabled,
    Disabled,
    Adaptive,
}

fn openai_thinking_mode(value: &Value) -> Result<Option<OpenAiThinkingMode>, PromptRenderError> {
    if let Some(enabled) = value.as_bool() {
        return Ok(Some(if enabled {
            OpenAiThinkingMode::Enabled
        } else {
            OpenAiThinkingMode::Disabled
        }));
    }

    let Some(thinking_object) = value.as_object() else {
        return Err(PromptRenderError::invalid_request(
            "`thinking` must be a boolean or an object with `type` set to `enabled`, `disabled`, or `adaptive`",
        ));
    };
    let Some(thinking_type) = thinking_object.get("type").and_then(|v| v.as_str()) else {
        return Err(PromptRenderError::invalid_request(
            "`thinking.type` must be `enabled`, `disabled`, or `adaptive`",
        ));
    };
    match thinking_type {
        "enabled" => Ok(Some(OpenAiThinkingMode::Enabled)),
        "disabled" => Ok(Some(OpenAiThinkingMode::Disabled)),
        "adaptive" => Ok(Some(OpenAiThinkingMode::Adaptive)),
        _ => Err(PromptRenderError::invalid_request(
            "`thinking.type` must be `enabled`, `disabled`, or `adaptive`",
        )),
    }
}
