// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The tools a request gives the model, and whether its output may be read as tool
//! calls.

use std::borrow::Cow;

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType,
    CreateChatCompletionRequest, FunctionObject,
};
use serde_json::Value;

/// Whether this request permits model output to be interpreted as tool calls.
///
/// A configured parser describes the model's wire format; it does not grant
/// every request permission to return tool calls. Permission depends only on
/// whether the request supplies effective tools and whether `tool_choice`
/// forbids them. Effective tools include both the OpenAI top-level list and
/// Kimi-style dynamic declarations carried by system messages.
/// Assistant-output constraints apply to assistant content and do not revoke
/// an `auto` request's ability to choose a tool call.
pub fn tool_call_parsing_enabled(request: &CreateChatCompletionRequest) -> bool {
    if !request.has_effective_tools() {
        return false;
    }

    match request
        .tool_choice
        .as_ref()
        .unwrap_or(&ChatCompletionToolChoiceOption::Auto)
    {
        ChatCompletionToolChoiceOption::None => false,
        ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_) => true,
        ChatCompletionToolChoiceOption::Auto => true,
    }
}

/// The parser crate's form of an OpenAI `tool_choice`.
pub fn convert_tool_choice(tool_choice: &ChatCompletionToolChoiceOption) -> ToolChoice {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => ToolChoice::None,
        ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
        ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
        ChatCompletionToolChoiceOption::Named(named) => {
            ToolChoice::Named(named.function.name.clone())
        }
    }
}

/// The parser crate's form of OpenAI tools.
pub fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

/// Merges request tools and tools declared in system messages, then runs `validate`
/// on the merged list.
///
/// Top-level OpenAI tools remain first. Kimi-style tools declared on system
/// messages follow in message order and may use either the wrapped OpenAI form
/// or Kimi's bare function-schema form. This view is used only by validation,
/// parser, and guided-decoding policy; dynamic declarations stay at their
/// original message positions for prompt rendering and KV-cache correctness.
///
/// `validate` is the caller's tool validation (names, schemas, limits). Dynamo passes
/// its own; it runs last, so its errors come after the errors below.
///
/// # Errors
///
/// When a system-message tool is malformed, when `tools` is an explicit empty array
/// and no system message declares tools, or when `validate` fails.
pub fn effective_tools<F>(
    request: &CreateChatCompletionRequest,
    validate: F,
) -> anyhow::Result<Cow<'_, [ChatCompletionTool]>>
where
    F: FnOnce(&[ChatCompletionTool]) -> anyhow::Result<()>,
{
    let dynamic_tools = request
        .dynamic_system_tools()
        .enumerate()
        .map(|(index, tool)| normalize_dynamic_system_tool(tool, index))
        .collect::<Result<Vec<_>, _>>()?;

    // Check the request field, not the merged slice below: an omitted `tools`
    // field also produces an empty slice and must stay valid. System-message
    // tools still make an explicit `tools: []` usable.
    if dynamic_tools.is_empty() && request.tools.as_ref().is_some_and(|tools| tools.is_empty()) {
        anyhow::bail!("tools array cannot be empty; omit the field entirely to disable tool use");
    }

    let tools = if dynamic_tools.is_empty() {
        Cow::Borrowed(request.tools.as_deref().unwrap_or_default())
    } else {
        let mut tools = request.tools.clone().unwrap_or_default();
        tools.extend(dynamic_tools);
        Cow::Owned(tools)
    };

    validate(tools.as_ref())?;
    Ok(tools)
}

fn normalize_dynamic_system_tool(
    tool: &Value,
    index: usize,
) -> Result<ChatCompletionTool, anyhow::Error> {
    let object = tool.as_object().ok_or_else(|| {
        anyhow::anyhow!("dynamic system tool at index {index} must be a JSON object")
    })?;
    let function = match (object.get("type"), object.get("function")) {
        (Some(Value::String(kind)), Some(Value::Object(function))) if kind == "function" => {
            function
        }
        (Some(kind), _) if kind.as_str() != Some("function") => {
            anyhow::bail!("dynamic system tool at index {index} must have type=\"function\"");
        }
        (Some(_), _) => {
            anyhow::bail!(
                "dynamic system tool at index {index} with type=\"function\" needs a function object"
            );
        }
        (None, Some(_)) => {
            anyhow::bail!(
                "dynamic system tool at index {index} with a function field needs type=\"function\""
            );
        }
        (None, None) => object,
    };

    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("dynamic system tool at index {index} needs a non-empty string name")
        })?;
    let description = optional_dynamic_string(function.get("description"), "description", index)?;
    let parameters = match function.get("parameters") {
        None | Some(Value::Null) => None,
        Some(parameters @ Value::Object(_)) => Some(parameters.clone()),
        Some(_) => {
            anyhow::bail!(
                "dynamic system tool at index {index} parameters must be a JSON Schema object"
            );
        }
    };
    let strict = match function.get("strict") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(strict)) => Some(*strict),
        Some(_) => {
            anyhow::bail!("dynamic system tool at index {index} strict must be a boolean");
        }
    };

    Ok(ChatCompletionTool {
        r#type: ChatCompletionToolType::Function,
        function: FunctionObject {
            name: name.to_string(),
            description,
            parameters,
            strict,
            ..Default::default()
        },
        ..Default::default()
    })
}

fn optional_dynamic_string(
    value: Option<&Value>,
    field: &str,
    index: usize,
) -> Result<Option<String>, anyhow::Error> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => anyhow::bail!("dynamic system tool at index {index} {field} must be a string"),
    }
}

fn is_kimi_k3_parser(parser_name: Option<&str>) -> bool {
    parser_name.is_some_and(|parser| matches!(parser, "kimi_k3" | "kimi-k3"))
}

// Unlike K3, the parser registry exposes only the canonical `kimi_k2` spelling.
fn is_kimi_k2_parser(parser_name: Option<&str>) -> bool {
    parser_name == Some("kimi_k2")
}

/// Whether a forced `tool_choice` needs the parser's native call format, which the
/// generic forced-tool JSON grammar cannot describe.
pub fn requires_native_tool_call_format(
    parser_name: Option<&str>,
    tool_choice: &ToolChoice,
) -> bool {
    // The generic forced-tool JSON grammar describes a different wire format
    // from Kimi's native marker-delimited calls. This also covers K3 required,
    // which uses its prompt/parser path when structural guidance is off.
    (is_kimi_k2_parser(parser_name)
        && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_)))
        || (is_kimi_k3_parser(parser_name)
            && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_)))
}
