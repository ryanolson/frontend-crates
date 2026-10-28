// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which parser path a chat request takes: the unified v2 parser, the v1 tool-call
//! jail, or no tool parsing at all.

use dynamo_protocols::types::{ChatCompletionToolChoiceOption, CreateChatCompletionRequest};

use crate::families::{is_v2_configured_family, selected_family};
use crate::tools::tool_call_parsing_enabled;
use crate::version::{PARSER_VERSION_ENV, ParserVersion};

/// A request that the selected parser version cannot serve. Dynamo answers it with
/// HTTP 400 and this message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InvalidRequest(pub String);

/// The parser path for one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolProcessingRoute {
    /// One unified v2 parser owns reasoning, text, and tool calls.
    Unified(&'static str),
    /// The v1 reasoning parser, then the v1 tool-call jail with this tool-call parser.
    LegacyJail(Option<String>),
    /// No tool-call parsing.
    PassThrough,
}

impl ToolProcessingRoute {
    /// Whether the request goes through the v1 tool-call jail.
    pub fn uses_legacy_jail(&self) -> bool {
        matches!(self, Self::LegacyJail(_))
    }
}

/// Pick the parser path for a request.
///
/// `tool_call_parser` and `reasoning_parser` are the deployment's configured names.
/// A parser pair with a unified family takes that family whatever the request asks.
/// Otherwise the request takes the v1 jail when it permits tool calls and the jail can
/// parse them (see [`should_apply_tool_jail`]), and passes through when it does not.
///
/// # Errors
///
/// [`InvalidRequest`] when `version` cannot serve the request: an explicit version
/// with no compatible parser, or [`ParserVersion::V2`] for a request that needs the
/// v1 jail.
pub fn tool_processing_route(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    request: &CreateChatCompletionRequest,
    version: ParserVersion,
) -> Result<ToolProcessingRoute, InvalidRequest> {
    let family = selected_family(tool_call_parser, reasoning_parser, version);
    // Options, capture, and aggregation reuse selection; only request routing
    // owns the operator-facing decision line.
    tracing::debug!(
        target: "dynamo_unified",
        ?tool_call_parser,
        ?reasoning_parser,
        ?family,
        ?version,
        "unified parser path decision"
    );
    if let Some(family) = family {
        return Ok(ToolProcessingRoute::Unified(family));
    }

    let effective_tool_call_parser = tool_call_parser.map(str::to_string).or_else(|| {
        reasoning_parser
            .filter(|parser| matches!(*parser, "kimi_k3" | "kimi-k3"))
            .map(str::to_string)
    });
    let parser_unwraps_all_kimi_k3_responses = effective_tool_call_parser
        .as_deref()
        .is_some_and(|parser| matches!(parser, "kimi_k3" | "kimi-k3"));
    let tool_call_parsing_enabled = tool_call_parsing_enabled(request);
    let has_tools = request.has_effective_tools();
    let should_jail = if tool_call_parsing_enabled || parser_unwraps_all_kimi_k3_responses {
        should_apply_tool_jail(
            effective_tool_call_parser.as_ref(),
            request.tool_choice.as_ref(),
            has_tools,
        )
    } else {
        false
    };

    if !should_jail {
        return Ok(ToolProcessingRoute::PassThrough);
    }

    validate_tool_request_mode(
        version,
        effective_tool_call_parser.as_deref(),
        reasoning_parser,
        should_jail,
    )
    .map_err(|error| InvalidRequest(format!("{error:#}")))?;

    if version == ParserVersion::V2 {
        return Err(InvalidRequest(format!(
            "{PARSER_VERSION_ENV}=2 was requested, but this tool choice requires the v1 tool-call jail"
        )));
    }
    Ok(ToolProcessingRoute::LegacyJail(effective_tool_call_parser))
}

/// Determine if we should apply the tool calling jail based on configuration.
pub fn should_apply_tool_jail(
    tool_call_parser: Option<&String>,
    tool_choice: Option<&ChatCompletionToolChoiceOption>,
    has_tools: bool,
) -> bool {
    // Necessary for now because K3 needs a special parser for XTML channel
    // parsing on all responses, regardless of tool or reasoning usage.
    // Keep its parser active so response/message wrappers never leak into
    // OpenAI `content`, even when the request has no tools.
    if tool_call_parser.is_some_and(|parser| matches!(parser.as_str(), "kimi_k3" | "kimi-k3")) {
        return true;
    }

    match (tool_call_parser, tool_choice, has_tools) {
        // tool_choice=required/named work without parser (use Immediate jail mode)
        (None, Some(ChatCompletionToolChoiceOption::Required), true) => true,
        (None, Some(ChatCompletionToolChoiceOption::Named(_)), true) => true,

        // tool_choice=auto requires a parser
        (None, Some(ChatCompletionToolChoiceOption::Auto), true) => {
            tracing::warn!(
                "Tool choice 'auto' specified but no tool parser configured; proceeding without jailing"
            );
            false
        }

        // Parser exists and tools might be called
        (Some(_), Some(ChatCompletionToolChoiceOption::None), _) => {
            false // Explicitly disabled
        }
        (Some(_), Some(_), true) => true, // Any other tool_choice with tools
        (Some(_), None, true) => true,    // Default behavior when tools present

        // No tools or no parser
        _ => false,
    }
}

/// Validate that an explicit parser generation exists for the configured family.
///
/// Dynamo runs this at start-up, when a model registers.
///
/// # Errors
///
/// When `version` is [`ParserVersion::V1`] and a configured name has no v1 parser, or
/// [`ParserVersion::V2`] and the pair has no unified family.
pub fn validate_parser_version(
    version: ParserVersion,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> anyhow::Result<()> {
    validate_parser_version_for_mode(version, tool_call_parser, reasoning_parser)
}

/// [`validate_parser_version`], and also refuse [`ParserVersion::V2`] with no tool-call
/// parser for a request that needs the v1 jail.
///
/// # Errors
///
/// As [`validate_parser_version`], and the jail case above.
pub fn validate_tool_request_mode(
    version: ParserVersion,
    effective_tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    requires_v1_jail: bool,
) -> anyhow::Result<()> {
    // Raw batch consumers also need model compatibility checks before parsing can fall back.
    validate_parser_version_for_mode(version, effective_tool_call_parser, reasoning_parser)?;
    if version == ParserVersion::V2 && requires_v1_jail && effective_tool_call_parser.is_none() {
        anyhow::bail!(
            "{PARSER_VERSION_ENV}=2 was requested, but this tool choice requires the v1 tool-call jail"
        );
    }
    Ok(())
}

fn validate_parser_version_for_mode(
    version: ParserVersion,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> anyhow::Result<()> {
    match version {
        ParserVersion::Auto => Ok(()),
        ParserVersion::V1 => {
            let tools = dynamo_parsers::tool_calling::parsers::get_available_tool_parsers();
            let reasoning = dynamo_parsers::reasoning::get_available_reasoning_parsers();
            if tool_call_parser.is_some_and(|name| !tools.contains(&name))
                || reasoning_parser.is_some_and(|name| !reasoning.contains(&name))
            {
                anyhow::bail!(
                    "{PARSER_VERSION_ENV}=1 was requested, but the configured parser has no compatible v1 parser",
                );
            }
            Ok(())
        }
        ParserVersion::V2 => {
            if (tool_call_parser.is_some() || reasoning_parser.is_some())
                && !is_v2_configured_family(tool_call_parser, reasoning_parser)
            {
                anyhow::bail!(
                    "{PARSER_VERSION_ENV}=2 was requested, but the configured parser pair has no compatible unified v2 implementation",
                );
            }
            Ok(())
        }
    }
}
