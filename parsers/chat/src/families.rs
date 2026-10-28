// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which unified v2 family a configured parser pair selects.
//!
//! A deployment names a tool-call parser and a reasoning parser. When both name one
//! `dynamo-parsers-v2` unified family, directly or through an alias, that family owns
//! reasoning, text, and tool calls through one state machine. Otherwise the request
//! keeps the v1 reasoning parser and the v1 tool-call jail.

use std::sync::LazyLock;

use crate::version::ParserVersion;

/// The `dynamo-parsers-v2` unified family that serves Qwen3.
///
/// `REGISTERED_UNIFIED_FAMILIES` accepts both `qwen3` and the `qwen3_coder` alias for
/// the same XML grammar; `qwen3` is the canonical registry name and the one the
/// conformance corpus uses, so it is what this module passes and logs.
pub const QWEN3_UNIFIED_FAMILY: &str = "qwen3";
pub const KIMI_K2_UNIFIED_FAMILY: &str = "kimi_k2";
pub const KIMI_K3_UNIFIED_FAMILY: &str = "kimi_k3";
pub const DEEPSEEK_V41_UNIFIED_FAMILY: &str = "deepseek_v41";
pub const DEEPSEEK_V4_UNIFIED_FAMILY: &str = "deepseek_v4";
pub const GEMMA4_UNIFIED_FAMILY: &str = "gemma4";
pub const GLM47_UNIFIED_FAMILY: &str = "glm47";
pub const MUSE_UNIFIED_FAMILY: &str = "muse_glimmer";

/// Dynamo's `--dyn-tool-call-parser` name that pairs into [`QWEN3_UNIFIED_FAMILY`].
pub const QWEN3_TOOL_CALL_PARSER: &str = "qwen3_coder";

/// Dynamo's parser names that the v2 registry does not know, and the family each one
/// selects. Every lookup goes through [`canonical_family`], so registration, stream,
/// and batch agree on them.
pub const DYNAMO_FAMILY_ALIASES: &[(&str, &str)] = &[
    ("deepseek-v4", DEEPSEEK_V4_UNIFIED_FAMILY),
    ("deepseekv4", DEEPSEEK_V4_UNIFIED_FAMILY),
    ("muse", MUSE_UNIFIED_FAMILY),
    ("gemma-4", GEMMA4_UNIFIED_FAMILY),
    ("glm45", GLM47_UNIFIED_FAMILY),
    ("kimi_k25", KIMI_K2_UNIFIED_FAMILY),
];

/// Canonical family names and aliases share one route for registration, stream, and batch.
pub fn canonical_family(parser: &str) -> Option<&'static str> {
    let parser = DYNAMO_FAMILY_ALIASES
        .iter()
        .find_map(|(alias, family)| (*alias == parser).then_some(*family))
        .unwrap_or(parser);
    dynamo_parsers_v2::canonical_unified_family(parser)
}

static FAMILY_NAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut names = dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES.to_vec();
    names.extend(DYNAMO_FAMILY_ALIASES.iter().map(|(alias, _)| *alias));
    names.sort_unstable();
    names.dedup();
    names
});

/// Every parser name that selects a unified family, registry names and Dynamo
/// aliases, sorted.
pub fn unified_family_names() -> &'static [&'static str] {
    &FAMILY_NAMES
}

/// A configured half may select a unified family, but two configured halves must agree.
pub fn configured_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    match (tool_call_parser, reasoning_parser) {
        (Some(tool), Some(reasoning)) => {
            let family = canonical_family(tool)?;
            (canonical_family(reasoning)? == family).then_some(family)
        }
        (Some(parser), None) | (None, Some(parser)) => canonical_family(parser),
        (None, None) => None,
    }
}

/// Whether the parser pair has a unified v2 implementation, whatever the version.
pub fn is_v2_configured_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> bool {
    configured_family(tool_call_parser, reasoning_parser).is_some()
}

/// The unified family to actually use for this parser pair, or `None` to keep the
/// existing split reasoning-parser + tool-call-jail path.
pub fn selected_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    version: ParserVersion,
) -> Option<&'static str> {
    let configured = configured_family(tool_call_parser, reasoning_parser);
    // Qwen3 thinking models also use this reasoning name with Hermes tool syntax.
    let selected = if tool_call_parser.is_none()
        && reasoning_parser == Some("qwen3")
        && version == ParserVersion::Auto
    {
        None
    } else {
        selected_family_for_selection(configured, version)
    };
    tracing::debug!(
        target: "dynamo_unified",
        ?tool_call_parser,
        ?reasoning_parser,
        ?configured,
        selection = ?version,
        "unified parser path decision"
    );
    selected
}

fn selected_family_for_selection(
    configured: Option<&'static str>,
    selection: ParserVersion,
) -> Option<&'static str> {
    match selection {
        ParserVersion::Auto | ParserVersion::V2 => configured,
        ParserVersion::V1 => None,
    }
}
