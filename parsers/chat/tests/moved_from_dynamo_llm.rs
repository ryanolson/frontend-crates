// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! dynamo-llm's tests of the moved rules, at ai-dynamo/dynamo `cffc4fa`.
//!
//! Each test names its source. Where the source built an `OpenAIPreprocessor` or ran
//! one process per `DYN_PARSER_VERSION`, the test calls the moved function directly
//! and loops over the versions. Assertions on dynamo-only stages (the stream, the HTTP
//! status, guided decoding) stay in dynamo-llm.

use std::borrow::Cow;

use dynamo_chat_parsing::{
    ParserVersion, ToolProcessingRoute,
    families::{QWEN3_UNIFIED_FAMILY, configured_family, selected_family},
    prefill::{
        prompt_injected_reasoning_start, prompt_injected_reasoning_start_for_parsers,
        prompt_reasoning_prefill_for_parsers,
    },
    route::{
        should_apply_tool_jail, tool_processing_route, validate_parser_version,
        validate_tool_request_mode,
    },
    special_tokens::{parser_requires_special_tokens, special_tokens_will_be_stripped},
    stop_tokens::{remove_tool_parser_end_tokens_from_hidden_stops, request_has_effective_tools},
    tools::{effective_tools, requires_native_tool_call_format, tool_call_parsing_enabled},
};
use dynamo_parsers::tool_calling::ToolChoice;
use dynamo_protocols::types::{
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionRequest,
};
use serde_json::{Value, json};

/// `tool_choice.rs` tests: a base request plus `extra` fields.
fn request(extra: Value) -> CreateChatCompletionRequest {
    let mut value = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}]
    });
    value
        .as_object_mut()
        .expect("base request is an object")
        .extend(extra.as_object().expect("extra is an object").clone());
    serde_json::from_value(value).expect("request must deserialize")
}

fn tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }])
}

/// `preprocessor.rs` `parser_route_test_request`: one user turn and one tool.
fn parser_route_test_request(
    tool_choice: ChatCompletionToolChoiceOption,
) -> CreateChatCompletionRequest {
    let mut request =
        request(json!({"tools": [{"type": "function", "function": {"name": "get_weather"}}]}));
    request.tool_choice = Some(tool_choice);
    request
}

fn named(name: &str) -> ChatCompletionToolChoiceOption {
    serde_json::from_value(json!({"type": "function", "function": {"name": name}})).unwrap()
}

/// Dynamo's tool validation rejects bad names after the merge. Only that part matters
/// to these tests.
fn validate_names(tools: &[ChatCompletionTool]) -> anyhow::Result<()> {
    for (i, tool) in tools.iter().enumerate() {
        let name = &tool.function.name;
        if !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        {
            anyhow::bail!("Function at index {i} has an invalid name: \"{name}\".");
        }
    }
    Ok(())
}

// --- preprocessor.rs ------------------------------------------------------------

/// `eight_family_routes_and_tool_suppression`, route half. The stream half stays.
#[test]
fn eight_family_routes() {
    for version in [ParserVersion::Auto, ParserVersion::V1, ParserVersion::V2] {
        for (tool, reason, family) in [
            (Some("qwen3_coder"), Some("qwen3"), "qwen3"),
            (Some("kimi_k2"), Some("kimi_k25"), "kimi_k2"),
            (Some("kimi-k3"), Some("kimi-k3"), "kimi_k3"),
            (Some("deepseekv4"), Some("deepseek-v4"), "deepseek_v4"),
            (Some("deepseek_v41"), Some("deepseek_v41"), "deepseek_v41"),
            (Some("gemma-4"), Some("gemma4"), "gemma4"),
            (Some("glm47"), Some("glm45"), "glm47"),
            (Some("muse"), None, "muse_glimmer"),
            (None, Some("muse_glimmer"), "muse_glimmer"),
        ] {
            if matches!(family, "gemma4" | "muse_glimmer") {
                let prompt = if family == "gemma4" {
                    "<|channel>"
                } else {
                    "assistant to=self<|message|>"
                };
                assert!(prompt_reasoning_prefill_for_parsers(tool, reason, Some(prompt)).unified());
            }
            for choice in [
                Value::Null,
                json!("auto"),
                json!("none"),
                json!("required"),
                json!({"type":"function","function":{"name":"write_file"}}),
            ] {
                let request: CreateChatCompletionRequest = serde_json::from_value(json!({"model":"test", "messages":[{"role":"user","content":"test"}], "tool_choice":choice, "tools":[{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{}}}}]})).unwrap();
                let route = tool_processing_route(tool, reason, &request, version);
                match selected_family(tool, reason, version) {
                    Some(expected) => assert_eq!(
                        route,
                        Ok(ToolProcessingRoute::Unified(expected)),
                        "{family} {choice} {version:?}"
                    ),
                    None => assert!(
                        !matches!(route, Ok(ToolProcessingRoute::Unified(_))),
                        "{family} {choice} {version:?}"
                    ),
                }
            }
        }
    }
}

/// `all_requested_families_use_unified_serving_route_for_every_tool_policy`. dynamo ran
/// it in a child process per setting; unset, `auto`, and the experimental flag set to
/// `false` resolve to Auto, and `2` and the flag set to `true` resolve to V2.
#[test]
fn all_requested_families_use_unified_serving_route_for_every_tool_policy() {
    for version in [ParserVersion::Auto, ParserVersion::V2] {
        for (tool, reasoning, family) in [
            ("deepseek_v4", "deepseek_v4", "deepseek_v4"),
            ("deepseek_v41", "deepseek_v41", "deepseek_v41"),
            ("gemma4", "gemma4", "gemma4"),
            ("glm47", "glm45", "glm47"),
            ("kimi_k2", "kimi_k25", "kimi_k2"),
            ("kimi_k3", "kimi_k3", "kimi_k3"),
            ("muse", "muse", "muse_glimmer"),
            ("qwen3_coder", "qwen3", "qwen3"),
        ] {
            for choice in [
                ChatCompletionToolChoiceOption::Auto,
                ChatCompletionToolChoiceOption::None,
                ChatCompletionToolChoiceOption::Required,
                named("get_weather"),
            ] {
                let mut request = parser_route_test_request(choice.clone());
                for has_tools in [true, false] {
                    if !has_tools {
                        request.tools = None;
                    }
                    if !has_tools
                        && !matches!(
                            choice,
                            ChatCompletionToolChoiceOption::Auto
                                | ChatCompletionToolChoiceOption::None
                        )
                    {
                        // Forced choices require declared tools at request validation.
                        continue;
                    }
                    assert_eq!(
                        tool_processing_route(Some(tool), Some(reasoning), &request, version),
                        Ok(ToolProcessingRoute::Unified(family)),
                        "{tool}/{reasoning} {choice:?} has_tools={has_tools} {version:?}"
                    );
                }
            }
        }
    }
}

/// `explicit_v2_rejects_modes_that_require_the_v1_jail`. The HTTP status check stays
/// in dynamo-llm, which maps the error to a 400.
#[test]
fn explicit_v2_rejects_modes_that_require_the_v1_jail() {
    for parser in ["muse_glimmer", "qwen3_coder"] {
        let request = parser_route_test_request(ChatCompletionToolChoiceOption::Required);
        assert_eq!(
            tool_processing_route(Some(parser), None, &request, ParserVersion::V2),
            Ok(ToolProcessingRoute::Unified(if parser == "muse_glimmer" {
                "muse_glimmer"
            } else {
                "qwen3"
            })),
        );
    }

    let request = parser_route_test_request(ChatCompletionToolChoiceOption::Required);
    let error = tool_processing_route(None, None, &request, ParserVersion::V2)
        .expect_err("explicit v2 must not construct an unconfigured immediate jail");
    assert!(error.0.contains("requires the v1 tool-call jail"));
    assert_eq!(error.to_string(), error.0);
}

/// `muse_prompt_prefill_follows_request_route`, route and prefill half. dynamo ran it
/// for `auto` and `2`; the stream half stays.
#[test]
fn muse_prompt_prefill_follows_request_route() {
    for version in [ParserVersion::Auto, ParserVersion::V2] {
        for alias in ["muse_glimmer", "muse"] {
            for (tool, reasoning) in [
                (None, Some(alias)),
                (Some("muse_glimmer"), Some(alias)),
                (Some("muse_glimmer"), None),
            ] {
                for choice in [
                    ChatCompletionToolChoiceOption::Required,
                    named("get_weather"),
                    ChatCompletionToolChoiceOption::Auto,
                    ChatCompletionToolChoiceOption::None,
                ] {
                    let request = parser_route_test_request(choice.clone());
                    let route = tool_processing_route(tool, reasoning, &request, version).unwrap();
                    assert!(matches!(route, ToolProcessingRoute::Unified(_)));
                    for (prompt, native_open) in [
                        ("assistant to=self<|message|>", true),
                        ("assistant<think>", false),
                        ("assistant", false),
                    ] {
                        let prefill =
                            prompt_reasoning_prefill_for_parsers(tool, reasoning, Some(prompt));
                        assert_eq!(
                            prefill.for_route(&route),
                            native_open,
                            "{tool:?}/{reasoning:?} {prompt}"
                        );
                    }
                }
            }
        }
    }
}

/// The v1 route uses only the generic opener for Muse names, from
/// `prompt_reasoning_prefill_for_parsers`'s Basic fallback.
#[test]
fn muse_legacy_route_uses_the_generic_opener() {
    let prefill =
        prompt_reasoning_prefill_for_parsers(None, Some("muse"), Some("assistant<think>"));
    assert!(prefill.for_route(&ToolProcessingRoute::PassThrough));
    assert!(!prefill.for_route(&ToolProcessingRoute::Unified("muse_glimmer")));
    let native = prompt_reasoning_prefill_for_parsers(
        None,
        Some("muse"),
        Some("assistant to=self<|message|>"),
    );
    assert!(!native.for_route(&ToolProcessingRoute::LegacyJail(None)));
    assert!(native.for_route(&ToolProcessingRoute::Unified("muse_glimmer")));
}

/// `effective_tool_predicates_match_protocol_definition`.
#[test]
fn effective_tool_predicates_match_protocol_definition() {
    let cases = [
        (
            "no tools",
            json!({"model": "test-model", "messages": [{"role": "user", "content": "test"}]}),
        ),
        (
            "empty top-level tools",
            json!({"model": "test-model", "messages": [{"role": "user", "content": "test"}], "tools": []}),
        ),
        (
            "top-level tools",
            json!({"model": "test-model", "messages": [{"role": "user", "content": "test"}], "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]}),
        ),
        (
            "empty system tools",
            json!({"model": "test-model", "messages": [{"role": "system", "content": "", "tools": []}, {"role": "user", "content": "test"}]}),
        ),
        (
            "system tools",
            json!({"model": "test-model", "messages": [{"role": "system", "content": "", "tools": [{"name": "lookup"}]}, {"role": "user", "content": "test"}]}),
        ),
        (
            "later system tools",
            json!({"model": "test-model", "messages": [{"role": "system", "content": "", "tools": []}, {"role": "user", "content": "test"}, {"role": "system", "content": "", "tools": [{"name": "lookup"}]}]}),
        ),
    ];
    for (case, value) in cases {
        let request: CreateChatCompletionRequest =
            serde_json::from_value(value).expect("request must deserialize");
        assert_eq!(
            request_has_effective_tools(&request),
            request.has_effective_tools(),
            "effective-tool predicates diverged for {case}"
        );
    }

    let developer_tools = serde_json::from_value::<CreateChatCompletionRequest>(json!({
        "model": "test-model",
        "messages": [
            {"role": "developer", "content": "policy", "tools": [{"name": "lookup"}]},
            {"role": "user", "content": "test"}
        ]
    }));
    match developer_tools {
        Ok(request) => assert_eq!(
            request_has_effective_tools(&request),
            request.has_effective_tools(),
            "generic predicate must track newly supported developer tools"
        ),
        Err(error) => assert!(
            error
                .to_string()
                .contains("`tools` is only accepted on system messages, not on role developer"),
            "unexpected developer-tools rejection: {error}"
        ),
    }
}

const HARMONY_CALL_TOKEN_ID: u32 = 42;

/// `HiddenStopTokenizer`: only Harmony's call marker is expected.
fn harmony_encode(input: &str) -> anyhow::Result<Vec<u32>> {
    if input != "<|call|>" {
        anyhow::bail!("unexpected Harmony tool-call end token {input:?}");
    }
    Ok(vec![HARMONY_CALL_TOKEN_ID])
}

/// `hidden_stop_path_recognizes_dynamic_system_tools_and_tool_choice_none`.
#[test]
fn hidden_stop_path_recognizes_dynamic_system_tools_and_tool_choice_none() {
    let dynamic: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "moonshotai/Kimi-K3",
        "messages": [
            {"role": "system", "content": "", "tools": [{"name": "lookup"}]},
            {"role": "user", "content": "test"}
        ]
    }))
    .unwrap();
    let none: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "moonshotai/Kimi-K3",
        "messages": [
            {"role": "system", "content": "", "tools": [{"name": "lookup"}]},
            {"role": "user", "content": "test"}
        ],
        "tool_choice": "none"
    }))
    .unwrap();
    assert!(dynamic.tools.is_none());
    assert!(none.tools.is_none());

    let mut dynamic_hidden = vec![HARMONY_CALL_TOKEN_ID, 7];
    let dynamic_visible = remove_tool_parser_end_tokens_from_hidden_stops(
        Some("harmony"),
        &dynamic,
        &mut dynamic_hidden,
        harmony_encode,
    )
    .unwrap();
    assert_eq!(dynamic_visible, [HARMONY_CALL_TOKEN_ID]);
    assert_eq!(dynamic_hidden, [7]);

    let mut none_hidden = vec![HARMONY_CALL_TOKEN_ID, 7];
    let none_visible = remove_tool_parser_end_tokens_from_hidden_stops(
        Some("harmony"),
        &none,
        &mut none_hidden,
        harmony_encode,
    )
    .unwrap();
    assert!(none_visible.is_empty());
    assert_eq!(none_hidden, [HARMONY_CALL_TOKEN_ID, 7]);
}

/// `test_parser_requires_special_tokens` (PRE.1), at the default version.
#[test]
fn test_parser_requires_special_tokens() {
    let cases: &[(Option<&str>, Option<&str>, bool, &str)] = &[
        (
            Some("gemma4"),
            None,
            true,
            "gemma4 tool-call only → required",
        ),
        (
            None,
            Some("gemma4"),
            true,
            "gemma4 reasoning only → required",
        ),
        (
            Some("gemma-4"),
            None,
            true,
            "gemma-4 hyphen alias (tool) → required",
        ),
        (
            None,
            Some("gemma-4"),
            true,
            "gemma-4 hyphen alias (reasoning) → required",
        ),
        (
            Some("gemma4"),
            Some("gemma4"),
            true,
            "gemma4 paired → required",
        ),
        (Some("hermes"), None, false, "hermes → not required"),
        (
            Some("harmony"),
            None,
            true,
            "harmony tool-call only → required",
        ),
        (
            None,
            Some("gpt_oss"),
            true,
            "gpt_oss reasoning only → required",
        ),
        (
            Some("harmony"),
            Some("gpt_oss"),
            true,
            "harmony + gpt_oss paired → required",
        ),
        (
            Some("kimi_k2"),
            Some("kimi_k25"),
            true,
            "kimi_k2 + kimi_k25 paired → required",
        ),
        (
            None,
            Some("kimi_k25"),
            true,
            "kimi_k25 reasoning only → required (`</think>` is special token id 163607)",
        ),
        (
            Some("kimi_k3"),
            Some("kimi_k3"),
            true,
            "kimi_k3 paired XTML parsers → required",
        ),
        (
            Some("kimi-k3"),
            Some("kimi-k3"),
            true,
            "kimi-k3 aliases → required",
        ),
        (
            None,
            Some("mistral"),
            true,
            "mistral reasoning only → required (`[THINK]` / `[/THINK]` are special tokens)",
        ),
        (
            Some("kimi_k2"),
            None,
            true,
            "kimi_k2 tool-call only → required",
        ),
        (
            Some("minimax_m3"),
            Some("minimax_m3"),
            true,
            "minimax_m3 paired → required",
        ),
        (
            Some("minimax-m3-nom"),
            Some("minimax-m3"),
            true,
            "MiniMax M3 SGLang aliases → required",
        ),
        (
            Some("muse_glimmer"),
            None,
            true,
            "muse_glimmer tool-call only → required",
        ),
        (
            None,
            Some("muse_glimmer"),
            true,
            "muse_glimmer reasoning-name only → required",
        ),
        (
            Some("muse"),
            None,
            true,
            "muse (SGLang's registered name, tool) → required",
        ),
        (None, None, false, "no parsers → not required"),
    ];
    for (tool, reasoning, expected, desc) in cases {
        assert_eq!(
            parser_requires_special_tokens(*tool, *reasoning, ParserVersion::Auto).unwrap(),
            *expected,
            "FAILED: {desc}",
        );
    }
}

/// `deepseek_v41_preserves_markers_and_initializes_backend_reasoning`, special-token
/// half.
#[test]
fn deepseek_v41_preserves_markers() {
    for (tool, reasoning) in [(Some("deepseek_v41"), None), (None, Some("deepseek_v41"))] {
        for version in [ParserVersion::Auto, ParserVersion::V1, ParserVersion::V2] {
            assert!(parser_requires_special_tokens(tool, reasoning, version).unwrap());
        }
    }
}

/// `test_special_tokens_will_be_stripped`.
#[test]
fn test_special_tokens_will_be_stripped() {
    for skip in [None, Some(false), Some(true)] {
        for required in [false, true] {
            assert_eq!(
                special_tokens_will_be_stripped(skip, required),
                skip == Some(true) && required,
            );
        }
    }
}

/// `test_kimi_k3_jail_always_unwraps_xtml_response_channels`.
#[test]
fn test_kimi_k3_jail_always_unwraps_xtml_response_channels() {
    let parser = "kimi_k3".to_string();
    assert!(should_apply_tool_jail(Some(&parser), None, false));

    let alias = "kimi-k3".to_string();
    assert!(should_apply_tool_jail(
        Some(&alias),
        Some(&ChatCompletionToolChoiceOption::None),
        true,
    ));
}

/// `test_prompt_injected_reasoning_start_by_parser`.
#[test]
fn test_prompt_injected_reasoning_start_by_parser() {
    let cases = [
        (
            Some("minimax_m3"),
            Some("...<mm:think>\n"),
            true,
            "MiniMax M3 starts from <mm:think>",
        ),
        (
            Some("minimax-m3"),
            Some("...</mm:think>\n"),
            false,
            "MiniMax M3 prefilled end marker means not in reasoning",
        ),
        (
            Some("minimax_m3"),
            Some("...<think>\n"),
            false,
            "MiniMax M3 must not use generic <think>",
        ),
        (
            Some("qwen3"),
            Some("...<think>\n"),
            true,
            "Qwen-style templated <think> behavior remains",
        ),
        (
            Some("kimi_k3"),
            Some("...<|open|>think<|sep|>\n"),
            true,
            "Kimi K3 canonical XTML prompt retains legacy behavior",
        ),
        (
            Some("kimi-k3"),
            Some("...<|open|> think <|sep|>\n"),
            true,
            "Kimi K3 ignores whitespace inside the suffix",
        ),
        (
            Some("kimi-k3"),
            Some("...<|open|>\u{2003}think\t<|sep|>\n"),
            true,
            "Kimi K3 ignores Unicode whitespace inside the suffix",
        ),
        (
            Some("kimi_k3"),
            Some("think<|sep|>"),
            false,
            "Kimi K3 requires the complete suffix",
        ),
        (
            Some("kimi-k3"),
            Some("...<think>\n"),
            false,
            "Kimi K3 must not use the generic think marker",
        ),
        (
            Some("deepseek_v4"),
            Some("...<think>\n"),
            true,
            "existing <think> parser behavior remains",
        ),
        (
            None,
            Some("...<think>\n"),
            true,
            "legacy no-parser detection",
        ),
        (Some("minimax_m3"), None, false, "no prompt"),
    ];
    for (parser, prompt, expected, desc) in cases {
        assert_eq!(
            prompt_injected_reasoning_start(parser, prompt),
            expected,
            "FAILED: {desc}"
        );
    }
}

/// `tool_only_unified_family_detects_its_prompt_injected_reasoning_marker`.
#[test]
fn tool_only_unified_family_detects_its_prompt_injected_reasoning_marker() {
    let cases = [
        (
            "gemma4",
            "...<|channel>thought",
            "Gemma 4 tool-only routing uses its thought channel",
        ),
        (
            "kimi_k3",
            "...<|open|>think<|sep|>",
            "Kimi K3 tool-only routing uses its think channel",
        ),
        (
            "muse_glimmer",
            "...assistant to=self<|message|>",
            "Muse tool-only routing uses its thought message",
        ),
    ];
    for (family, prompt, description) in cases {
        assert!(
            prompt_injected_reasoning_start_for_parsers(Some(family), None, Some(prompt)),
            "{description}"
        );
    }
    assert!(
        !prompt_injected_reasoning_start_for_parsers(
            Some("gemma4"),
            Some("qwen3"),
            Some("...<|channel>thought"),
        ),
        "mismatched parser families must keep the configured reasoning parser"
    );
}

// --- unified_parser.rs ----------------------------------------------------------

/// `deepseek_v41_uses_native_parsing_without_an_experimental_flag`.
#[test]
fn deepseek_v41_uses_native_parsing_without_an_experimental_flag() {
    let pair = (Some("deepseek_v41"), Some("deepseek_v41"));
    assert_eq!(configured_family(pair.0, pair.1), Some("deepseek_v41"));
    for version in [ParserVersion::Auto, ParserVersion::V1, ParserVersion::V2] {
        assert_eq!(
            selected_family(pair.0, pair.1, version),
            (version != ParserVersion::V1).then_some("deepseek_v41")
        );
    }
    assert_eq!(configured_family(Some("deepseek_v41"), Some("qwen3")), None);
    assert_eq!(
        configured_family(Some("deepseek_v41"), None),
        Some("deepseek_v41")
    );
    assert_eq!(
        configured_family(None, Some("deepseek_v41")),
        Some("deepseek_v41")
    );
}

/// `pairs_only_qwen3_coder_with_qwen3`.
#[test]
fn pairs_only_qwen3_coder_with_qwen3() {
    assert_eq!(
        configured_family(Some("qwen3_coder"), Some("qwen3")),
        Some(QWEN3_UNIFIED_FAMILY)
    );
    // A single configured half selects the same family.
    assert_eq!(configured_family(Some("qwen3_coder"), None), Some("qwen3"));
    assert_eq!(configured_family(None, Some("qwen3")), Some("qwen3"));
    assert_eq!(configured_family(None, None), None);
    // Right tool parser, wrong reasoning family.
    assert_eq!(
        configured_family(Some("qwen3_coder"), Some("deepseek_r1")),
        None
    );
    // Right reasoning parser, wrong tool family.
    assert_eq!(configured_family(Some("kimi_k2"), Some("qwen3")), None);
    // The canonical name and Dynamo tool alias select the same owner.
    assert_eq!(
        configured_family(Some("qwen3"), Some("qwen3")),
        Some("qwen3")
    );
}

/// `batch_routing_uses_the_configured_pair_for_every_output_mode`.
#[test]
fn batch_routing_uses_the_configured_pair_for_every_output_mode() {
    assert_eq!(
        configured_family(Some("qwen3_coder"), Some("qwen3")),
        Some(QWEN3_UNIFIED_FAMILY),
        "the carried constraint selects native versus guided parsing after routing"
    );
}

/// `gpt_oss_aliases_select_the_unified_route_unless_v1`.
#[test]
fn gpt_oss_aliases_select_the_unified_route_unless_v1() {
    for (tool, reasoning) in [
        (Some("harmony"), Some("gpt_oss")),
        (Some("harmony"), None),
        (None, Some("gpt_oss")),
    ] {
        assert_eq!(configured_family(tool, reasoning), Some("gpt_oss"));
        for (version, expected) in [
            (ParserVersion::Auto, Some("gpt_oss")),
            (ParserVersion::V1, None),
            (ParserVersion::V2, Some("gpt_oss")),
        ] {
            assert_eq!(selected_family(tool, reasoning, version), expected);
        }
    }
}

/// Qwen3's reasoning name alone keeps the v1 route at Auto, because Qwen3 thinking
/// models also pair it with Hermes tool syntax. An explicit V2 still selects the family.
#[test]
fn qwen3_reasoning_alone_stays_on_v1_unless_v2_is_explicit() {
    assert_eq!(
        selected_family(None, Some("qwen3"), ParserVersion::Auto),
        None
    );
    assert_eq!(
        selected_family(None, Some("qwen3"), ParserVersion::V2),
        Some("qwen3")
    );
    assert_eq!(
        selected_family(None, Some("qwen3"), ParserVersion::V1),
        None
    );
    assert_eq!(
        selected_family(Some("hermes"), Some("qwen3"), ParserVersion::Auto),
        None
    );
}

// --- tool_parser_v2.rs ----------------------------------------------------------

/// `parser_versions_accept_numeric_values_and_reject_aliases`.
#[test]
fn parser_versions_accept_numeric_values_and_reject_aliases() {
    for (input, expected) in [
        (None, ParserVersion::Auto),
        (Some("auto"), ParserVersion::Auto),
        (Some("1"), ParserVersion::V1),
        (Some("2"), ParserVersion::V2),
    ] {
        assert_eq!(ParserVersion::try_from(input).unwrap(), expected);
    }
    for value in ["", "0", "3", "v1", "v2", "v3", "true", "V2"] {
        assert!(ParserVersion::try_from(Some(value)).is_err());
    }
    for (tool, reasoning) in [
        ("deepseek_v4", "deepseek_v4"),
        ("gemma4", "gemma4"),
        ("glm47", "glm45"),
        ("kimi_k2", "kimi_k25"),
        ("kimi_k3", "kimi_k3"),
        ("qwen3_coder", "qwen3"),
    ] {
        assert!(
            validate_parser_version(ParserVersion::V1, Some(tool), Some(reasoning)).is_ok(),
            "{tool}/{reasoning}"
        );
        assert!(
            validate_parser_version(ParserVersion::V2, Some(tool), Some(reasoning)).is_ok(),
            "{tool}/{reasoning}"
        );
    }
    assert!(
        validate_parser_version(
            ParserVersion::V1,
            Some("deepseek_v41"),
            Some("deepseek_v41")
        )
        .is_err()
    );
}

/// `parser_version_rejects_unavailable_generations`. Its `supports_family` checks
/// belong to the batch path, which stays in dynamo-llm.
#[test]
fn parser_version_rejects_unavailable_generations() {
    assert!(validate_parser_version(ParserVersion::V1, Some("muse_glimmer"), None).is_err());
    assert!(validate_parser_version(ParserVersion::V1, None, Some("muse")).is_err());
    assert!(validate_parser_version(ParserVersion::V1, Some("qwen3_coder"), Some("qwen3")).is_ok());
    assert!(validate_parser_version(ParserVersion::Auto, Some("muse_glimmer"), None).is_ok());
    assert!(validate_parser_version(ParserVersion::V2, Some("muse_glimmer"), None).is_ok());
    assert!(validate_parser_version(ParserVersion::V2, Some("qwen3_coder"), Some("qwen3")).is_ok());
    assert!(validate_parser_version(ParserVersion::V2, None, Some("muse_glimmer")).is_ok());
    assert!(
        validate_parser_version(
            ParserVersion::V2,
            Some("muse_glimmer"),
            Some("muse_glimmer")
        )
        .is_ok()
    );
    assert!(
        validate_parser_version(ParserVersion::V2, Some("qwen3_coder"), Some("hermes")).is_err()
    );
    assert!(validate_parser_version(ParserVersion::V2, Some("hermes"), None).is_err());
    assert!(validate_tool_request_mode(ParserVersion::V2, None, None, true).is_err());
    for alias in ["deepseek_v4", "deepseek-v4", "deepseekv4"] {
        assert!(validate_parser_version(ParserVersion::V2, Some(alias), None).is_ok());
    }
}

// --- tool_choice.rs -------------------------------------------------------------

/// `tool_call_parsing_requires_tools`.
#[test]
fn tool_call_parsing_requires_tools() {
    assert!(!tool_call_parsing_enabled(&request(json!({}))));
    assert!(!tool_call_parsing_enabled(&request(
        json!({"tool_choice": "required"})
    )));
}

/// `tool_call_parsing_honors_each_tool_choice`.
#[test]
fn tool_call_parsing_honors_each_tool_choice() {
    for (tool_choice, expected) in [
        (json!(null), true),
        (json!("none"), false),
        (json!("auto"), true),
        (json!("required"), true),
        (
            json!({"type": "function", "function": {"name": "get_weather"}}),
            true,
        ),
    ] {
        let mut extra = json!({"tools": tools()});
        if !tool_choice.is_null() {
            extra["tool_choice"] = tool_choice;
        }
        assert_eq!(tool_call_parsing_enabled(&request(extra)), expected);
    }
}

/// `effective_tools_borrows_top_level_tools_without_dynamic_declarations`.
#[test]
fn effective_tools_borrows_top_level_tools_without_dynamic_declarations() {
    let request = request(json!({"tools": tools()}));
    let top_level = request.tools.as_deref().unwrap();

    let effective =
        effective_tools(&request, validate_names).expect("top-level tools must validate");

    assert!(matches!(&effective, Cow::Borrowed(_)));
    assert!(std::ptr::eq(effective.as_ptr(), top_level.as_ptr()));
}

/// `dynamic_system_tools_enable_parsing_and_honor_tool_choice_none`.
#[test]
fn dynamic_system_tools_enable_parsing_and_honor_tool_choice_none() {
    for (tool_choice, expected) in [(None, true), (Some(json!("none")), false)] {
        let mut extra = json!({
            "messages": [
                {"role": "system", "content": "", "tools": [{"name": "lookup", "parameters": {"type": "object"}}]},
                {"role": "user", "content": "look it up"}
            ]
        });
        if let Some(tool_choice) = tool_choice {
            extra["tool_choice"] = tool_choice;
        }
        assert_eq!(tool_call_parsing_enabled(&request(extra)), expected);
    }
}

/// `effective_tools_preserve_order_and_normalize_wrapped_and_bare_shapes`.
#[test]
fn effective_tools_preserve_order_and_normalize_wrapped_and_bare_shapes() {
    let request = request(json!({
        "tools": [{"type": "function", "function": {"name": "static_tool", "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "user", "content": "start"},
            {
                "role": "system",
                "content": "",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "wrapped_dynamic",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                            "additionalProperties": false
                        },
                        "strict": true
                    }
                }]
            },
            {
                "role": "system",
                "content": "",
                "tools": [{
                    "name": "bare_dynamic",
                    "description": "bare form",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": false,
                    "vendor_hint": "kept only in the original message"
                }]
            },
            {"role": "user", "content": "continue"}
        ]
    }));

    let normalized =
        effective_tools(&request, validate_names).expect("dynamic tools must normalize");
    assert!(matches!(&normalized, Cow::Owned(_)));
    assert_eq!(
        normalized
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<Vec<_>>(),
        ["static_tool", "wrapped_dynamic", "bare_dynamic"]
    );
    assert_eq!(normalized[1].function.strict, Some(true));
    assert_eq!(normalized[2].function.strict, Some(false));
    assert_eq!(
        normalized[2].function.description.as_deref(),
        Some("bare form")
    );
    let original = serde_json::to_value(&request.messages).unwrap();
    assert_eq!(
        original[2]["tools"][0]["vendor_hint"], "kept only in the original message",
        "normalization must not rewrite dynamic message declarations"
    );
}

/// `effective_tools_reject_invalid_dynamic_metadata`. Its bad-name case comes from
/// the caller's validation, which this crate runs last.
#[test]
fn effective_tools_reject_invalid_dynamic_metadata() {
    for (field, value) in [
        ("strict", json!("yes")),
        ("parameters", json!([])),
        ("description", json!(7)),
    ] {
        let mut tool = json!({"name": "lookup"});
        tool[field] = value;
        let request = request(json!({
            "messages": [
                {"role": "system", "content": "", "tools": [tool]},
                {"role": "user", "content": "go"}
            ]
        }));
        let error = effective_tools(&request, validate_names).expect_err("invalid field must fail");
        assert!(error.to_string().contains(field), "{field}: {error}");
    }

    let request = request(json!({
        "messages": [
            {"role": "system", "content": "", "tools": [{"name": "bad name"}]},
            {"role": "user", "content": "go"}
        ]
    }));
    let error = effective_tools(&request, validate_names).expect_err("invalid name must fail");
    assert!(error.to_string().contains("has an invalid name"));
    // Without the caller's validation the merge itself accepts the name.
    assert!(effective_tools(&request, |_| Ok(())).is_ok());
}

/// `assistant_constraints_do_not_revoke_auto_tool_permission`.
#[test]
fn assistant_constraints_do_not_revoke_auto_tool_permission() {
    let constraints = [
        json!({"response_format": {"type": "json_object"}}),
        json!({"response_format": {"type": "json_schema", "json_schema": {"name": "event", "schema": {"type": "object"}}}}),
        json!({"guided_json": {"type": "object"}}),
        json!({"guided_regex": "[a-z]+"}),
        json!({"guided_choice": ["a", "b"]}),
        json!({"guided_grammar": "root ::= 'a'"}),
    ];
    for constraint in constraints {
        for choice in ["auto", "required"] {
            let mut extra = json!({"tools": tools(), "tool_choice": choice});
            extra
                .as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(tool_call_parsing_enabled(&request(extra)));
        }
    }
}

/// `text_response_format_does_not_disable_auto_tool_parsing`.
#[test]
fn text_response_format_does_not_disable_auto_tool_parsing() {
    assert!(tool_call_parsing_enabled(&request(json!({
        "tools": tools(),
        "tool_choice": "auto",
        "response_format": {"type": "text"}
    }))));
}

// --- structural_tag.rs ----------------------------------------------------------

/// `kimi_forced_choices_are_classified_as_native_format_fallbacks`.
#[test]
fn kimi_forced_choices_are_classified_as_native_format_fallbacks() {
    let named = ToolChoice::Named("get_weather".to_string());
    assert!(requires_native_tool_call_format(
        Some("kimi_k2"),
        &ToolChoice::Required
    ));
    assert!(requires_native_tool_call_format(Some("kimi_k2"), &named));
    assert!(requires_native_tool_call_format(
        Some("kimi_k3"),
        &ToolChoice::Required
    ));
    assert!(requires_native_tool_call_format(Some("kimi_k3"), &named));
    assert!(requires_native_tool_call_format(Some("kimi-k3"), &named));
    assert!(!requires_native_tool_call_format(
        Some("kimi_k3"),
        &ToolChoice::Auto
    ));
    assert!(!requires_native_tool_call_format(
        Some("hermes"),
        &ToolChoice::Named("get_weather".to_string())
    ));
    assert!(!requires_native_tool_call_format(
        Some("kimi_k2"),
        &ToolChoice::Auto
    ));
}
