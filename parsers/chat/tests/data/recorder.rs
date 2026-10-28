// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The recorder that wrote `dynamo_llm_parity.jsonl`. This crate does not build it.
//!
//! It ran as a test inside dynamo-llm at ai-dynamo/dynamo `cffc4fa`, as
//! `lib/llm/src/preprocessor/parity_recorder.rs`, with `mod parity_recorder;` under
//! `#[cfg(test)]` in `preprocessor.rs`. Run it once per `DYN_PARSER_VERSION` value,
//! because dynamo-llm reads the version once per process:
//!
//! ```text
//! PARITY_OUT=auto.jsonl cargo test -p dynamo-llm --lib parity_recorder
//! DYN_PARSER_VERSION=1 PARITY_OUT=1.jsonl cargo test -p dynamo-llm --lib parity_recorder
//! DYN_PARSER_VERSION=2 PARITY_OUT=2.jsonl cargo test -p dynamo-llm --lib parity_recorder
//! python3 compact.py dynamo_llm_parity.jsonl auto.jsonl 1.jsonl 2.jsonl
//! ```

use std::io::Write;
use std::sync::Arc;

use serde_json::{Value, json};

use super::{OpenAIPreprocessor, PromptReasoningPrefill, ToolProcessingRoute};
use crate::model_card::ModelDeploymentCard;
use crate::preprocessor::structural_tag::requires_native_tool_call_format;
use crate::preprocessor::tool_choice;
use crate::protocols::openai::GuidedToolConstraint;
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, tool_parser_v2, unified_parser,
};
use crate::tokenizers::Encoding;
use crate::tokenizers::traits::Tokenizer;
use dynamo_parsers::tool_calling::ToolChoice;
use dynamo_protocols::types::{ChatCompletionToolChoiceOption, ChatCompletionNamedToolChoice};
use dynamo_runtime::error::DynamoError;

fn fnv1a32(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

/// A deterministic stand-in tokenizer. The replay side computes the same ids.
fn fake_encode(s: &str) -> Vec<u32> {
    let h = fnv1a32(s) % 100_000 + 1_000;
    if s.len() % 3 == 2 { vec![h, h + 1] } else { vec![h] }
}

struct FakeTokenizer;

impl crate::tokenizers::traits::Encoder for FakeTokenizer {
    fn encode(&self, input: &str) -> anyhow::Result<Encoding> {
        Ok(Encoding::Sp(fake_encode(input)))
    }
    fn encode_batch(&self, inputs: &[&str]) -> anyhow::Result<Vec<Encoding>> {
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

impl crate::tokenizers::traits::Decoder for FakeTokenizer {
    fn decode(
        &self,
        _token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> anyhow::Result<crate::tokenizers::traits::DecodeResult> {
        Ok(crate::tokenizers::traits::DecodeResult::Complete(String::new()))
    }
}

impl Tokenizer for FakeTokenizer {}

fn err_json(error: &anyhow::Error) -> Value {
    match error.downcast_ref::<DynamoError>() {
        Some(dynamo) => json!({"invalid_argument": dynamo.message()}),
        None => json!({"error": format!("{error:#}")}),
    }
}

fn names() -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    names.extend(dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES.iter().map(|s| s.to_string()));
    names.extend(unified_parser::FAMILY_NAMES.iter().map(|s| s.to_string()));
    names.extend(
        dynamo_parsers::tool_calling::parsers::get_available_tool_parsers()
            .into_iter()
            .map(str::to_string),
    );
    names.extend(
        dynamo_parsers::reasoning::get_available_reasoning_parsers()
            .into_iter()
            .map(str::to_string),
    );
    // Names that only exist on a newer registry, names that literal rules match, and junk.
    for extra in [
        "muse_spark",
        "muse_spark_open",
        "muse",
        "muse_glimmer",
        "deepseek_v41",
        "deepseek-v41",
        "gpt_oss",
        "harmony",
        "qwen3",
        "qwen3_coder",
        "hermes",
        "basic",
        "beam",
        "kimi_k25",
        "kimi_k3",
        "kimi-k3",
        "minimax_m2",
        "minimax_m3",
        "minimax-m3",
        "minimax_m3_nom",
        "minimax-m3-nom",
        "minimax_append_think",
        "inkling",
        "mistral",
        "gemma4",
        "gemma-4",
        "glm45",
        "glm47",
        "nemotron3",
        "nemotron_v3",
        "nemotron_nano",
        "nemotron_deci",
        "",
        "junk",
        "GLM45",
        "glm45 ",
    ] {
        names.push(extra.to_string());
    }
    names.sort();
    names.dedup();
    names
}

fn opt_names() -> Vec<Option<String>> {
    std::iter::once(None).chain(names().into_iter().map(Some)).collect()
}

const PROMPTS: &[Option<&str>] = &[
    None,
    Some(""),
    Some(" "),
    Some("<think>"),
    Some("<think>\n"),
    Some("<think> \n\t"),
    Some("x<think>"),
    Some("<think>x"),
    Some("</think>"),
    Some("<mm:think>"),
    Some("<mm:think>\n"),
    Some("<|open|>think<|sep|>"),
    Some("<|open|> think <|sep|>\n"),
    Some("<|open|>think"),
    Some("<|channel>thought"),
    Some("<|channel>thought\n"),
    Some("<|channel>thou"),
    Some("<|channel>"),
    Some("<|channel>t"),
    Some("assistant to=self<|message|>"),
    Some("assistant to=self<|message|>\n"),
    Some("<|message|>"),
    Some("hello"),
];

fn prefill_json(prefill: PromptReasoningPrefill) -> Value {
    json!([prefill.legacy, prefill.unified, prefill.prefix])
}

fn tool_choice_values() -> Vec<Option<Value>> {
    vec![
        None,
        Some(json!("none")),
        Some(json!("auto")),
        Some(json!("required")),
        Some(json!({"type": "function", "function": {"name": "lookup"}})),
    ]
}

/// Requests for the route, the tool gate, and the hidden-stop rule.
fn route_requests() -> Vec<Value> {
    let mut out = Vec::new();
    for choice in tool_choice_values() {
        for tools in [
            None,
            Some(json!([])),
            Some(json!([{"type": "function", "function": {"name": "lookup"}}])),
        ] {
            for system_tools in [
                None,
                Some(json!([])),
                Some(json!([{"name": "lookup"}])),
                Some(json!([{"type": "function", "function": {"name": "sys_tool"}}])),
            ] {
                let mut system = json!({"role": "system", "content": "be brief"});
                if let Some(system_tools) = system_tools {
                    system["tools"] = system_tools;
                }
                let mut request = json!({
                    "model": "m",
                    "messages": [system, {"role": "user", "content": "hi"}],
                });
                if let Some(tools) = &tools {
                    request["tools"] = tools.clone();
                }
                if let Some(choice) = &choice {
                    request["tool_choice"] = choice.clone();
                }
                out.push(request);
            }
        }
    }
    out
}

/// Requests for effective-tool normalization: wrapped and bare system tools and the
/// metadata errors.
fn tools_requests() -> Vec<Value> {
    let system_variants = vec![
        None,
        Some(json!([])),
        Some(json!([{"name": "bare"}])),
        Some(json!([{"name": "bare", "description": "d", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}, "strict": false}])),
        Some(json!([{"type": "function", "function": {"name": "wrapped", "description": "w", "parameters": {"type": "object"}}}])),
        Some(json!([{"name": "a"}, {"type": "function", "function": {"name": "b"}}])),
        Some(json!([{"name": "n", "parameters": null, "description": null, "strict": null}])),
        Some(json!([{"name": "n", "parameters": {"type": "object", "properties": {"big": {"type": "number", "maximum": 12345678901234567890.5}}}}])),
        Some(json!(["not an object"])),
        Some(json!([{"type": "custom", "name": "x"}])),
        Some(json!([{"type": "function"}])),
        Some(json!([{"function": {"name": "x"}}])),
        Some(json!([{"type": 7, "name": "x"}])),
        Some(json!([{"name": ""}])),
        Some(json!([{"name": 3}])),
        Some(json!([{"description": "no name"}])),
        Some(json!([{"name": "x", "description": 5}])),
        Some(json!([{"name": "x", "parameters": "string"}])),
        Some(json!([{"name": "x", "parameters": []}])),
        Some(json!([{"name": "x", "strict": "yes"}])),
        Some(json!([{"name": "bad name!"}])),
    ];
    let top_variants = vec![
        None,
        Some(json!([])),
        Some(json!([{"type": "function", "function": {"name": "top", "parameters": {"type": "object"}}}])),
    ];
    let mut out = Vec::new();
    for top in &top_variants {
        for system_tools in &system_variants {
            let mut system = json!({"role": "system", "content": "s"});
            if let Some(system_tools) = system_tools {
                system["tools"] = system_tools.clone();
            }
            let mut messages = vec![system];
            // A second system message checks message order.
            if system_tools.is_some() {
                messages.push(json!({"role": "system", "content": "s2", "tools": [{"name": "later"}]}));
            }
            messages.push(json!({"role": "user", "content": "hi"}));
            let mut request = json!({"model": "m", "messages": messages});
            if let Some(top) = top {
                request["tools"] = top.clone();
            }
            out.push(request);
        }
    }
    out
}

fn route_json(route: anyhow::Result<ToolProcessingRoute>) -> Value {
    match route {
        Ok(ToolProcessingRoute::Unified(family)) => json!({"unified": family}),
        Ok(ToolProcessingRoute::LegacyJail(parser)) => json!({"legacy_jail": parser}),
        Ok(ToolProcessingRoute::PassThrough) => json!("pass_through"),
        Err(error) => err_json(&error),
    }
}

fn route_pairs() -> Vec<(Option<String>, Option<String>)> {
    let mut pairs = Vec::new();
    for name in names() {
        pairs.push((Some(name.clone()), None));
        pairs.push((None, Some(name.clone())));
        pairs.push((Some(name.clone()), Some(name.clone())));
    }
    pairs.push((None, None));
    for (tool, reasoning) in [
        ("hermes", "qwen3"),
        ("qwen3_coder", "qwen3"),
        ("qwen3_coder", "nemotron3"),
        ("kimi_k2", "basic"),
        ("kimi_k2", "kimi_k25"),
        ("kimi_k2", "kimi"),
        ("harmony", "gpt_oss"),
        ("glm45", "glm45"),
        ("glm47", "glm45"),
        ("glm47", "qwen3"),
        ("minimax_m3", "minimax_m3"),
        ("minimax-m3", "minimax_m3"),
        ("minimax_m2", "minimax_append_think"),
        ("deepseek_v4", "deepseek_v4"),
        ("deepseekv4", "deepseek-v4"),
        ("deepseek_v3", "deepseek_r1"),
        ("deepseek_v41", "deepseek_v4"),
        ("kimi_k3", "kimi-k3"),
        ("gemma-4", "gemma4"),
        ("muse", "muse_glimmer"),
        ("hermes", "deepseek_r1"),
        ("llama3_json", "qwen3"),
        ("junk", "qwen3"),
        ("qwen3_coder", "junk"),
    ] {
        pairs.push((Some(tool.to_string()), Some(reasoning.to_string())));
    }
    pairs
}

fn named(name: &str) -> ChatCompletionToolChoiceOption {
    serde_json::from_value::<ChatCompletionNamedToolChoice>(
        json!({"type": "function", "function": {"name": name}}),
    )
    .map(ChatCompletionToolChoiceOption::Named)
    .unwrap()
}

#[test]
fn record_parser_route_parity() {
    let Ok(path) = std::env::var("PARITY_OUT") else {
        return;
    };
    let version = format!("{:?}", tool_parser_v2::selected_version().unwrap());
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut emit = |value: Value| {
        serde_json::to_writer(&mut out, &value).unwrap();
        out.write_all(b"\n").unwrap();
    };

    let card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    let mut preprocessor = match Arc::try_unwrap(OpenAIPreprocessor::new(card).unwrap()) {
        Ok(value) => value,
        Err(_) => panic!("unexpected shared preprocessor"),
    };
    preprocessor.tokenizer = Arc::new(FakeTokenizer);

    let opt = opt_names();
    let routes = route_requests();
    let route_requests: Vec<NvCreateChatCompletionRequest> = routes
        .iter()
        .map(|r| serde_json::from_value(r.clone()).unwrap())
        .collect();

    // Version-dependent: the selected family, special tokens, and the route.
    for tool in &opt {
        for reasoning in &opt {
            let (t, r) = (tool.as_deref(), reasoning.as_deref());
            let selected = unified_parser::selected_family(t, r);
            let special = match OpenAIPreprocessor::parser_requires_special_tokens(t, r) {
                Ok(value) => json!(value),
                Err(error) => err_json(&error),
            };
            emit(json!({"k": "selected", "v": version, "t": t, "r": r, "family": selected, "special": special}));
        }
    }
    for (tool, reasoning) in route_pairs() {
        preprocessor.tool_call_parser = tool.clone();
        preprocessor.runtime_config.reasoning_parser = reasoning.clone();
        let out: Vec<Value> = route_requests
            .iter()
            .map(|request| {
                route_json(preprocessor.tool_processing_route(request, &GuidedToolConstraint::None))
            })
            .collect();
        emit(json!({"k": "route", "v": version, "t": tool, "r": reasoning, "out": out}));
    }

    if version != "Auto" {
        return;
    }

    // Version-independent below: written once, from the Auto run.
    emit(json!({"k": "route_requests", "requests": routes}));
    emit(json!({"k": "family_names", "names": tool_parser_v2::unified_family_names()}));
    for value in [None, Some("1"), Some("2"), Some("auto"), Some(""), Some("3"), Some("v2"), Some("AUTO")] {
        let parsed = match dynamo_runtime::config::ParserVersion::try_from(value) {
            Ok(version) => json!(format!("{version:?}")),
            Err(error) => err_json(&error),
        };
        emit(json!({"k": "version_parse", "value": value, "out": parsed}));
    }
    for name in names() {
        emit(json!({
            "k": "name",
            "name": name,
            "canonical": unified_parser::canonical_family(&name),
            "tool_only_canonical": tool_parser_v2::canonical_family(&name),
            "tool_only_supported": tool_parser_v2::supports_family(&name),
        }));
    }
    for reasoning in &opt {
        let r = reasoning.as_deref();
        let out: Vec<bool> = PROMPTS
            .iter()
            .map(|p| OpenAIPreprocessor::prompt_injected_reasoning_start(r, *p))
            .collect();
        emit(json!({"k": "prompt_start", "r": r, "out": out}));
    }
    for tool in &opt {
        for reasoning in &opt {
            let (t, r) = (tool.as_deref(), reasoning.as_deref());
            let for_parsers: Vec<bool> = PROMPTS
                .iter()
                .map(|p| OpenAIPreprocessor::prompt_injected_reasoning_start_for_parsers(t, r, *p))
                .collect();
            let prefill: Vec<Value> = PROMPTS
                .iter()
                .map(|p| prefill_json(OpenAIPreprocessor::prompt_reasoning_prefill_for_parsers(t, r, *p)))
                .collect();
            let modes: Vec<Value> = [
                tool_parser_v2::ParserVersion::Auto,
                tool_parser_v2::ParserVersion::V1,
                tool_parser_v2::ParserVersion::V2,
            ]
            .into_iter()
            .map(|version| {
                let mut row = Vec::new();
                for requires_v1_jail in [false, true] {
                    row.push(
                        match tool_parser_v2::validate_tool_request_mode(version, t, r, requires_v1_jail) {
                            Ok(()) => json!(null),
                            Err(error) => err_json(&error),
                        },
                    );
                }
                json!(row)
            })
            .collect();
            emit(json!({
                "k": "pair",
                "t": t,
                "r": r,
                "configured": unified_parser::configured_family(t, r),
                "is_v2": unified_parser::is_v2_configured_family(t, r),
                "start_for_parsers": for_parsers,
                "prefill": prefill,
                "tool_request_mode": modes,
            }));
        }
    }
    emit(json!({"k": "prompts", "prompts": PROMPTS}));
    let choices = vec![
        ToolChoice::None,
        ToolChoice::Auto,
        ToolChoice::Required,
        ToolChoice::Named("lookup".to_string()),
    ];
    let options = vec![
        None,
        Some(ChatCompletionToolChoiceOption::None),
        Some(ChatCompletionToolChoiceOption::Auto),
        Some(ChatCompletionToolChoiceOption::Required),
        Some(named("lookup")),
    ];
    for tool in &opt {
        let t = tool.as_deref();
        let native: Vec<bool> = choices
            .iter()
            .map(|choice| requires_native_tool_call_format(t, choice))
            .collect();
        let jail: Vec<Value> = options
            .iter()
            .flat_map(|choice| {
                [false, true].map(|has_tools| {
                    match OpenAIPreprocessor::should_apply_tool_jail(tool.as_ref(), choice.as_ref(), has_tools) {
                        Ok(value) => json!(value),
                        Err(error) => err_json(&error),
                    }
                })
            })
            .collect();
        // Hidden-stop rule: every end token of the parser as a hidden EOS, plus id 7.
        preprocessor.tool_call_parser = tool.clone();
        let mut hidden_inputs: Vec<Vec<u32>> = vec![vec![], vec![7]];
        if let Some(config) = t.and_then(|t| dynamo_parsers::tool_calling::parsers::get_tool_parser_map().get(t)) {
            let mut ids: Vec<u32> = config
                .parser_config
                .tool_call_end_tokens()
                .iter()
                .flat_map(|token| fake_encode(token))
                .collect();
            ids.push(7);
            hidden_inputs.push(ids);
        }
        let hidden: Vec<Value> = hidden_inputs
            .iter()
            .map(|input| {
                let out: Vec<Value> = route_requests
                    .iter()
                    .map(|request| {
                        let mut hidden = input.clone();
                        match preprocessor.remove_tool_parser_end_tokens_from_hidden_stops(request, &mut hidden) {
                            Ok(visible) => json!([visible, hidden]),
                            Err(error) => err_json(&error),
                        }
                    })
                    .collect();
                json!({"input": input, "out": out})
            })
            .collect();
        emit(json!({"k": "tool_parser", "t": t, "native": native, "jail": jail, "hidden": hidden}));
    }
    for request_json in route_requests_and_tools() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(request_json.clone()).unwrap();
        let effective = match tool_choice::effective_tools(&request.inner) {
            Ok(tools) => json!({"ok": serde_json::to_value(tools.as_ref()).unwrap()}),
            Err(error) => json!({"err": error.message()}),
        };
        let definitions = match tool_choice::effective_tool_definitions(&request.inner) {
            Ok(tools) => json!({"ok": tools.iter().map(|t| json!({"name": t.name, "parameters": t.parameters, "strict": t.strict})).collect::<Vec<_>>()}),
            Err(error) => json!({"err": error.message()}),
        };
        emit(json!({
            "k": "tools",
            "request": request_json,
            "parsing_enabled": OpenAIPreprocessor::tool_call_parsing_enabled(&request),
            "request_has_effective_tools": OpenAIPreprocessor::request_has_effective_tools(&request),
            "effective": effective,
            "definitions": definitions,
        }));
    }
    for skip in [None, Some(false), Some(true)] {
        for requires in [false, true] {
            emit(json!({
                "k": "stripped",
                "skip": skip,
                "requires": requires,
                "out": OpenAIPreprocessor::special_tokens_will_be_stripped(skip, requires),
            }));
        }
    }
}

fn route_requests_and_tools() -> Vec<Value> {
    let mut all = route_requests();
    all.extend(tools_requests());
    all
}
