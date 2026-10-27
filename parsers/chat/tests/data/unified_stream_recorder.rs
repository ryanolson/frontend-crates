// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The recorder that wrote `unified_stream_parity.jsonl.gz`. This crate does not build it.
//!
//! It ran as a test inside dynamo-llm at ai-dynamo/dynamo `cffc4fa`, as
//! `lib/llm/src/protocols/openai/chat_completions/unified_stream_recorder.rs`, with
//! `#[cfg(test)] #[path = "unified_stream_recorder.rs"] mod unified_stream_recorder;`
//! at the end of `unified_parser.rs`. Run it once per `DYN_PARSER_VERSION` value,
//! because dynamo-llm reads the version once per process:
//!
//! ```text
//! PARITY_OUT=auto.jsonl cargo test -p dynamo-llm --lib unified_stream_recorder -- --ignored
//! DYN_PARSER_VERSION=2 PARITY_OUT=2.jsonl cargo test -p dynamo-llm --lib unified_stream_recorder -- --ignored
//! python3 pack_unified_stream_parity.py unified_stream_parity.jsonl.gz auto.jsonl 2.jsonl
//! ```

use super::*;
use super::super::ToolCallCompletion;
use crate::protocols::common::metrics::LLMMetricAnnotation;
use dynamo_protocols::types::{CreateChatCompletionStreamResponse, Role};
use dynamo_runtime::error::DynamoError;
use futures::stream;
use serde_json::{Map, Value, json};
use std::io::Write;
use std::sync::Arc;

type Item = Annotated<NvCreateChatCompletionStreamResponse>;

const FAMILIES: &[&str] = &[
    DEEPSEEK_V4_UNIFIED_FAMILY,
    DEEPSEEK_V41_UNIFIED_FAMILY,
    GLM47_UNIFIED_FAMILY,
    "gpt_oss",
    QWEN3_UNIFIED_FAMILY,
    KIMI_K2_UNIFIED_FAMILY,
    KIMI_K3_UNIFIED_FAMILY,
    GEMMA4_UNIFIED_FAMILY,
    MUSE_UNIFIED_FAMILY,
];

#[derive(Clone)]
enum Arg {
    Str(String),
    Num(&'static str),
}

fn reasoning_pair(family: &str) -> (&'static str, &'static str) {
    match family {
        "gpt_oss" => (
            "<|channel|>analysis<|message|>",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
        ),
        GEMMA4_UNIFIED_FAMILY => ("<|channel>thought\n", "<channel|>"),
        MUSE_UNIFIED_FAMILY => ("<|start|>assistant to=self<|message|>", "<|eom|>"),
        KIMI_K3_UNIFIED_FAMILY => ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
        _ => ("<think>", "</think>"),
    }
}

fn json_args(fields: &[(&str, Arg)]) -> String {
    let body = fields
        .iter()
        .map(|(key, value)| {
            let value = match value {
                Arg::Str(text) => serde_json::to_string(text).unwrap(),
                Arg::Num(raw) => raw.to_string(),
            };
            format!("{}:{value}", serde_json::to_string(key).unwrap())
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{body}}}")
}

/// Native call text for one family, with raw numbers kept as written.
fn native_call(family: &str, name: &str, fields: &[(&str, Arg)]) -> String {
    let parameters: String = fields
        .iter()
        .map(|(key, value)| {
            let (raw, is_string) = match value {
                Arg::Str(text) => (text.clone(), true),
                Arg::Num(raw) => (raw.to_string(), false),
            };
            match family {
                QWEN3_UNIFIED_FAMILY => format!("<parameter={key}>{raw}</parameter>"),
                KIMI_K3_UNIFIED_FAMILY => format!(
                    "<|open|>argument key=\"{key}\" type=\"{}\"<|sep|>{raw}<|close|>argument<|sep|>",
                    if is_string { "string" } else { "number" }
                ),
                DEEPSEEK_V4_UNIFIED_FAMILY | DEEPSEEK_V41_UNIFIED_FAMILY => {
                    let gap = if family == DEEPSEEK_V41_UNIFIED_FAMILY { " " } else { "" };
                    format!(
                        "<｜DSML｜{gap}parameter name=\"{key}\" string=\"{is_string}\">{raw}</｜DSML｜{gap}parameter>"
                    )
                }
                GLM47_UNIFIED_FAMILY => format!("<arg_key>{key}</arg_key><arg_value>{raw}</arg_value>"),
                MUSE_UNIFIED_FAMILY => format!("<atem:parameter name=\"{key}\">{raw}</atem:parameter>"),
                _ => String::new(),
            }
        })
        .collect();
    match family {
        QWEN3_UNIFIED_FAMILY => {
            format!("<tool_call><function={name}>{parameters}</function></tool_call>")
        }
        KIMI_K2_UNIFIED_FAMILY => format!(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
            json_args(fields)
        ),
        KIMI_K3_UNIFIED_FAMILY => format!(
            "<|open|>tools<|sep|><|open|>call tool=\"{name}\" index=\"1\"<|sep|>{parameters}<|close|>call<|sep|><|close|>tools<|sep|>"
        ),
        DEEPSEEK_V4_UNIFIED_FAMILY => format!(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"{name}\">{parameters}</｜DSML｜invoke></｜DSML｜tool_calls>"
        ),
        DEEPSEEK_V41_UNIFIED_FAMILY => format!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"{name}\">{parameters}</｜DSML｜ invoke></｜DSML｜ calls>"
        ),
        GEMMA4_UNIFIED_FAMILY => {
            let parameters = fields
                .iter()
                .map(|(key, value)| match value {
                    Arg::Str(raw) => format!("{key}:<|\"|>{raw}<|\"|>"),
                    Arg::Num(raw) => format!("{key}:{raw}"),
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("<|tool_call>call:{name}{{{parameters}}}<tool_call|>")
        }
        GLM47_UNIFIED_FAMILY => format!("<tool_call>{name}{parameters}</tool_call>"),
        MUSE_UNIFIED_FAMILY => format!(
            "<|start|>assistant to={name}<|message|><atem:invoke name=\"{name}\">{parameters}</atem:invoke><|eom|>"
        ),
        "gpt_oss" => format!(
            "<|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{}<|call|>",
            json_args(fields)
        ),
        _ => unreachable!(),
    }
}

fn first_marker(family: &str) -> &'static str {
    match family {
        QWEN3_UNIFIED_FAMILY | GLM47_UNIFIED_FAMILY => "<tool_call>",
        KIMI_K2_UNIFIED_FAMILY => "<|tool_calls_section_begin|>",
        KIMI_K3_UNIFIED_FAMILY => "<|open|>tools<|sep|>",
        DEEPSEEK_V4_UNIFIED_FAMILY => "<｜DSML｜tool_calls>",
        DEEPSEEK_V41_UNIFIED_FAMILY => "<｜DSML｜ calls>",
        GEMMA4_UNIFIED_FAMILY => "<|tool_call>",
        MUSE_UNIFIED_FAMILY => "<|start|>assistant to=",
        _ => "<|start|>assistant to=functions.",
    }
}

fn tools_defs() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "write_file".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "count": {"type": "number"}
                },
                "required": ["content"]
            })),
            strict: None,
        },
        ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
            strict: None,
        },
    ]
}

fn write_fields(content: &str) -> Vec<(&'static str, Arg)> {
    vec![("count", Arg::Num("2")), ("content", Arg::Str(content.to_string()))]
}

/// Native scenarios for one family: (name, text, prefill-opened variant).
fn native_scenarios(family: &str) -> Vec<(&'static str, String)> {
    let (open, close) = reasoning_pair(family);
    let call = native_call(family, "write_file", &write_fields("é hi"));
    let call2 = native_call(family, "get_weather", &[("city", Arg::Str("Paris".into()))]);
    let numeric = native_call(
        family,
        "write_file",
        &[
            ("count", Arg::Num("1.0")),
            ("path", Arg::Str("n.txt".into())),
            ("content", Arg::Str("12345678901234567890".into())),
        ],
    );
    let numeric_big = native_call(
        family,
        "write_file",
        &[("count", Arg::Num("12345678901234567890")), ("content", Arg::Str("x".into()))],
    );
    let numeric_exp = native_call(
        family,
        "write_file",
        &[("count", Arg::Num("1e3")), ("content", Arg::Str("-0.0".into()))],
    );
    let cut = call.char_indices().nth(call.chars().count() * 3 / 5).unwrap().0;
    let final_open = if family == "gpt_oss" { "<|channel|>final<|message|>" } else { "" };
    vec![
        ("text", format!("{final_open}Hello, world. The answer is 42.")),
        ("reason_text", format!("{open}Think it over.{close}Answer here.")),
        ("reason_call", format!("{open}Need a tool.{close}{call}")),
        ("call_text", format!("{call}{final_open}Done.")),
        ("two_calls", format!("{call}{call2}")),
        (
            "quoted",
            format!("{final_open}He typed \"{}\" and '{close}' literally.", first_marker(family)),
        ),
        ("truncated", call[..cut].to_string()),
        ("numeric", numeric),
        ("numeric_big", numeric_big),
        ("numeric_exp", numeric_exp),
        ("reason_unclosed", format!("{open}never closes")),
        ("empty", String::new()),
        ("prefilled_reason", format!("inner thought{close}{call}")),
    ]
}

fn guided_scenarios(family: &str, named: bool) -> Vec<(&'static str, String)> {
    let (open, close) = reasoning_pair(family);
    let args = json_args(&write_fields("é hi"));
    let numeric = json_args(&[("count", Arg::Num("1.0")), ("content", Arg::Str("x".into()))]);
    let payload = |args: &str| {
        if named {
            args.to_string()
        } else {
            format!("[{{\"name\":\"write_file\",\"parameters\":{args}}}]")
        }
    };
    vec![
        ("guided", payload(&args)),
        ("guided_ws", format!("  \n{}", payload(&args))),
        ("guided_reason", format!("{open}plan{close}{}", payload(&args))),
        ("guided_prefilled", format!("plan{close}{}", payload(&args))),
        ("guided_text", "Sorry, I cannot call a tool.".to_string()),
        ("guided_numeric", payload(&numeric)),
        ("guided_truncated", payload(&args)[..payload(&args).len() - 3].to_string()),
    ]
}

fn structured_scenarios(family: &str) -> Vec<(&'static str, String)> {
    let (open, close) = reasoning_pair(family);
    vec![
        ("json", "{\"answer\": 1.50}".to_string()),
        ("json_ws", " \n{\"answer\": 1}".to_string()),
        ("json_reason", format!("{open}plan{close}{{\"answer\": 1}}")),
        ("json_prefilled", format!("plan{close}{{\"answer\": 1}}")),
        ("json_array", "[1, 2, 3]".to_string()),
        ("json_scalar", "\"text\"".to_string()),
    ]
}

fn split(text: &str, size: usize) -> Vec<String> {
    if size == 0 || text.is_empty() {
        return vec![text.to_string()];
    }
    let chars: Vec<char> = text.chars().collect();
    chars.chunks(size).map(|c| c.iter().collect()).collect()
}

#[allow(deprecated)]
fn choice(index: u32, text: Option<&str>, role: bool) -> ChatChoiceStream {
    ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role: role.then_some(Role::Assistant),
            content: text.map(|t| ChatCompletionMessageContent::Text(t.to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    }
}

fn data_item(choices: Vec<ChatChoiceStream>) -> Item {
    Annotated::from_data(NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "chatcmpl-parity".to_string(),
            choices,
            created: 1,
            model: "parity-model".to_string(),
            system_fingerprint: None,
            service_tier: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        },
        nvext: None,
        llm_metrics: None,
        prompt_logprobs: None,
        tool_call_completion: Vec::new(),
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Envelope {
    Plain,
    NoFinish,
    UsageTrailer,
    Metadata,
    TwoChoices,
    ErrorMid,
    ReasoningDetour,
    ParsedCallDetour,
    ParsedPartialCallDetour,
    Evidence,
    FinishLength,
    FinishToolCalls,
}

const ENVELOPES: &[(Envelope, &str)] = &[
    (Envelope::NoFinish, "no_finish"),
    (Envelope::UsageTrailer, "usage_trailer"),
    (Envelope::Metadata, "metadata"),
    (Envelope::TwoChoices, "two_choices"),
    (Envelope::ErrorMid, "error_mid"),
    (Envelope::ReasoningDetour, "reasoning_detour"),
    (Envelope::ParsedCallDetour, "parsed_call_detour"),
    (Envelope::ParsedPartialCallDetour, "parsed_partial_call_detour"),
    (Envelope::Evidence, "evidence"),
    (Envelope::FinishLength, "finish_length"),
    (Envelope::FinishToolCalls, "finish_tool_calls"),
];

fn build_input(text: &str, size: usize, envelope: Envelope, second: &str) -> Vec<Item> {
    let pieces = split(text, size);
    let mut items: Vec<Item> = Vec::new();
    let finish = match envelope {
        Envelope::NoFinish => None,
        Envelope::FinishLength => Some(FinishReason::Length),
        Envelope::FinishToolCalls => Some(FinishReason::ToolCalls),
        _ => Some(FinishReason::Stop),
    };
    let second_pieces = split(second, size);
    let rounds = if envelope == Envelope::TwoChoices {
        pieces.len().max(second_pieces.len())
    } else {
        pieces.len()
    };
    for i in 0..rounds {
        let mut choices = Vec::new();
        if let Some(piece) = pieces.get(i) {
            let mut c = choice(0, Some(piece), i == 0);
            if i + 1 == pieces.len() {
                c.finish_reason = finish;
            }
            choices.push(c);
        }
        if envelope == Envelope::TwoChoices {
            if let Some(piece) = second_pieces.get(i) {
                let mut c = choice(1, Some(piece), i == 0);
                if i + 1 == second_pieces.len() {
                    c.finish_reason = finish;
                }
                choices.push(c);
            }
        }
        let mut item = data_item(choices);
        if envelope == Envelope::Metadata {
            let data = item.data.as_mut().unwrap();
            data.llm_metrics = Some(LLMMetricAnnotation {
                chunk_tokens: i + 1,
                ..Default::default()
            });
            data.nvext = Some(json!({"seq": i}));
            if i == 0 {
                data.prompt_logprobs = Some(Arc::new(vec![None, None]));
            }
            if i % 2 == 0 {
                item.id = Some(format!("evt-{i}"));
                item.comment = Some(vec![format!("c-{i}")]);
            }
            if i % 3 == 1 {
                item.event = Some(format!("note-{i}"));
            }
        }
        if envelope == Envelope::Evidence && i + 1 == rounds {
            item.data.as_mut().unwrap().tool_call_completion = vec![
                ToolCallCompletion { choice_index: 0, tool_index: 0, complete: false },
                ToolCallCompletion { choice_index: 0, tool_index: 1, complete: true },
            ];
        }
        items.push(item);
        if i == rounds / 2 {
            match envelope {
                Envelope::ErrorMid => {
                    let mut error: Item = Annotated {
                        data: None,
                        id: None,
                        event: Some("error".to_string()),
                        comment: Some(vec!["boom".to_string()]),
                        error: Some(DynamoError::msg("boom")),
                    };
                    error.id = Some("err-1".to_string());
                    items.push(error);
                }
                Envelope::ReasoningDetour => {
                    let mut c = choice(0, None, false);
                    c.delta.reasoning_content = Some("parsed thought".to_string());
                    items.push(data_item(vec![c]));
                }
                Envelope::ParsedCallDetour | Envelope::ParsedPartialCallDetour => {
                    let arguments = if envelope == Envelope::ParsedCallDetour {
                        "{\"city\":\"Oslo\"}"
                    } else {
                        "{\"city\":"
                    };
                    let mut c = choice(0, None, false);
                    c.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
                        index: 0,
                        id: Some("call_parsed".to_string()),
                        r#type: Some(FunctionType::Function),
                        function: Some(FunctionCallStream {
                            name: Some("get_weather".to_string()),
                            arguments: Some(arguments.to_string()),
                        }),
                    }]);
                    items.push(data_item(vec![c]));
                }
                _ => {}
            }
        }
    }
    if envelope == Envelope::UsageTrailer || envelope == Envelope::Metadata {
        let mut trailer = data_item(Vec::new());
        let data = trailer.data.as_mut().unwrap();
        data.inner.usage = Some(dynamo_protocols::types::CompletionUsage {
            prompt_tokens: 7,
            completion_tokens: 9,
            total_tokens: 16,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        });
        if envelope == Envelope::Metadata {
            data.nvext = Some(json!({"trailer": true}));
            data.llm_metrics = Some(LLMMetricAnnotation { chunk_tokens: 99, ..Default::default() });
        }
        items.push(trailer);
    }
    items
}

fn strip_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, strip_nulls(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(strip_nulls).collect()),
        other => other,
    }
}

fn compact(item: &Item) -> Value {
    let mut out = Map::new();
    if let Some(id) = &item.id {
        out.insert("id".into(), json!(id));
    }
    if let Some(event) = &item.event {
        out.insert("event".into(), json!(event));
    }
    if let Some(comment) = &item.comment {
        out.insert("comment".into(), json!(comment));
    }
    if let Some(error) = &item.error {
        out.insert("error".into(), json!(error.message()));
    }
    if let Some(data) = &item.data {
        let mut inner = strip_nulls(serde_json::to_value(&data.inner).unwrap());
        let map = inner.as_object_mut().unwrap();
        for (key, default) in [
            ("id", json!("chatcmpl-parity")),
            ("object", json!("chat.completion.chunk")),
            ("created", json!(1)),
            ("model", json!("parity-model")),
        ] {
            if map.get(key) == Some(&default) {
                map.remove(key);
            }
        }
        if let Some(Value::Array(choices)) = map.get_mut("choices") {
            for choice in choices {
                if let Some(Value::Array(calls)) = choice.pointer_mut("/delta/tool_calls") {
                    for call in calls {
                        if call
                            .get("id")
                            .and_then(Value::as_str)
                            .is_some_and(|id| id.starts_with("call-"))
                        {
                            call["id"] = json!("call-*");
                        }
                    }
                }
            }
        }
        if let Some(nvext) = &data.nvext {
            map.insert("nvext".into(), nvext.clone());
        }
        if let Some(metrics) = &data.llm_metrics {
            map.insert("llm_metrics".into(), json!(metrics.chunk_tokens));
        }
        if let Some(logprobs) = &data.prompt_logprobs {
            map.insert("prompt_logprobs".into(), json!(logprobs.len()));
        }
        if !data.tool_call_completion.is_empty() {
            map.insert(
                "tool_call_completion".into(),
                serde_json::to_value(&data.tool_call_completion).unwrap(),
            );
        }
        out.insert("data".into(), inner);
    }
    Value::Object(out)
}

struct Case {
    name: String,
    family: &'static str,
    tools: bool,
    constraint: GuidedToolConstraint,
    prefill: UnifiedParserStartingState,
    prefix: Option<&'static str>,
    policy: UnifiedRequestPolicy,
    guided_streaming: bool,
    input: Vec<Item>,
}

fn constraint_json(constraint: &GuidedToolConstraint) -> Value {
    serde_json::to_value(constraint).unwrap()
}

fn prefill_name(state: UnifiedParserStartingState) -> &'static str {
    match state {
        UnifiedParserStartingState::None => "none",
        UnifiedParserStartingState::Reasoning => "reasoning",
        UnifiedParserStartingState::Response => "response",
        #[allow(unreachable_patterns)]
        _ => unreachable!(),
    }
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let default_case = |name: String, family: &'static str, input: Vec<Item>| Case {
        name,
        family,
        tools: true,
        constraint: GuidedToolConstraint::None,
        prefill: UnifiedParserStartingState::None,
        prefix: None,
        policy: UnifiedRequestPolicy::default(),
        guided_streaming: true,
        input,
    };
    for &family in FAMILIES {
        let scenarios = native_scenarios(family);
        // A: every native scenario at three split sizes.
        for (scenario, text) in &scenarios {
            for size in [0usize, 4] {
                let (prefill, label) = if *scenario == "prefilled_reason" {
                    (UnifiedParserStartingState::Reasoning, "reasoning")
                } else {
                    (UnifiedParserStartingState::None, "none")
                };
                let mut case = default_case(
                    format!("core/{family}/{scenario}/split{size}/{label}"),
                    family,
                    build_input(text, size, Envelope::Plain, ""),
                );
                case.prefill = prefill;
                cases.push(case);
            }
            if !matches!(*scenario, "reason_call" | "text" | "quoted") {
                continue;
            }
            // No tools at all.
            let mut case = default_case(
                format!("core/{family}/{scenario}/no_tools"),
                family,
                build_input(text, 5, Envelope::Plain, ""),
            );
            case.tools = false;
            cases.push(case);
        }
        // B: envelope shapes.
        let second = scenarios.iter().find(|(n, _)| *n == "reason_text").unwrap().1.clone();
        for scenario in ["reason_call", "truncated", "call_text"] {
            let text = &scenarios.iter().find(|(n, _)| *n == scenario).unwrap().1;
            for (envelope, label) in ENVELOPES {
                cases.push(default_case(
                    format!("envelope/{family}/{scenario}/{label}"),
                    family,
                    build_input(text, 5, *envelope, &second),
                ));
            }
        }
        // C: request options.
        let prefills = [
            UnifiedParserStartingState::None,
            UnifiedParserStartingState::Reasoning,
            UnifiedParserStartingState::Response,
        ];
        let policies = [(false, false), (true, false), (false, true), (true, true)];
        for constraint in [GuidedToolConstraint::None, GuidedToolConstraint::StructuralTag] {
            for scenario in ["reason_call", "prefilled_reason"] {
                if constraint == GuidedToolConstraint::StructuralTag && scenario != "reason_call" {
                    continue;
                }
                let text = &scenarios.iter().find(|(n, _)| *n == scenario).unwrap().1;
                for prefill in prefills {
                    for (reasoning_disabled, structured_response) in policies {
                        let mut case = default_case(
                            format!(
                                "options/{family}/{}/{scenario}/{}/rd{}/sr{}",
                                constraint_json(&constraint),
                                prefill_name(prefill),
                                reasoning_disabled as u8,
                                structured_response as u8
                            ),
                            family,
                            build_input(text, 7, Envelope::Plain, ""),
                        );
                        case.constraint = constraint.clone();
                        case.prefill = prefill;
                        case.policy = UnifiedRequestPolicy { reasoning_disabled, structured_response };
                        cases.push(case);
                    }
                }
            }
        }
        for named in [true, false] {
            let constraint = if named {
                GuidedToolConstraint::GuidedJsonNamed { tool_name: "write_file".to_string() }
            } else {
                GuidedToolConstraint::GuidedJsonRequired
            };
            for (scenario, text) in guided_scenarios(family, named) {
                for prefill in [UnifiedParserStartingState::None, UnifiedParserStartingState::Reasoning] {
                    for guided_streaming in [true, false] {
                        if !guided_streaming
                            && !matches!(scenario, "guided" | "guided_truncated" | "guided_reason")
                        {
                            continue;
                        }
                        for (reasoning_disabled, structured_response) in [(false, false), (true, false)] {
                            if reasoning_disabled && !matches!(scenario, "guided_reason" | "guided_prefilled") {
                                continue;
                            }
                            let mut case = default_case(
                                format!(
                                    "guided/{family}/{}/{scenario}/{}/gs{}/rd{}/sr{}",
                                    if named { "named" } else { "required" },
                                    prefill_name(prefill),
                                    guided_streaming as u8,
                                    reasoning_disabled as u8,
                                    structured_response as u8
                                ),
                                family,
                                build_input(&text, 6, Envelope::Plain, ""),
                            );
                            case.constraint = constraint.clone();
                            case.prefill = prefill;
                            case.guided_streaming = guided_streaming;
                            case.policy = UnifiedRequestPolicy { reasoning_disabled, structured_response };
                            cases.push(case);
                        }
                    }
                }
            }
        }
        for (scenario, text) in structured_scenarios(family) {
            for prefill in prefills {
                for reasoning_disabled in [false, true] {
                    let mut case = default_case(
                        format!(
                            "structured/{family}/{scenario}/{}/rd{}",
                            prefill_name(prefill),
                            reasoning_disabled as u8
                        ),
                        family,
                        build_input(&text, 5, Envelope::Plain, ""),
                    );
                    case.prefill = prefill;
                    case.policy = UnifiedRequestPolicy { reasoning_disabled, structured_response: true };
                    cases.push(case);
                }
            }
        }
    }
    // Gemma prompt prefixes: the prompt ends partway into the thought header.
    const HEADER: &str = "<|channel>thought\n";
    for end in "<|channel>".len()..HEADER.len() {
        let prefix: &'static str = &HEADER[..end];
        let rest = &HEADER[end..];
        for scenario in ["think", "answer"] {
            let text = if scenario == "think" {
                format!("{rest}deep thought<channel|>Answer.")
            } else {
                "Answer only.".to_string()
            };
            let mut case = default_case(
                format!("gemma_prefix/{end}/{scenario}"),
                GEMMA4_UNIFIED_FAMILY,
                build_input(&text, 2, Envelope::Plain, ""),
            );
            case.prefill = UnifiedParserStartingState::Reasoning;
            case.prefix = Some(prefix);
            cases.push(case);
        }
    }
    cases
}

#[tokio::test]
#[ignore = "scratch recorder"]
async fn unified_stream_recorder() {
    let Ok(path) = std::env::var("PARITY_OUT") else {
        return;
    };
    let selected = parsers_v2_selected();
    let mut file = std::fs::File::create(path).unwrap();
    let mut count = 0usize;
    for case in cases() {
        let input_json: Vec<Value> = case.input.iter().map(compact).collect();
        let output: Vec<Item> = apply_stream_with_policy(
            stream::iter(case.input),
            case.tools.then(tools_defs),
            case.constraint.clone(),
            StreamPrefill { starting_state: case.prefill, prefix: case.prefix },
            case.family,
            case.guided_streaming,
            case.policy,
        )
        .collect()
        .await;
        let record = json!({
            "name": case.name,
            "family": case.family,
            "tools": case.tools,
            "constraint": constraint_json(&case.constraint),
            "prefill": prefill_name(case.prefill),
            "prefix": case.prefix,
            "reasoning_disabled": case.policy.reasoning_disabled,
            "structured_response": case.policy.structured_response,
            "guided_streaming": case.guided_streaming,
            "parsers_v2_selected": selected,
            "input": input_json,
            "output": output.iter().map(compact).collect::<Vec<_>>(),
        });
        writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        count += 1;
    }
    eprintln!("recorded {count} cases, parsers_v2_selected={selected}");
}
