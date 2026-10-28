// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parity with dynamo-llm's chat aggregator.
//!
//! `tests/data/aggregator_parity.json.gz` holds 74,124 `stream: false` replies
//! and the response that dynamo-llm's own
//! `NvCreateChatCompletionResponse::from_annotated_stream` gave for each, before
//! the aggregator moved here. It was recorded at ai-dynamo/dynamo
//! `cffc4fa75c15017bcd32e62f7d0fe4bad468c35d` (dynamo-parsers 10.0.7,
//! dynamo-parsers-v2 0.7.35). The recorder is `tests/data/aggregator_recorder.rs`.
//!
//! A case is a parser pair, the request options the aggregator reads, and the
//! chunks it folds. dynamo ran the recorder with `DYN_PARSER_VERSION` unset, `1`,
//! and `2`. This test runs every case at all three versions and expects that
//! version's output.
//!
//! The expected differences depend on the build ([`Build`]), and the test pins
//! each count:
//!
//! - On the `dynamo-parsers-v2-v0.7.35` tag, with dynamo-parsers 10.0.7 (the
//!   versions dynamo pins), every run matches.
//! - With dynamo-parsers-v2 0.8 and serde_json `arbitrary_precision` off,
//!   144 runs differ: GLM tool arguments keep `1000` where dynamo sent
//!   `1000.0` ([`MainDrift::NumberText`]). 0.8 passes arguments through as the
//!   model wrote them (frontend-crates #339). This is a known difference from
//!   stock dynamo, which goes away when dynamo moves to 0.8.
//! - While frontend-crates `main` turns `arbitrary_precision` on for the whole
//!   build, 1,956 runs differ: 1,336 [`MainDrift::NumberText`] runs (`1e+3`
//!   where dynamo sent `1000.0`), and 620 [`MainDrift::HugeNumberKept`] runs,
//!   where a call holding `1e400`, which dynamo drops or never finds, reaches
//!   the client. Only the call, its finish reason, and the content a
//!   guided-JSON fallback consumed may change. These come from the feature, not
//!   from the move, and go away when the feature is opt-in; remove this case
//!   then.
//!
//! The chunks hold only the OpenAI part and the tool-call completion evidence.
//! dynamo folds `nvext` and prompt logprobs around the aggregator, so the
//! recorder sent none. [`run`] stands in for that loop, as dynamo-llm's
//! `DeltaAggregator::apply` will after the move: an error item ends the reply
//! with dynamo's error text.

use std::io::Read;

use dynamo_chat_parsing::aggregator::{AggregatorOptions, DeltaAggregator};
use dynamo_chat_parsing::guided::GuidedToolConstraint;
use dynamo_chat_parsing::stream_item::ToolCallCompletion;
use dynamo_chat_parsing::version::ParserVersion;
use dynamo_parsers::tool_calling::ToolDefinition;
use serde_json::{Value, json};

struct Fixture {
    inputs: Vec<Value>,
    options: Vec<Value>,
    outputs: Vec<Value>,
    cases: Vec<Value>,
}

fn fixture() -> Fixture {
    let bytes = include_bytes!("data/aggregator_parity.json.gz");
    let mut text = String::new();
    flate2::read::GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .expect("decompress fixture");
    let mut blob: Value = serde_json::from_str(&text).expect("fixture");
    let mut take = |key: &str| match blob[key].take() {
        Value::Array(items) => items,
        other => panic!("{key}: {other}"),
    };
    Fixture {
        inputs: take("inputs"),
        options: take("options"),
        outputs: take("outputs"),
        cases: take("cases"),
    }
}

/// The recorder's tools.
fn tools() -> Vec<ToolDefinition> {
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

/// The recorder's request options. `tool_choice` is in the file too, but the
/// aggregator never read it. `tool_parsing_disabled` means dynamo's handler
/// called `with_tool_call_parsing_enabled(false)` on the options.
fn options(value: &Value, parser_version: ParserVersion) -> AggregatorOptions {
    let text = |key: &str| value[key].as_str().map(str::to_string);
    let flag = |key: &str| value[key].as_bool().expect(key);
    let options = AggregatorOptions {
        tool_call_parser: text("tool_call_parser"),
        reasoning_parser: text("reasoning_parser"),
        reasoning_disabled: flag("reasoning_disabled"),
        structured_response: flag("structured_response"),
        suppress_tool_calls: flag("suppress_tool_calls"),
        guided_tool_constraint: serde_json::from_value::<GuidedToolConstraint>(
            value["guided_tool_constraint"].clone(),
        )
        .expect("constraint"),
        parallel_tool_calls: value["parallel_tool_calls"].as_bool(),
        move_reasoning_to_content_when_empty: flag("move_reasoning_to_content_when_empty"),
        tools: if flag("tools") { tools() } else { Vec::new() },
        parser_version,
    };
    options.with_tool_call_parsing_enabled(!flag("tool_parsing_disabled"))
}

/// A hyphenated UUID: 36 bytes, hex digits, dashes at 8, 13, 18, and 23.
fn is_uuid(text: &str) -> bool {
    text.len() == 36
        && text.bytes().enumerate().all(|(at, byte)| match at {
            8 | 13 | 18 | 23 => byte == b'-',
            _ => byte.is_ascii_hexdigit(),
        })
}

/// The recorder's normalization of generated call IDs.
fn normalize(value: &mut Value) {
    match value {
        Value::String(text) => {
            if text.strip_prefix("call-").is_some_and(is_uuid) {
                *text = "call-*".to_string();
            } else if text
                .strip_prefix("call_")
                .is_some_and(|rest| rest.len() == 24 && rest.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                *text = "call_*".to_string();
            } else if is_uuid(text) {
                *text = "uuid-*".to_string();
            }
        }
        Value::Array(items) => items.iter_mut().for_each(normalize),
        Value::Object(map) => map.values_mut().for_each(normalize),
        _ => {}
    }
}

/// The pack script's compaction: the fixed envelope fields are left out.
fn compact(mut response: Value) -> Value {
    let map = response.as_object_mut().expect("response object");
    for (key, fixed) in [
        ("id", json!("chatcmpl-parity")),
        ("object", json!("chat.completion")),
        ("created", json!(1700000000u32)),
        ("model", json!("parity-model")),
    ] {
        if map.get(key) == Some(&fixed) {
            map.remove(key);
        }
    }
    response
}

/// How a run on frontend-crates `main` may differ from the recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MainDrift {
    NumberText,
    HugeNumberKept,
}

/// Whether serde_json keeps numbers as written (`arbitrary_precision`), as on
/// frontend-crates `main`.
fn numbers_kept_as_written() -> bool {
    serde_json::from_str::<Value>("1e400").is_ok()
}

/// The build the replay runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Build {
    /// dynamo-parsers-v2 0.7.35, as dynamo pins it.
    Pinned,
    /// dynamo-parsers-v2 0.8 with serde_json `arbitrary_precision` off.
    PassThrough,
    /// dynamo-parsers-v2 0.8 with `arbitrary_precision` on, as on `main` today.
    ArbitraryPrecision,
}

fn build() -> Build {
    if numbers_kept_as_written() {
        return Build::ArbitraryPrecision;
    }
    // 0.7.35 writes a GLM number argument as a float; 0.8 keeps an integer.
    let parsed = dynamo_chat_parsing::unified_stream::parse_complete_with_policy(
        "glm47",
        "<tool_call>write_file<arg_key>count</arg_key><arg_value>1e3</arg_value></tool_call>",
        &GuidedToolConstraint::None,
        &tools(),
        Default::default(),
    )
    .expect("GLM parse");
    match parsed.tool_calls[0].function.arguments.as_str() {
        r#"{"count":1000.0}"# => Build::Pinned,
        r#"{"count":1000}"# => Build::PassThrough,
        other => panic!("unknown GLM argument text: {other}"),
    }
}

fn tool_calls(output: &Value) -> Vec<&Value> {
    output["ok"]["choices"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|choice| {
            choice["message"]["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .collect()
}

/// A number's exact value as `<digits>e<exponent>`, so `1e+3`, `1000`, and
/// `1000.0` compare equal and `9007199254740992` and `9007199254740993` do not.
fn exact_number(text: &str) -> String {
    let (sign, rest) = text
        .strip_prefix('-')
        .map_or(("", text), |rest| ("-", rest));
    let (mantissa, exponent) = match rest.find(['e', 'E']) {
        Some(at) => (
            &rest[..at],
            rest[at + 1..].parse::<i64>().expect("exponent"),
        ),
        None => (rest, 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return "0".to_string();
    }
    let significant = digits.trim_end_matches('0');
    let exponent = exponent - fraction.len() as i64 + (digits.len() - significant.len()) as i64;
    format!("{sign}{significant}e{exponent}")
}

/// Every number as its exact value, as a string.
fn exact_numbers(value: &mut Value) {
    match value {
        Value::Number(number) => *value = json!(exact_number(&number.to_string())),
        Value::Array(items) => items.iter_mut().for_each(exact_numbers),
        Value::Object(map) => map.values_mut().for_each(exact_numbers),
        _ => {}
    }
}

/// Each call's arguments parsed, with numbers as exact values.
fn with_parsed_arguments(output: &Value) -> Value {
    let mut output = output.clone();
    for choice in output["ok"]["choices"].as_array_mut().into_iter().flatten() {
        for call in choice["message"]["tool_calls"]
            .as_array_mut()
            .into_iter()
            .flatten()
        {
            let arguments = &mut call["function"]["arguments"];
            if let Some(mut parsed) = arguments
                .as_str()
                .and_then(|text| serde_json::from_str(text).ok())
            {
                exact_numbers(&mut parsed);
                *arguments = parsed;
            }
        }
    }
    output
}

/// `actual` with the kept call taken back out: no tool calls, the finish
/// reason the call turned into `tool_calls` restored, the empty wrapper a
/// guided-JSON fallback leaves restored to dynamo's content, and reasoning
/// that dynamo moved to content (because it had no call) moved again.
fn without_kept_call(expected: &Value, actual: &Value) -> Value {
    let mut actual = actual.clone();
    let expected_choices = expected["ok"]["choices"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for (choice, expected) in actual["ok"]["choices"]
        .as_array_mut()
        .into_iter()
        .flatten()
        .zip(&expected_choices)
    {
        let message = choice["message"].as_object_mut().expect("message");
        message.remove("tool_calls");
        if message.get("content") == Some(&json!("<tool_call>\n\n</tool_call>")) {
            message.insert("content".into(), expected["message"]["content"].clone());
        }
        if message.get("content") == Some(&Value::Null)
            && expected["message"].get("reasoning_content").is_none()
            && message.get("reasoning_content") == expected["message"].get("content")
        {
            let reasoning = message.remove("reasoning_content").expect("reasoning");
            message.insert("content".into(), reasoning);
        }
        if choice["finish_reason"] == json!("tool_calls")
            && matches!(expected["finish_reason"].as_str(), None | Some("stop"))
        {
            choice["finish_reason"] = expected["finish_reason"].clone();
        }
    }
    actual
}

/// Every string in the input and the request options, joined: a kept call's
/// name and ID must come from one of them, or the ID is generated.
fn input_text(value: &Value, text: &mut String) {
    match value {
        Value::String(item) => text.push_str(item),
        Value::Array(items) => items.iter().for_each(|item| input_text(item, text)),
        Value::Object(map) => map.values().for_each(|item| input_text(item, text)),
        _ => {}
    }
}

fn main_drift(input: &Value, expected: &Value, actual: &Value) -> Option<MainDrift> {
    let expected_calls = tool_calls(expected);
    let actual_calls = tool_calls(actual);
    let mut sent = String::new();
    input_text(input, &mut sent);
    if expected_calls.is_empty()
        && !actual_calls.is_empty()
        && actual_calls.iter().all(|call| {
            let arguments = call["function"]["arguments"].as_str().unwrap_or_default();
            let name = call["function"]["name"].as_str().unwrap_or_default();
            let id = call["id"].as_str().unwrap_or_default();
            (arguments.contains("1e400") || arguments.contains("1e+400"))
                && sent.contains(name)
                && (matches!(id, "call-*" | "call_*") || sent.contains(id))
        })
        && &without_kept_call(expected, actual) == expected
    {
        return Some(MainDrift::HugeNumberKept);
    }
    (!expected_calls.is_empty() && with_parsed_arguments(expected) == with_parsed_arguments(actual))
        .then_some(MainDrift::NumberText)
}

/// dynamo's aggregation loop around the moved code.
async fn run(input: &Value, options: &AggregatorOptions) -> Value {
    let mut aggregator = DeltaAggregator::new();
    for item in input.as_array().expect("input items") {
        if let Some(error) = item.get("error") {
            // dynamo-runtime's error for an `Annotated::from_error` item.
            return json!({"err": format!("Internal: {}", error.as_str().expect("error"))});
        }
        let chunk = serde_json::from_value(item["data"].clone()).expect("chunk");
        let completion: Vec<ToolCallCompletion> = item
            .get("completion")
            .map(|value| serde_json::from_value(value.clone()).expect("evidence"))
            .unwrap_or_default();
        aggregator.push(chunk, &completion);
    }
    match aggregator.finish(options).await {
        Ok(response) => {
            let mut value = serde_json::to_value(&response).expect("serialize response");
            normalize(&mut value);
            json!({"ok": compact(value)})
        }
        // dynamo's `DynamoError` display for `ErrorClass::InvalidRequest`.
        Err(error) => json!({"err": format!("InvalidRequest: {error}")}),
    }
}

#[tokio::test]
async fn moved_aggregator_matches_dynamo_llm() {
    let fixture = fixture();
    assert_eq!(fixture.cases.len(), 74_124);
    let build = build();
    let mut drift = std::collections::HashMap::new();
    let mut mismatches = Vec::new();
    let mut drifted = Vec::new();
    for case in &fixture.cases {
        let name = case[0].as_str().expect("name");
        let option_values = &fixture.options[case[1].as_u64().expect("options") as usize];
        let input = &fixture.inputs[case[2].as_u64().expect("input") as usize];
        let auto = case[3].as_u64().expect("output") as usize;
        for (slot, version) in [
            (3, ParserVersion::Auto),
            (4, ParserVersion::V1),
            (5, ParserVersion::V2),
        ] {
            let expected = &fixture.outputs[case[slot].as_u64().map_or(auto, |at| at as usize)];
            let actual = run(input, &options(option_values, version)).await;
            if &actual == expected {
                continue;
            }
            if build != Build::Pinned
                && let Some(kind) = main_drift(&json!([input, option_values]), expected, &actual)
            {
                *drift.entry(kind).or_insert(0usize) += 1;
                drifted.push(json!({
                    "name": name,
                    "version": format!("{version:?}"),
                    "drift": format!("{kind:?}"),
                    "expected": expected,
                    "actual": actual,
                }));
                continue;
            }
            {
                mismatches.push(json!({
                    "name": name,
                    "version": format!("{version:?}"),
                    "expected": expected,
                    "actual": actual,
                }));
            }
        }
    }
    // `PARITY_DUMP=<path>` writes every run that differs, allowed or not.
    if let Ok(path) = std::env::var("PARITY_DUMP") {
        let lines: Vec<String> = mismatches
            .iter()
            .chain(&drifted)
            .map(Value::to_string)
            .collect();
        std::fs::write(path, lines.join("\n")).expect("write PARITY_DUMP");
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} runs differ from dynamo-llm; first 5:\n{}",
        mismatches.len(),
        fixture.cases.len() * 3,
        mismatches[..mismatches.len().min(5)]
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
    let pinned: std::collections::HashMap<_, _> = match build {
        Build::Pinned => [].into(),
        Build::PassThrough => [(MainDrift::NumberText, 144)].into(),
        Build::ArbitraryPrecision => [
            (MainDrift::NumberText, 1_336),
            (MainDrift::HugeNumberKept, 620),
        ]
        .into(),
    };
    assert_eq!(drift, pinned, "runs that differ on the {build:?} build");
}

#[tokio::test]
async fn main_drift_rejects_unrelated_huge_number_changes() {
    let fixture = fixture();
    let name = "structured/Some(\"glm47\")/Some(\"glm45\")/big_number_args/stop/base";
    let case = fixture.cases.iter().find(|case| case[0] == name).unwrap();
    let expected = &fixture.outputs[case[5].as_u64().unwrap() as usize];
    let input = &fixture.inputs[case[2].as_u64().unwrap() as usize];
    let option_values = &fixture.options[case[1].as_u64().unwrap() as usize];
    let actual = run(
        input,
        &options(
            &fixture.options[case[1].as_u64().unwrap() as usize],
            ParserVersion::V2,
        ),
    )
    .await;
    if !numbers_kept_as_written() {
        // Without `arbitrary_precision` no call with `1e400` is kept.
        assert_eq!(&actual, expected);
        return;
    }
    assert_eq!(
        main_drift(&json!([input, option_values]), expected, &actual),
        Some(MainDrift::HugeNumberKept)
    );

    let mut admitted = Vec::new();
    for (label, pointer, replacement) in [
        ("role", "/ok/choices/0/message/role", json!("user")),
        (
            "content",
            "/ok/choices/0/message/content",
            json!("unexpected content"),
        ),
        (
            "tool name",
            "/ok/choices/0/message/tool_calls/0/function/name",
            json!("different_tool"),
        ),
        (
            "tool id",
            "/ok/choices/0/message/tool_calls/0/id",
            json!("different_id"),
        ),
        (
            "finish reason",
            "/ok/choices/0/finish_reason",
            json!("length"),
        ),
    ] {
        let mut changed = actual.clone();
        *changed.pointer_mut(pointer).unwrap() = replacement;
        if main_drift(&json!([input, option_values]), expected, &changed).is_some() {
            admitted.push(label);
        }
    }
    let mut changed_model = actual;
    changed_model["ok"]["model"] = json!("different-model");
    if main_drift(&json!([input, option_values]), expected, &changed_model).is_some() {
        admitted.push("model");
    }
    assert!(
        admitted.is_empty(),
        "accepted unrelated drift: {admitted:?}"
    );
}

#[test]
fn main_drift_rejects_changed_integer_values() {
    let expected = json!({"ok": {"choices": [{"message": {"tool_calls": [{
        "id": "call_a", "type": "function", "function": {
            "name": "write_file", "arguments": "{\"count\":9007199254740992}"
        }
    }]}}]}});
    let mut actual = expected.clone();
    actual["ok"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] =
        json!("{\"count\":9007199254740993}");
    assert_eq!(main_drift(&json!([]), &expected, &actual), None);
}

#[test]
fn replay_covers_optional_envelope_metadata() {
    let fixture = fixture();
    let missing: Vec<_> = ["system_fingerprint", "service_tier"]
        .into_iter()
        .filter(|field| {
            !fixture.inputs.iter().any(|input| {
                input.as_array().unwrap().iter().any(|item| {
                    item.get("data")
                        .and_then(|data| data.get(field))
                        .is_some_and(|value| !value.is_null())
                })
            })
        })
        .collect();
    assert!(
        missing.is_empty(),
        "no nonempty replay inputs for {missing:?}"
    );
}
