// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parity with dynamo-llm's v2 stream adapter.
//!
//! `tests/data/unified_stream_parity.jsonl.gz` holds input streams and the
//! output that dynamo-llm's own `apply_stream_with_policy` gave for them, before
//! the adapter moved here. It was recorded at ai-dynamo/dynamo
//! `cffc4fa75c15017bcd32e62f7d0fe4bad468c35d` (dynamo-parsers-v2 0.7.35). The
//! adapter's code there is the same as at `45e30e07`. Each line is one case:
//! a family, a request shape (tools, `GuidedToolConstraint`, prompt prefill,
//! `UnifiedRequestPolicy`, guided streaming), and the input and output items.
//!
//! The cases cover 9 families (`deepseek_v4`, `deepseek_v41`, `glm47`,
//! `gpt_oss`, `qwen3`, `kimi_k2`, `kimi_k3`, `gemma4`, `muse_glimmer`): text,
//! reasoning, native calls, quoted markers, cut-off calls, and numeric
//! arguments, at several chunk sizes; then stream shapes (no finish reason, a
//! usage trailer, per-chunk metadata, two choices, an error mid-stream, already
//! parsed reasoning and tool-call chunks, completion evidence, and `length` and
//! `tool_calls` finish reasons); then guided JSON, structured responses, and
//! Gemma's prompt prefixes.
//!
//! dynamo ran the recorder twice: with `DYN_PARSER_VERSION` unset and with
//! `DYN_PARSER_VERSION=2`. The file holds every case from the first run
//! (`parsers_v2_selected: false`) and the 6 cases whose output differed in the
//! second (`parsers_v2_selected: true`). The other cases gave the same output
//! in both runs, so this test also runs each of them with the flag set and
//! expects the same output.
//!
//! On frontend-crates `main`, 16 cases differ in one place each: the
//! arguments of a tool call with a numeric value. dynamo-parsers-v2 0.8 keeps
//! a number as the model wrote it (frontend-crates #339), and 0.7.35 did not.
//! [`PARSERS_V2_0_8_ARGUMENTS`] lists each change. On the 0.7.35 tag, with the
//! same move applied, every case matches the recording with no change.
//!
//! Chunks in the file leave out the fixed envelope (`id`, `object`, `created`,
//! `model`) and null fields. dynamo's `llm_metrics` is written as its
//! `chunk_tokens` value, and `prompt_logprobs` as its length. Generated
//! tool-call IDs are written as `call-*`.

use std::io::Read;

use dynamo_chat_parsing::guided::GuidedToolConstraint;
use dynamo_chat_parsing::stream_item::{ToolCallCompletion, UnifiedStreamChunk};
use dynamo_chat_parsing::unified_stream::{
    StreamPrefill, UnifiedRequestPolicy, apply_stream_with_policy,
};
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_parsers_v2::UnifiedParserStartingState;
use dynamo_protocols::types::CreateChatCompletionStreamResponse;
use futures::StreamExt;
use serde_json::{Map, Value, json};

/// The shape of dynamo-llm's `NvCreateChatCompletionStreamResponse`. The
/// metadata fields hold the markers the recorder wrote.
#[derive(Clone, Debug)]
struct ParityChunk {
    inner: CreateChatCompletionStreamResponse,
    nvext: Option<Value>,
    llm_metrics: Option<u64>,
    prompt_logprobs: Option<u64>,
    tool_call_completion: Vec<ToolCallCompletion>,
}

impl dynamo_parsers::sources::SourceSlot for ParityChunk {}

impl UnifiedStreamChunk for ParityChunk {
    fn inner(&self) -> &CreateChatCompletionStreamResponse {
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut CreateChatCompletionStreamResponse {
        &mut self.inner
    }

    fn tool_call_completion(&self) -> &[ToolCallCompletion] {
        &self.tool_call_completion
    }

    fn tool_call_completion_mut(&mut self) -> &mut Vec<ToolCallCompletion> {
        &mut self.tool_call_completion
    }

    fn clear_nvext(&mut self) {
        self.nvext = None;
    }

    fn clear_llm_metrics(&mut self) {
        self.llm_metrics = None;
    }

    // dynamo's `scrub_synthetic_chunk_metadata`, less the annotation fields,
    // which a synthetic item never has.
    fn scrub_synthetic(&mut self) {
        self.inner.usage = None;
        self.llm_metrics = None;
        self.tool_call_completion.clear();
        self.nvext = None;
        self.prompt_logprobs = None;
    }
}

type Item = Annotated<ParityChunk>;

const ENVELOPE: [(&str, &str); 3] = [
    ("id", "chatcmpl-parity"),
    ("object", "chat.completion.chunk"),
    ("model", "parity-model"),
];

fn fixture() -> Vec<Value> {
    let bytes = include_bytes!("data/unified_stream_parity.jsonl.gz");
    let mut text = String::new();
    flate2::read::GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .expect("decompress fixture");
    text.lines()
        .map(|line| serde_json::from_str(line).expect("fixture line"))
        .collect()
}

fn to_item(value: &Value) -> Item {
    let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
    let data = value.get("data").map(|data| {
        let mut data = data.as_object().expect("data object").clone();
        let nvext = data.remove("nvext");
        let llm_metrics = data.remove("llm_metrics").and_then(|v| v.as_u64());
        let prompt_logprobs = data.remove("prompt_logprobs").and_then(|v| v.as_u64());
        let tool_call_completion = data
            .remove("tool_call_completion")
            .map(|v| serde_json::from_value(v).expect("completion evidence"))
            .unwrap_or_default();
        for (key, default) in ENVELOPE {
            data.entry(key).or_insert(json!(default));
        }
        data.entry("created").or_insert(json!(1));
        ParityChunk {
            inner: serde_json::from_value(Value::Object(data)).expect("chunk"),
            nvext,
            llm_metrics,
            prompt_logprobs,
            tool_call_completion,
        }
    });
    Annotated {
        data,
        id: text("id"),
        event: text("event"),
        comment: value
            .get("comment")
            .map(|v| serde_json::from_value(v.clone()).expect("comment")),
        error: text("error"),
    }
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

fn to_compact(item: &Item) -> Value {
    let mut out = Map::new();
    for (key, field) in [("id", &item.id), ("event", &item.event)] {
        if let Some(field) = field {
            out.insert(key.into(), json!(field));
        }
    }
    if let Some(comment) = &item.comment {
        out.insert("comment".into(), json!(comment));
    }
    if let Some(error) = &item.error {
        out.insert("error".into(), json!(error));
    }
    if let Some(data) = &item.data {
        let mut inner = strip_nulls(serde_json::to_value(&data.inner).expect("serialize chunk"));
        let map = inner.as_object_mut().expect("chunk object");
        for (key, default) in ENVELOPE {
            if map.get(key) == Some(&json!(default)) {
                map.remove(key);
            }
        }
        if map.get("created") == Some(&json!(1)) {
            map.remove("created");
        }
        if let Some(Value::Array(choices)) = map.get_mut("choices") {
            for choice in choices {
                let Some(Value::Array(calls)) = choice.pointer_mut("/delta/tool_calls") else {
                    continue;
                };
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
        if let Some(nvext) = &data.nvext {
            map.insert("nvext".into(), nvext.clone());
        }
        if let Some(metrics) = data.llm_metrics {
            map.insert("llm_metrics".into(), json!(metrics));
        }
        if let Some(logprobs) = data.prompt_logprobs {
            map.insert("prompt_logprobs".into(), json!(logprobs));
        }
        if !data.tool_call_completion.is_empty() {
            map.insert(
                "tool_call_completion".into(),
                serde_json::to_value(&data.tool_call_completion).expect("evidence"),
            );
        }
        out.insert("data".into(), inner);
    }
    Value::Object(out)
}

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

fn family(name: &str) -> &'static str {
    match name {
        "deepseek_v4" => "deepseek_v4",
        "deepseek_v41" => "deepseek_v41",
        "glm47" => "glm47",
        "gpt_oss" => "gpt_oss",
        "qwen3" => "qwen3",
        "kimi_k2" => "kimi_k2",
        "kimi_k3" => "kimi_k3",
        "gemma4" => "gemma4",
        "muse_glimmer" => "muse_glimmer",
        other => panic!("unknown family {other}"),
    }
}

fn starting_state(name: &str) -> UnifiedParserStartingState {
    match name {
        "none" => UnifiedParserStartingState::None,
        "reasoning" => UnifiedParserStartingState::Reasoning,
        "response" => UnifiedParserStartingState::Response,
        other => panic!("unknown prefill {other}"),
    }
}

/// dynamo's prompt prefixes are static slices of Gemma's thought header.
fn prefix(value: &Value) -> Option<&'static str> {
    const HEADER: &str = "<|channel>thought\n";
    let prefix = value.as_str()?;
    assert!(HEADER.starts_with(prefix), "unexpected prefix {prefix:?}");
    Some(&HEADER[..prefix.len()])
}

/// Tool-call arguments that dynamo-parsers-v2 0.8 writes differently from
/// 0.7.35, which dynamo used for the recording: (family and scenario, 0.7.35,
/// 0.8). Each case runs at two chunk sizes. The change is in the parser, not in
/// this adapter.
const PARSERS_V2_0_8_ARGUMENTS: &[(&str, &str, &str)] = &[
    (
        "deepseek_v4/numeric_exp/",
        r#"{"count":1000.0,"content":"-0.0"}"#,
        r#"{"count":1e3,"content":"-0.0"}"#,
    ),
    (
        "deepseek_v41/numeric_exp/",
        r#"{"count":1000.0,"content":"-0.0"}"#,
        r#"{"count":1e3,"content":"-0.0"}"#,
    ),
    (
        "glm47/numeric/",
        r#"{"count":1.0,"path":"n.txt","content":"12345678901234567890"}"#,
        r#"{"count":1,"path":"n.txt","content":"12345678901234567890"}"#,
    ),
    (
        "glm47/numeric_exp/",
        r#"{"count":1000.0,"content":"-0.0"}"#,
        r#"{"count":1000,"content":"-0.0"}"#,
    ),
    (
        "kimi_k2/numeric_exp/",
        r#"{"count":1000.0,"content":"-0.0"}"#,
        r#"{"count":1e3,"content":"-0.0"}"#,
    ),
    // 0.7.35 dropped these Gemma 4 arguments.
    (
        "gemma4/numeric_big/",
        "{}",
        r#"{"count":12345678901234567890,"content":"x"}"#,
    ),
    (
        "gemma4/numeric_exp/",
        "{}",
        r#"{"count":1e3,"content":"-0.0"}"#,
    ),
    (
        "muse_glimmer/numeric_exp/",
        r#"{"count":1000.0,"content":-0.0}"#,
        r#"{"count":1e3,"content":-0.0}"#,
    ),
];

/// The recorded output, with the arguments that dynamo-parsers-v2 0.8 changes.
fn expected_on_main(name: &str, recorded: &Value) -> Value {
    let Some((_, old, new)) = PARSERS_V2_0_8_ARGUMENTS
        .iter()
        .find(|(case, _, _)| name.starts_with(&format!("core/{case}")))
    else {
        return recorded.clone();
    };
    let text = serde_json::to_string(recorded).expect("serialize output");
    let old = serde_json::to_string(old).expect("quote");
    assert_eq!(
        text.matches(&old).count(),
        1,
        "{name}: arguments appear once"
    );
    let new = serde_json::to_string(new).expect("quote");
    serde_json::from_str(&text.replace(&old, &new)).expect("parse output")
}

fn run(case: &Value, parsers_v2_selected: bool) -> Vec<Value> {
    let input: Vec<Item> = case["input"]
        .as_array()
        .expect("input")
        .iter()
        .map(to_item)
        .collect();
    let flag = |key: &str| case[key].as_bool().expect(key);
    let output: Vec<Item> = futures::executor::block_on(
        apply_stream_with_policy(
            futures::stream::iter(input),
            flag("tools").then(tools),
            serde_json::from_value::<GuidedToolConstraint>(case["constraint"].clone())
                .expect("constraint"),
            StreamPrefill {
                starting_state: starting_state(case["prefill"].as_str().expect("prefill")),
                prefix: prefix(&case["prefix"]),
            },
            family(case["family"].as_str().expect("family")),
            flag("guided_streaming"),
            UnifiedRequestPolicy {
                reasoning_disabled: flag("reasoning_disabled"),
                structured_response: flag("structured_response"),
                parsers_v2_selected,
            },
        )
        .collect(),
    );
    output.iter().map(to_compact).collect()
}

#[test]
fn fixture_items_round_trip() {
    for case in fixture() {
        for item in case["input"].as_array().unwrap() {
            assert_eq!(&to_compact(&to_item(item)), item, "{}", case["name"]);
        }
    }
}

#[test]
fn moved_adapter_matches_dynamo_llm() {
    let cases = fixture();
    let mut v2_outputs = std::collections::HashMap::new();
    for case in &cases {
        if case["parsers_v2_selected"] == json!(true) {
            v2_outputs.insert(
                case["name"].as_str().unwrap().to_string(),
                case["output"].clone(),
            );
        }
    }
    let mut failures = Vec::new();
    let mut checked = 0;
    for case in cases
        .iter()
        .filter(|case| case["parsers_v2_selected"] == json!(false))
    {
        let name = case["name"].as_str().unwrap();
        for parsers_v2_selected in [false, true] {
            let recorded = match (parsers_v2_selected, v2_outputs.get(name)) {
                (true, Some(output)) => output,
                _ => &case["output"],
            };
            let expected = expected_on_main(name, recorded);
            let actual = Value::Array(run(case, parsers_v2_selected));
            checked += 1;
            if actual != expected {
                failures.push(format!(
                    "{name} (parsers_v2_selected={parsers_v2_selected})\n  expected: {expected}\n  actual:   {actual}"
                ));
            }
        }
    }
    assert_eq!(v2_outputs.len(), 6, "cases that differ with V2 selected");
    assert!(
        failures.is_empty(),
        "{} of {checked} runs differ from dynamo-llm:\n{}",
        failures.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// A queued Qwen reasoning source carries completion evidence for a call that the
/// adapter renumbered. The recorded corpus does not reach this path.
#[test]
fn qwen_queued_source_remaps_completion_evidence() {
    let case = json!({
        "family": "qwen3",
        "tools": true,
        "constraint": "none",
        "prefill": "none",
        "prefix": null,
        "guided_streaming": false,
        "reasoning_disabled": false,
        "structured_response": false,
        "input": [
            {"data": {"choices": [{"index": 0, "delta": {"content": "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>"}}]}},
            {"data": {"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0,
                "id": "backend-call",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }]}}]}},
            {"data": {"choices": [{"index": 0, "delta": {"content": "<tool_c"}}]}},
            {
                "id": "queued-source",
                "data": {
                    "choices": [{"index": 0, "delta": {"reasoning_content": "detour"}}],
                    "tool_call_completion": [{"choice_index": 0, "tool_index": 0, "complete": true}]
                }
            },
            {"data": {"choices": [{
                "index": 0,
                "delta": {"content": "all><function=get_weather><parameter=city>Rome</parameter></function></tool_call>"},
                "finish_reason": "stop"
            }]}}
        ]
    });
    for selected in [false, true] {
        let output = run(&case, selected);
        let calls: Vec<_> = output
            .iter()
            .filter_map(|item| item.pointer("/data/choices/0/delta/tool_calls/0"))
            .collect();
        assert_eq!(calls[0]["index"], json!(0));
        let backend = calls
            .iter()
            .find(|call| call["id"] == json!("backend-call"))
            .unwrap();
        assert_eq!(backend["index"], json!(1));
        assert!(calls.iter().any(|call| call["index"] == json!(2)));
        let source: Vec<_> = output
            .iter()
            .filter(|item| item["id"] == json!("queued-source"))
            .collect();
        assert_eq!(source.len(), 1);
        assert_eq!(
            source[0].pointer("/data/choices/0/delta/reasoning_content"),
            Some(&json!("detour"))
        );
        assert_eq!(
            source[0]["data"]["tool_call_completion"],
            json!([{"choice_index": 0, "tool_index": 1, "complete": true}]),
            "queued source evidence must use the emitted call index"
        );
    }
}

#[test]
fn non_data_annotations_survive_between_native_fragments() {
    let note = json!({
        "id": "source-note",
        "event": "update",
        "comment": ["backend progress", "second line"]
    });
    let heartbeat = json!({"comment": ["keepalive"]});
    let case = json!({
        "family": "qwen3",
        "tools": true,
        "constraint": "none",
        "prefill": "none",
        "prefix": null,
        "guided_streaming": false,
        "reasoning_disabled": false,
        "structured_response": false,
        "input": [
            {"data": {"choices": [{"index": 0, "delta": {"content": "before <tool_c"}}]}},
            note.clone(),
            heartbeat.clone(),
            {"data": {"choices": [{
                "index": 0,
                "delta": {"content": "all><function=get_weather><parameter=city>Paris</parameter></function></tool_call>"},
                "finish_reason": "stop"
            }]}}
        ]
    });
    for selected in [false, true] {
        let output = run(&case, selected);
        let annotations: Vec<_> = output
            .iter()
            .filter(|item| item.get("data").is_none())
            .cloned()
            .collect();
        assert_eq!(
            annotations,
            vec![note.clone(), heartbeat.clone()],
            "non-data annotations must pass through unchanged and in order"
        );
        let position = output.iter().position(|item| item == &note).unwrap();
        assert_eq!(output[position + 1], heartbeat);
        assert!(output[..position].iter().any(|item| {
            item.pointer("/data/choices/0/delta/content") == Some(&json!("before "))
        }));
        assert!(output[position + 2..].iter().any(|item| {
            item.pointer("/data/choices/0/delta/tool_calls/0/function/name")
                == Some(&json!("get_weather"))
        }));
    }
}
