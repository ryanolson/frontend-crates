// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The recorder that wrote `aggregator_parity.jsonl.gz`. This crate does not build it.
//!
//! It ran as a test inside dynamo-llm at ai-dynamo/dynamo `cffc4fa`, as
//! `lib/llm/src/protocols/openai/chat_completions/aggregator_recorder.rs`, with
//! `#[cfg(test)] #[path = "aggregator_recorder.rs"] mod aggregator_recorder;`
//! at the end of `aggregator.rs`. Run it once per `DYN_PARSER_VERSION` value,
//! because dynamo-llm reads the version once per process:
//!
//! ```text
//! PARITY_OUT=auto.jsonl cargo test -p dynamo-llm --lib --no-default-features aggregator_recorder -- --ignored
//! DYN_PARSER_VERSION=1 PARITY_OUT=1.jsonl cargo test -p dynamo-llm --lib --no-default-features aggregator_recorder -- --ignored
//! DYN_PARSER_VERSION=2 PARITY_OUT=2.jsonl cargo test -p dynamo-llm --lib --no-default-features aggregator_recorder -- --ignored
//! python3 pack_aggregator_parity.py aggregator_parity.jsonl.gz auto.jsonl 1.jsonl 2.jsonl
//! ```
//!
//! Each case is one `stream: false` reply: the parser pair and request options
//! dynamo's HTTP handler puts in `ParsingOptions`, and the chunks the
//! aggregator reads. The output is what
//! `NvCreateChatCompletionResponse::from_annotated_stream` returned, the entry
//! point dynamo's chat handler calls.

use super::super::ToolCallCompletion;
use super::super::unified_parser::{self, StreamPrefill, UnifiedRequestPolicy};
use super::*;
use crate::protocols::openai::GuidedToolConstraint;
use dynamo_parsers::tool_calling::ToolDefinition;
use serde_json::{Value, json};
use std::io::Write;

type Item = Annotated<NvCreateChatCompletionStreamResponse>;

/// Parser pairs: Rhino's 8 families first, then the other unified families,
/// then each v1 tool parser alone.
fn parser_pairs() -> Vec<(Option<String>, Option<String>)> {
    let mut pairs: Vec<(Option<&str>, Option<&str>)> = vec![
        (None, Some("deepseek_v4")),            // DeepSeek V4.1 (Rhino)
        (Some("deepseek_v4"), Some("deepseek_v4")), // DeepSeek V4
        (Some("glm45"), Some("glm45")),         // GLM5
        (Some("glm47"), Some("glm45")),         // GLM-5.3
        (Some("harmony"), Some("gpt_oss")),     // GPT-OSS
        (Some("kimi_k2"), Some("basic")),       // Kimi-Linear
        (Some("minimax_m3"), Some("minimax_m3")), // MiniMax
        (Some("hermes"), Some("qwen3")),        // Qwen3
        (Some("qwen3_coder"), Some("qwen3")),
        (Some("deepseek_v41"), Some("deepseek_v41")),
        (Some("kimi_k2"), Some("kimi_k2")),
        (Some("kimi_k3"), Some("kimi_k3")),
        (Some("gemma4"), Some("gemma4")),
        (Some("muse"), Some("muse")),
        (None, Some("qwen3")),
        (None, None),
    ];
    let mut v1: Vec<&str> = dynamo_parsers::tool_calling::parsers::get_available_tool_parsers();
    v1.sort_unstable();
    for parser in v1 {
        pairs.push((Some(parser), None));
    }
    pairs
        .into_iter()
        .map(|(tool, reasoning)| (tool.map(str::to_string), reasoning.map(str::to_string)))
        .collect()
}

/// The 8 Rhino pairs get the option matrix.
const RHINO_PAIRS: usize = 8;

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

/// Whole-reply texts the aggregator re-parses: one or two per grammar, plus the
/// shapes the finalize rules key on (cut-off calls, quoted markers, reasoning
/// markers, numbers, guided JSON).
fn texts() -> Vec<(&'static str, String)> {
    let hermes = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"#;
    let hermes2 = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
<tool_call>
{"name": "write_file", "arguments": {"content": "hi", "count": 2}}
</tool_call>"#;
    let numbers = r#"<tool_call>
{"name": "write_file", "arguments": {"count": 1.50, "path": 1e+5, "content": "x", "big": 1e400, "int": 12345678901234567890}}
</tool_call>"#;
    let glm = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
    let glm_numbers = "<tool_call>write_file<arg_key>count</arg_key><arg_value>1e3</arg_value><arg_key>content</arg_key><arg_value>x</arg_value></tool_call>";
    let qwen_xml = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
    let qwen_xml_numbers = "<tool_call>\n<function=write_file>\n<parameter=count>\n1.50\n</parameter>\n<parameter=content>\nx\n</parameter>\n</function>\n</tool_call>";
    let kimi_k2 = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city":"Paris"}<|tool_call_end|><|tool_calls_section_end|>"#;
    let kimi_k2_numbers = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{"count":1e3,"content":"x"}<|tool_call_end|><|tool_calls_section_end|>"#;
    let kimi_k3 = r#"<|open|>tools<|sep|><|open|>call tool="get_weather" index="1"<|sep|><|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"#;
    let kimi_k3_response = "<|open|>think<|sep|>hmm<|close|>think<|sep|><|open|>response<|sep|>Hello<|close|>response<|sep|>";
    let ds_v4 = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
    let ds_v4_numbers = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"write_file\"><｜DSML｜parameter name=\"count\" string=\"false\">1e3</｜DSML｜parameter><｜DSML｜parameter name=\"content\" string=\"true\">x</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>";
    let ds_v41 = "<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\"><｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
    let ds_v3 = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    let ds_v31 = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
    let harmony_call = r#"<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}<|call|>"#;
    let harmony_final = "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>Hello there.<|return|>";
    let harmony_numbers = r#"<|start|>assistant<|channel|>commentary to=functions.write_file <|constrain|>json<|message|>{"count":1.50,"content":"x"}<|call|>"#;
    let minimax = "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"get_weather\">]<]minimax[>[<city>Paris]<]minimax[>[</city>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>";
    let gemma = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";
    let muse = "<|start|>assistant to=get_weather<|message|><atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke><|eom|>";
    let llama = r#"<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}"#;
    let bare = r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#;
    let mistral = r#"[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris"}}]"#;
    let pythonic = r#"[get_weather(city="Paris")]"#;
    let phi4 = r#"functools[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#;
    let nemotron = r#"<TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</TOOLCALL>"#;
    let jamba = r#"<tool_calls>[{"name": "get_weather", "arguments": {"city": "Paris"}}]</tool_calls>"#;
    vec![
        ("empty", String::new()),
        ("plain", "The answer is 42.".to_string()),
        ("whitespace", "\n\n".to_string()),
        ("quoted_marker", r#"Write "<tool_call>" to open a call."#.to_string()),
        ("hermes", hermes.to_string()),
        ("hermes_prose", format!("Let me check. {hermes}")),
        ("hermes_two", hermes2.to_string()),
        ("hermes_cut", r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Par"#.to_string()),
        ("hermes_cut_second", format!("{hermes}\n<tool_call>\n{{\"name\": \"write_file\", \"arguments\": {{\"content\": \"h")),
        ("hermes_numbers", numbers.to_string()),
        ("hermes_bad_json", "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": Paris}}\n</tool_call>".to_string()),
        ("hermes_empty_args", "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>".to_string()),
        ("glm", glm.to_string()),
        ("glm_numbers", glm_numbers.to_string()),
        ("glm_unfinished", format!("{glm}<tool_call>write_file<arg_key>content</arg_key><arg_value>par")),
        ("qwen_xml", qwen_xml.to_string()),
        ("qwen_xml_numbers", qwen_xml_numbers.to_string()),
        ("qwen_xml_cut", "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar".to_string()),
        ("kimi_k2", kimi_k2.to_string()),
        ("kimi_k2_numbers", kimi_k2_numbers.to_string()),
        ("kimi_k3", kimi_k3.to_string()),
        ("kimi_k3_response", kimi_k3_response.to_string()),
        ("ds_v4", ds_v4.to_string()),
        ("ds_v4_numbers", ds_v4_numbers.to_string()),
        ("ds_v41", ds_v41.to_string()),
        ("ds_v3", ds_v3.to_string()),
        ("ds_v31", ds_v31.to_string()),
        ("harmony_call", harmony_call.to_string()),
        ("harmony_final", harmony_final.to_string()),
        ("harmony_numbers", harmony_numbers.to_string()),
        ("harmony_bare_to", r#"to=functions.get_weather<|channel|>commentary json<|message|>{"city":"Paris"}<|call|>"#.to_string()),
        ("minimax", minimax.to_string()),
        ("gemma", gemma.to_string()),
        ("muse", muse.to_string()),
        ("llama", llama.to_string()),
        ("bare_json", bare.to_string()),
        ("mistral", mistral.to_string()),
        ("pythonic", pythonic.to_string()),
        ("phi4", phi4.to_string()),
        ("nemotron", nemotron.to_string()),
        ("jamba", jamba.to_string()),
        ("think", "<think>Let me think.</think>The answer is 42.".to_string()),
        ("think_prefilled", "Let me think.</think>The answer is 42.".to_string()),
        ("think_only", "<think>Only thinking here.</think>".to_string()),
        ("think_then_call", format!("<think>Need weather.</think>{hermes}")),
        ("think_then_glm", format!("<think>Need weather.</think>{glm}")),
        ("think_then_qwen_xml", format!("<think>Need weather.</think>{qwen_xml}")),
        ("think_then_ds_v4", format!("<think>Need weather.</think>{ds_v4}")),
        ("guided_named", r#"{"city": "Paris"}"#.to_string()),
        ("guided_required", r#"[{"name": "get_weather", "parameters": {"city": "Paris"}}]"#.to_string()),
        ("guided_required_obj", r#"{"name": "write_file", "parameters": {"content": "x", "count": 1.50}}"#.to_string()),
        ("guided_bad", r#"{"city": "#.to_string()),
        ("guided_scalar", "42".to_string()),
        ("json_response", r#"{"answer": 42, "ratio": 1.50}"#.to_string()),
    ]
}

/// A request shape: everything in `ParsingOptions` the aggregator reads.
#[derive(Clone)]
struct Shape {
    name: &'static str,
    tools: bool,
    tool_choice: Option<Value>,
    guided: GuidedToolConstraint,
    suppress_tool_calls: bool,
    reasoning_disabled: bool,
    structured_response: bool,
    parallel_tool_calls: Option<bool>,
    move_reasoning: bool,
    /// dynamo's handler calls `with_tool_call_parsing_enabled(false)` for a
    /// request that forbids tool calls.
    tool_parsing_disabled: bool,
}

impl Shape {
    fn base() -> Self {
        Self {
            name: "base",
            tools: true,
            tool_choice: None,
            guided: GuidedToolConstraint::None,
            suppress_tool_calls: false,
            reasoning_disabled: false,
            structured_response: false,
            parallel_tool_calls: None,
            move_reasoning: false,
            tool_parsing_disabled: false,
        }
    }

    fn matrix() -> Vec<Self> {
        let base = Self::base();
        vec![
            Self { name: "no_tools", tools: false, ..base.clone() },
            Self { name: "suppress", suppress_tool_calls: true, tool_choice: Some(json!("none")), ..base.clone() },
            Self { name: "serial", parallel_tool_calls: Some(false), ..base.clone() },
            Self { name: "move_reasoning", move_reasoning: true, ..base.clone() },
            Self { name: "reasoning_disabled", reasoning_disabled: true, ..base.clone() },
            Self { name: "structured", structured_response: true, ..base.clone() },
            Self {
                name: "named_weather",
                tool_choice: Some(json!({"type": "function", "function": {"name": "get_weather"}})),
                guided: GuidedToolConstraint::GuidedJsonNamed { tool_name: "get_weather".into() },
                ..base.clone()
            },
            Self {
                name: "named_write",
                tool_choice: Some(json!({"type": "function", "function": {"name": "write_file"}})),
                guided: GuidedToolConstraint::GuidedJsonNamed { tool_name: "write_file".into() },
                ..base.clone()
            },
            Self {
                name: "required",
                tool_choice: Some(json!("required")),
                guided: GuidedToolConstraint::GuidedJsonRequired,
                ..base.clone()
            },
            Self {
                name: "tool_parsing_disabled",
                tool_choice: Some(json!("none")),
                tool_parsing_disabled: true,
                ..base.clone()
            },
            Self {
                name: "tool_parsing_disabled_named",
                tool_choice: Some(json!({"type": "function", "function": {"name": "get_weather"}})),
                guided: GuidedToolConstraint::GuidedJsonNamed { tool_name: "get_weather".into() },
                tool_parsing_disabled: true,
                ..base.clone()
            },
            Self {
                name: "structural_tag",
                tool_choice: Some(json!("required")),
                guided: GuidedToolConstraint::StructuralTag,
                ..base.clone()
            },
            base,
        ]
    }

    fn options(&self, pair: &(Option<String>, Option<String>)) -> ParsingOptions {
        let mut options = ParsingOptions::new(pair.0.clone(), pair.1.clone());
        options.tool_choice = self
            .tool_choice
            .clone()
            .map(|choice| serde_json::from_value(choice).unwrap());
        options.guided_tool_constraint = self.guided.clone();
        options.suppress_tool_calls = self.suppress_tool_calls;
        options.reasoning_disabled = self.reasoning_disabled;
        options.structured_response = self.structured_response;
        options.parallel_tool_calls = self.parallel_tool_calls;
        options.move_reasoning_to_content_when_empty = self.move_reasoning;
        options.tools = if self.tools { tools_defs() } else { Vec::new() };
        if self.tool_parsing_disabled {
            options = options.with_tool_call_parsing_enabled(false);
        }
        options
    }

    fn to_json(&self, pair: &(Option<String>, Option<String>)) -> Value {
        json!({
            "tool_call_parser": pair.0,
            "reasoning_parser": pair.1,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "guided_tool_constraint": self.guided,
            "suppress_tool_calls": self.suppress_tool_calls,
            "reasoning_disabled": self.reasoning_disabled,
            "structured_response": self.structured_response,
            "parallel_tool_calls": self.parallel_tool_calls,
            "move_reasoning_to_content_when_empty": self.move_reasoning,
            "tool_parsing_disabled": self.tool_parsing_disabled,
        })
    }
}

fn chunk(choices: Value) -> Value {
    json!({
        "id": "chatcmpl-parity",
        "object": "chat.completion.chunk",
        "created": 1700000000u32,
        "model": "parity-model",
        "choices": choices,
    })
}

fn delta(index: u32, delta: Value, finish: Option<&str>) -> Value {
    json!({"index": index, "delta": delta, "finish_reason": finish})
}

/// One input item as the file stores it.
#[derive(Clone)]
enum In {
    Data { chunk: Value, completion: Vec<ToolCallCompletion> },
    Error(String),
}

impl In {
    fn data(chunk: Value) -> Self {
        Self::Data { chunk, completion: Vec::new() }
    }

    fn to_item(&self) -> Item {
        match self {
            In::Data { chunk, completion } => {
                let mut response: NvCreateChatCompletionStreamResponse =
                    serde_json::from_value(chunk.clone()).unwrap();
                response.tool_call_completion = completion.clone();
                Annotated::from_data(response)
            }
            In::Error(message) => Annotated::from_error(message.clone()),
        }
    }

    fn to_json(&self) -> Value {
        match self {
            In::Data { chunk, completion } => {
                if completion.is_empty() {
                    json!({"data": chunk})
                } else {
                    json!({"data": chunk, "completion": completion})
                }
            }
            In::Error(message) => json!({"error": message}),
        }
    }
}

/// A raw reply as one content delta (or reasoning then content), then the finish.
fn raw_stream(text: &str, reasoning: Option<&str>, finish: Option<&str>) -> Vec<In> {
    let mut items = vec![In::data(chunk(json!([delta(0, json!({"role": "assistant"}), None)])))];
    if let Some(reasoning) = reasoning {
        items.push(In::data(chunk(json!([delta(0, json!({"reasoning_content": reasoning}), None)]))));
    }
    if !text.is_empty() {
        // Split once, mid-text, on a char boundary, so the fold concatenates.
        let mid = text
            .char_indices()
            .map(|(offset, _)| offset)
            .find(|offset| *offset >= text.len() / 2)
            .unwrap_or(0);
        for part in [&text[..mid], &text[mid..]] {
            if !part.is_empty() {
                items.push(In::data(chunk(json!([delta(0, json!({"content": part}), None)]))));
            }
        }
    }
    let mut last = chunk(json!([delta(0, json!({}), finish)]));
    last["usage"] = json!({"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12});
    items.push(In::data(last));
    items
}

fn tool_chunk(index: u32, id: Option<&str>, name: Option<&str>, arguments: Option<&str>) -> Value {
    let mut function = serde_json::Map::new();
    if let Some(name) = name {
        function.insert("name".into(), json!(name));
    }
    if let Some(arguments) = arguments {
        function.insert("arguments".into(), json!(arguments));
    }
    let mut call = json!({"index": index, "function": function});
    if let Some(id) = id {
        call["id"] = json!(id);
        call["type"] = json!("function");
    }
    call
}

/// Streams that already carry structured tool calls, reasoning, logprobs, and
/// several choices: what the stream path hands the aggregator.
fn structured_streams() -> Vec<(&'static str, Vec<In>)> {
    let role = || In::data(chunk(json!([delta(0, json!({"role": "assistant"}), None)])));
    let tc = |calls: Vec<Value>| In::data(chunk(json!([delta(0, json!({"tool_calls": calls}), None)])));
    let fin = |finish: Option<&str>| In::data(chunk(json!([delta(0, json!({}), finish)])));
    let mut cases: Vec<(&'static str, Vec<In>)> = Vec::new();
    for (finish_name, finish) in [
        ("stop", Some("stop")),
        ("length", Some("length")),
        ("tool_calls", Some("tool_calls")),
        ("content_filter", Some("content_filter")),
        ("none", None),
    ] {
        let leak = |name: &'static str| -> &'static str {
            Box::leak(format!("{name}/{finish_name}").into_boxed_str())
        };
        cases.push((leak("split_call"), vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), None)]),
            tc(vec![tool_chunk(0, None, None, Some("{\"city\":"))]),
            tc(vec![tool_chunk(0, None, None, Some("\"Paris\"}"))]),
            fin(finish),
        ]));
        cases.push((leak("interleaved_two"), vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), Some("{\"city\""))]),
            tc(vec![tool_chunk(1, Some("call_b"), Some("write_file"), Some("{\"content\":"))]),
            tc(vec![tool_chunk(0, None, None, Some(":\"Paris\"}"))]),
            tc(vec![tool_chunk(1, None, None, Some("\"x\",\"count\":1.50}"))]),
            fin(finish),
        ]));
        cases.push((leak("cut_last"), vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), Some("{\"city\":\"Paris\"}"))]),
            tc(vec![tool_chunk(1, Some("call_b"), Some("write_file"), Some("{\"content\":\"pa"))]),
            fin(finish),
        ]));
        cases.push((leak("cut_first"), vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), Some("{\"city\":\"Pa"))]),
            tc(vec![tool_chunk(1, Some("call_b"), Some("write_file"), Some("{\"content\":\"x\"}"))]),
            fin(finish),
        ]));
        cases.push((leak("empty_args"), vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), None)]),
            fin(finish),
        ]));
        cases.push((leak("big_number_args"), vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("write_file"), Some("{\"content\":\"x\",\"big\":1e400}"))]),
            fin(finish),
        ]));
        cases.push((leak("no_opener"), vec![
            role(),
            tc(vec![tool_chunk(0, None, None, Some("{\"city\":\"Paris\"}"))]),
            tc(vec![tool_chunk(1, Some("call_b"), None, Some("{}"))]),
            fin(finish),
        ]));
        let mut evidence = vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), Some("{\"city\":\"Paris\"}"))]),
            tc(vec![tool_chunk(1, Some("call_b"), Some("write_file"), Some("{\"content\":\"x\"}"))]),
        ];
        evidence.push(In::Data {
            chunk: chunk(json!([])),
            completion: vec![
                ToolCallCompletion { choice_index: 0, tool_index: 0, complete: true },
                ToolCallCompletion { choice_index: 0, tool_index: 1, complete: false },
            ],
        });
        evidence.push(fin(finish));
        cases.push((leak("evidence_mixed"), evidence));
        let mut late = vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), Some("{\"city\":\"Paris\"}"))]),
        ];
        late.push(In::Data {
            chunk: chunk(json!([])),
            completion: vec![ToolCallCompletion { choice_index: 0, tool_index: 0, complete: false }],
        });
        late.push(In::Data {
            chunk: chunk(json!([])),
            completion: vec![ToolCallCompletion { choice_index: 0, tool_index: 0, complete: true }],
        });
        late.push(fin(finish));
        cases.push((leak("evidence_false_then_true"), late));
        let mut sticky = vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), Some("{\"city\":\"Paris\"}"))]),
        ];
        sticky.push(In::Data {
            chunk: chunk(json!([])),
            completion: vec![ToolCallCompletion { choice_index: 0, tool_index: 0, complete: true }],
        });
        sticky.push(In::Data {
            chunk: chunk(json!([])),
            completion: vec![ToolCallCompletion { choice_index: 0, tool_index: 0, complete: false }],
        });
        sticky.push(fin(finish));
        cases.push((leak("evidence_true_then_false"), sticky));
        cases.push((leak("renamed_call"), vec![
            role(),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), None)]),
            tc(vec![tool_chunk(0, Some("call_b"), Some("write_file"), Some("{\"city\":\"Paris\"}"))]),
            fin(finish),
        ]));
        cases.push((leak("content_and_call"), vec![
            role(),
            In::data(chunk(json!([delta(0, json!({"content": "Checking."}), None)]))),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), Some("{\"city\":\"Paris\"}"))]),
            fin(finish),
        ]));
        cases.push((leak("reasoning_only"), vec![
            role(),
            In::data(chunk(json!([delta(0, json!({"reasoning_content": "Thinking hard."}), None)]))),
            In::data(chunk(json!([delta(0, json!({"content": "\n"}), None)]))),
            fin(finish),
        ]));
        cases.push((leak("reasoning_and_call"), vec![
            role(),
            In::data(chunk(json!([delta(0, json!({"reasoning_content": "Need weather."}), None)]))),
            tc(vec![tool_chunk(0, Some("call_a"), Some("get_weather"), Some("{\"city\":\"Paris\"}"))]),
            fin(finish),
        ]));
        cases.push((leak("reasoning_and_raw_call"), vec![
            role(),
            In::data(chunk(json!([delta(0, json!({"reasoning_content": "Need weather."}), None)]))),
            In::data(chunk(json!([delta(0, json!({"content": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"}), None)]))),
            fin(finish),
        ]));
        cases.push((leak("parts_and_reasoning"), vec![
            role(),
            In::data(chunk(json!([delta(0, json!({"reasoning_content": "Look."}), None)]))),
            In::data(chunk(json!([delta(0, json!({"content": [{"type": "text", "text": "part"}]}), None)]))),
            fin(finish),
        ]));
        cases.push((leak("refusal"), vec![
            role(),
            In::data(chunk(json!([delta(0, json!({"refusal": "I can"}), None)]))),
            In::data(chunk(json!([delta(0, json!({"refusal": "not."}), None)]))),
            fin(finish),
        ]));
        cases.push((leak("two_choices"), vec![
            In::data(chunk(json!([
                delta(1, json!({"role": "assistant", "content": "B says hi."}), None),
                delta(0, json!({"role": "assistant", "reasoning_content": "A thinks."}), None),
            ]))),
            In::data(chunk(json!([
                delta(0, json!({"content": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"}), None),
                delta(1, json!({"tool_calls": [tool_chunk(0, Some("call_b"), Some("write_file"), Some("{\"content\":\"x\"}"))]}), None),
            ]))),
            In::data(chunk(json!([delta(0, json!({}), finish), delta(1, json!({}), finish)]))),
        ]));
        cases.push((leak("logprobs"), vec![
            role(),
            In::data(chunk(json!([{
                "index": 0,
                "delta": {"content": "Hi"},
                "logprobs": {
                    "content": [{"token": "Hi", "logprob": -0.5, "bytes": [72, 105], "top_logprobs": []}],
                    "refusal": [{"token": "No", "logprob": -2.0, "bytes": [78, 111], "top_logprobs": []}]
                },
                "finish_reason": null
            }]))),
            In::data(chunk(json!([{
                "index": 0,
                "delta": {"content": "!"},
                "logprobs": {"content": [{"token": "!", "logprob": -0.25, "bytes": [33], "top_logprobs": []}], "refusal": []},
                "finish_reason": finish
            }]))),
        ]));
    }
    cases.push(("no_role", vec![
        In::data(chunk(json!([delta(0, json!({"content": "Hi"}), Some("stop"))]))),
    ]));
    cases.push(("no_items", vec![]));
    // The last chunk's service tier wins, even when it is absent; the last
    // fingerprint that was sent wins.
    let mut first = chunk(json!([delta(0, json!({"role": "assistant", "content": "Hi"}), None)]));
    first["system_fingerprint"] = json!("fp-1");
    first["service_tier"] = json!("default");
    let mut second = chunk(json!([delta(0, json!({"content": "!"}), None)]));
    second["system_fingerprint"] = json!("fp-2");
    second["service_tier"] = json!("flex");
    let third = chunk(json!([delta(0, json!({}), Some("stop"))]));
    cases.push(("envelope_metadata", vec![In::data(first.clone()), In::data(second.clone()), In::data(third)]));
    cases.push(("envelope_metadata_kept", vec![In::data(first), In::data(second)]));
    cases.push(("error_after_text", vec![
        In::data(chunk(json!([delta(0, json!({"role": "assistant", "content": "Hi"}), None)]))),
        In::Error("backend failed".into()),
    ]));
    cases
}

/// A hyphenated UUID: 36 bytes, hex digits, dashes at 8, 13, 18, and 23.
fn is_uuid(text: &str) -> bool {
    text.len() == 36
        && text.bytes().enumerate().all(|(at, byte)| match at {
            8 | 13 | 18 | 23 => byte == b'-',
            _ => byte.is_ascii_hexdigit(),
        })
}

/// Replace generated call IDs, which are random, so two runs compare equal:
/// `call-<uuid>`, `call_<24 hex digits>`, and a bare UUID.
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

async fn run(options: ParsingOptions, input: &[In]) -> Value {
    let items: Vec<Item> = input.iter().map(In::to_item).collect();
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        futures::stream::iter(items),
        options,
    )
    .await;
    match result {
        Ok(response) => {
            let mut value = serde_json::to_value(&response).unwrap();
            normalize(&mut value);
            json!({"ok": value})
        }
        Err(error) => json!({"err": error.to_string()}),
    }
}

/// Native text for the unified families, through dynamo's own stream adapter:
/// the chunks a `stream: false` request reaches the aggregator with when the
/// v2 adapter ran first.
async fn adapter_stream(
    family: &'static str,
    text: &str,
    shape: &Shape,
    finish: Option<&str>,
) -> Vec<In> {
    let raw: Vec<Item> = raw_stream(text, None, finish).iter().map(In::to_item).collect();
    let output = unified_parser::apply_stream_with_policy(
        futures::stream::iter(raw),
        Some(if shape.tools { tools_defs() } else { Vec::new() }),
        shape.guided.clone(),
        StreamPrefill::from(unified_parser::stream_prefill(family, false)),
        family,
        true,
        UnifiedRequestPolicy {
            reasoning_disabled: shape.reasoning_disabled,
            structured_response: shape.structured_response,
        },
    );
    let items: Vec<Item> = output.collect().await;
    items
        .into_iter()
        .map(|item| match item.data {
            Some(data) => In::Data {
                completion: data.tool_call_completion.clone(),
                chunk: {
                    let mut value = serde_json::to_value(&data.inner).unwrap();
                    normalize(&mut value);
                    value
                },
            },
            None => In::Error(item.error.map(|error| error.to_string()).unwrap_or_default()),
        })
        .collect()
}

#[tokio::test]
#[ignore = "writes the parity fixture; run by hand"]
async fn aggregator_recorder() {
    let path = std::env::var("PARITY_OUT").expect("set PARITY_OUT");
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut write = |name: String, options: Value, input: &[In], output: Value| {
        let line = json!({
            "name": name,
            "options": options,
            "input": input.iter().map(In::to_json).collect::<Vec<_>>(),
            "output": output,
        });
        writeln!(out, "{line}").unwrap();
    };
    let finishes = [
        ("stop", Some("stop")),
        ("length", Some("length")),
        ("tool_calls", Some("tool_calls")),
        ("none", None),
    ];
    let pairs = parser_pairs();
    let texts = texts();
    let base = Shape::base();
    let mut count = 0usize;

    // 1. Every parser pair, every text, every finish, the base request.
    for pair in &pairs {
        for (text_name, text) in &texts {
            for (finish_name, finish) in finishes {
                let input = raw_stream(text, None, finish);
                let output = run(base.options(pair), &input).await;
                let name = format!("raw/{:?}/{:?}/{text_name}/{finish_name}/base", pair.0, pair.1);
                write(name, base.to_json(pair), &input, output);
                count += 1;
            }
        }
    }

    // 2. Rhino's pairs: every text under every request shape, at stop and length,
    //    with and without a reasoning delta before the text.
    for pair in pairs.iter().take(RHINO_PAIRS) {
        for shape in Shape::matrix() {
            for (text_name, text) in &texts {
                for (finish_name, finish) in [("stop", Some("stop")), ("length", Some("length"))] {
                    for reasoning in [None, Some("Earlier thought.")] {
                        if shape.name == "base" && reasoning.is_none() {
                            continue; // recorded in part 1
                        }
                        let input = raw_stream(text, reasoning, finish);
                        let output = run(shape.options(pair), &input).await;
                        let name = format!(
                            "raw/{:?}/{:?}/{text_name}/{finish_name}/{}{}",
                            pair.0,
                            pair.1,
                            shape.name,
                            if reasoning.is_some() { "/reasoning" } else { "" }
                        );
                        write(name, shape.to_json(pair), &input, output);
                        count += 1;
                    }
                }
            }
        }
    }

    // 3. Structured streams, for a spread of pairs and shapes.
    let structured_pairs: Vec<_> = pairs
        .iter()
        .filter(|pair| {
            matches!(
                (pair.0.as_deref(), pair.1.as_deref()),
                (None, None)
                    | (Some("hermes"), Some("qwen3"))
                    | (Some("glm47"), Some("glm45"))
                    | (Some("deepseek_v4"), Some("deepseek_v4"))
                    | (None, Some("deepseek_v4"))
                    | (Some("harmony"), Some("gpt_oss"))
                    | (Some("kimi_k2"), Some("basic"))
                    | (Some("minimax_m3"), Some("minimax_m3"))
                    | (Some("kimi_k3"), Some("kimi_k3"))
            )
        })
        .collect();
    for pair in structured_pairs {
        for shape in Shape::matrix() {
            for (stream_name, input) in structured_streams() {
                let output = run(shape.options(pair), &input).await;
                let name = format!("structured/{:?}/{:?}/{stream_name}/{}", pair.0, pair.1, shape.name);
                write(name, shape.to_json(pair), &input, output);
                count += 1;
            }
        }
    }

    // 4. The v2 adapter's own output, aggregated: Rhino's unified pairs and the
    //    other unified families, native texts, each shape, each finish.
    let adapter_pairs: &[(&str, Option<&str>, Option<&str>)] = &[
        ("deepseek_v4", None, Some("deepseek_v4")),
        ("deepseek_v4", Some("deepseek_v4"), Some("deepseek_v4")),
        ("glm47", Some("glm45"), Some("glm45")),
        ("glm47", Some("glm47"), Some("glm45")),
        ("gpt_oss", Some("harmony"), Some("gpt_oss")),
        ("qwen3", Some("qwen3_coder"), Some("qwen3")),
        ("deepseek_v41", Some("deepseek_v41"), Some("deepseek_v41")),
        ("kimi_k2", Some("kimi_k2"), Some("kimi_k2")),
        ("kimi_k3", Some("kimi_k3"), Some("kimi_k3")),
        ("gemma4", Some("gemma4"), Some("gemma4")),
        ("muse_glimmer", Some("muse"), Some("muse")),
    ];
    for (family, tool, reasoning) in adapter_pairs {
        let pair = (tool.map(str::to_string), reasoning.map(str::to_string));
        for shape in Shape::matrix() {
            for (text_name, text) in &texts {
                for (finish_name, finish) in finishes {
                    let input = adapter_stream(family, text, &shape, finish).await;
                    let output = run(shape.options(&pair), &input).await;
                    let name = format!(
                        "adapter/{family}/{:?}/{:?}/{text_name}/{finish_name}/{}",
                        pair.0, pair.1, shape.name
                    );
                    write(name, shape.to_json(&pair), &input, output);
                    count += 1;
                }
            }
        }
    }
    eprintln!("recorded {count} cases");
}
