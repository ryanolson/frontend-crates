// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The recorded corpus through the real v2 parsers with recording on, then
//! through the unified response policies: the tool-call policy, the
//! reasoning-off fold, the `reasoning_tokens` estimate, and the
//! `force_nonempty_content` deferral with its usage hold. Recording must not
//! change a chunk, and the ranges must read back each output choice's text
//! from that choice's input and never overlap.

use std::collections::HashMap;
use std::io::Read;
use std::ops::Range;

use dynamo_parsers::chat_stream::SourcedChunk;
use dynamo_parsers::sources::{ChoiceSources, SourceSlot};
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_parsers_v2::UnifiedParserStartingState;
use dynamo_protocols::types::{ChatCompletionMessageContent, CreateChatCompletionStreamResponse};
use futures::StreamExt;
use serde_json::{Value, json};

use super::apply_unified_response_policies;
use crate::guided::GuidedToolConstraint;
use crate::stream_item::{ToolCallCompletion, UnifiedStreamChunk};
use crate::unified_stream::{StreamPrefill, UnifiedRequestPolicy, apply_stream_with_policy};

#[derive(Clone, Debug)]
struct Slot {
    inner: CreateChatCompletionStreamResponse,
    completion: Vec<ToolCallCompletion>,
    slot: Option<Vec<ChoiceSources>>,
}

impl SourceSlot for Slot {
    fn sources(&self) -> Option<&[ChoiceSources]> {
        self.slot.as_deref()
    }
    fn sources_mut(&mut self) -> Option<&mut Vec<ChoiceSources>> {
        self.slot.as_mut()
    }
}

impl UnifiedStreamChunk for Slot {
    fn inner(&self) -> &CreateChatCompletionStreamResponse {
        &self.inner
    }
    fn inner_mut(&mut self) -> &mut CreateChatCompletionStreamResponse {
        &mut self.inner
    }
    fn tool_call_completion(&self) -> &[ToolCallCompletion] {
        &self.completion
    }
    fn tool_call_completion_mut(&mut self) -> &mut Vec<ToolCallCompletion> {
        &mut self.completion
    }
    fn clear_nvext(&mut self) {}
    fn clear_llm_metrics(&mut self) {}
    fn scrub_synthetic(&mut self) {
        self.inner.usage = None;
        self.completion.clear();
    }
}

fn corpus() -> Vec<Value> {
    let bytes = include_bytes!("../tests/data/unified_stream_parity.jsonl.gz");
    let mut text = String::new();
    flate2::read::GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .unwrap();
    text.lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|case| case["parsers_v2_selected"] == json!(false))
        .collect()
}

fn text_of(content: &Option<ChatCompletionMessageContent>) -> &str {
    match content {
        Some(ChatCompletionMessageContent::Text(text)) => text,
        _ => "",
    }
}

/// The corpus input, with or without slots, and each choice's input text
/// (reasoning then content, chunk by chunk).
// Each chunk's text has one input range per field.
#[allow(clippy::single_range_in_vec_init)]
fn input(case: &Value, record: bool) -> (Vec<Annotated<Slot>>, HashMap<u32, String>) {
    let mut texts: HashMap<u32, String> = HashMap::new();
    let items = case["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| {
            let s = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
            let data = value.get("data").map(|data| {
                let mut data = data.as_object().unwrap().clone();
                for key in ["nvext", "llm_metrics", "prompt_logprobs"] {
                    data.remove(key);
                }
                let completion = data
                    .remove("tool_call_completion")
                    .map(|v| serde_json::from_value(v).unwrap())
                    .unwrap_or_default();
                for (key, default) in [
                    ("id", json!("chatcmpl-e2e")),
                    ("object", json!("chat.completion.chunk")),
                    ("model", json!("e2e")),
                    ("created", json!(1)),
                ] {
                    data.entry(key).or_insert(default);
                }
                let inner: CreateChatCompletionStreamResponse =
                    serde_json::from_value(Value::Object(data)).unwrap();
                let slot = record.then(|| {
                    inner
                        .choices
                        .iter()
                        .map(|choice| {
                            let text = texts.entry(choice.index).or_default();
                            let mut sources = ChoiceSources {
                                index: choice.index,
                                ..ChoiceSources::default()
                            };
                            if let Some(reasoning) = &choice.delta.reasoning_content {
                                sources.reasoning = vec![text.len()..text.len() + reasoning.len()];
                                text.push_str(reasoning);
                            }
                            if let Some(ChatCompletionMessageContent::Text(content)) =
                                &choice.delta.content
                            {
                                sources.content = vec![text.len()..text.len() + content.len()];
                                text.push_str(content);
                            }
                            sources
                        })
                        .collect()
                });
                Slot {
                    inner,
                    completion,
                    slot,
                }
            });
            Annotated {
                data,
                id: s("id"),
                event: s("event"),
                comment: value
                    .get("comment")
                    .map(|v| serde_json::from_value(v.clone()).unwrap()),
                error: s("error"),
            }
        })
        .collect();
    (items, texts)
}

fn weather_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "get_weather".to_string(),
        parameters: Some(json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        })),
        strict: None,
    }]
}

async fn run(case: &Value, record: bool, fold: bool, defer: bool) -> Vec<SourcedChunk> {
    const HEADER: &str = "<|channel>thought\n";
    let family: &'static str = Box::leak(
        case["family"]
            .as_str()
            .unwrap()
            .to_string()
            .into_boxed_str(),
    );
    let constraint: GuidedToolConstraint =
        serde_json::from_value(case["constraint"].clone()).unwrap();
    let prefill = StreamPrefill {
        starting_state: match case["prefill"].as_str().unwrap() {
            "reasoning" => UnifiedParserStartingState::Reasoning,
            "response" => UnifiedParserStartingState::Response,
            _ => UnifiedParserStartingState::None,
        },
        prefix: case["prefix"].as_str().map(|p| &HEADER[..p.len()]),
    };
    let (items, _) = input(case, record);
    let adapted = apply_stream_with_policy(
        futures::stream::iter(items),
        case["tools"].as_bool().unwrap().then(weather_tools),
        constraint,
        prefill,
        family,
        case["guided_streaming"].as_bool().unwrap(),
        UnifiedRequestPolicy {
            reasoning_disabled: case["reasoning_disabled"].as_bool().unwrap(),
            structured_response: case["structured_response"].as_bool().unwrap(),
            parsers_v2_selected: false,
        },
    )
    .map(|item: Annotated<Slot>| {
        let (data, sources) = match item.data {
            Some(slot) => (Some(slot.inner), slot.slot),
            None => (None, None),
        };
        SourcedChunk {
            chunk: Annotated {
                data,
                id: item.id,
                event: item.event,
                comment: item.comment,
                error: item.error,
            },
            sources,
        }
    });
    apply_unified_response_policies(adapted, true, defer, fold)
        .collect()
        .await
}

fn read(text: &str, ranges: &[Range<usize>]) -> Option<String> {
    ranges
        .iter()
        .map(|range| text.get(range.clone()))
        .collect::<Option<Vec<_>>>()
        .map(|parts| parts.concat())
}

fn check(output: &[SourcedChunk], texts: &HashMap<u32, String>) -> Result<(), String> {
    let mut used: HashMap<u32, Vec<Range<usize>>> = HashMap::new();
    for (position, item) in output.iter().enumerate() {
        let Some(data) = &item.chunk.data else {
            continue;
        };
        let Some(sources) = &item.sources else {
            return Err(format!("item {position}: no slot"));
        };
        for choice in &data.choices {
            let Some(entry) = sources.iter().find(|e| e.index == choice.index) else {
                return Err(format!(
                    "item {position}: no entry for choice {}",
                    choice.index
                ));
            };
            let text = texts.get(&choice.index).map_or("", String::as_str);
            let reasoning = choice.delta.reasoning_content.as_deref().unwrap_or("");
            for (field, expected, ranges) in [
                ("content", text_of(&choice.delta.content), &entry.content),
                ("reasoning", reasoning, &entry.reasoning),
            ] {
                if read(text, ranges).as_deref() != Some(expected) {
                    return Err(format!(
                        "item {position}: {field} {expected:?} has ranges {ranges:?} over {text:?}"
                    ));
                }
                used.entry(choice.index)
                    .or_default()
                    .extend(ranges.iter().cloned());
            }
        }
    }
    for (index, mut ranges) in used {
        ranges.sort_by_key(|r| r.start);
        for pair in ranges.windows(2) {
            if pair[0].end > pair[1].start {
                return Err(format!("choice {index}: {pair:?} overlap"));
            }
        }
    }
    Ok(())
}

/// Whether this case refuses to record by design: Kimi K3 and Muse Glimmer
/// always, and Gemma 4 when the prompt prefix ends inside the `thought` label.
fn refuses_by_design(case: &Value) -> bool {
    let partial_label = case["prefix"]
        .as_str()
        .is_some_and(|prefix| prefix.len() > "<|channel>".len());
    matches!(case["family"].as_str(), Some("kimi_k3" | "muse_glimmer")) || partial_label
}

/// A chunk list as JSON, with each generated tool-call id written as `call-*`.
fn id_shape(items: &[SourcedChunk]) -> Vec<Value> {
    items
        .iter()
        .map(|item| {
            let mut value = serde_json::to_value(&item.chunk).unwrap();
            let choices = value
                .pointer_mut("/data/choices")
                .and_then(Value::as_array_mut);
            for choice in choices.into_iter().flatten() {
                let calls = choice
                    .pointer_mut("/delta/tool_calls")
                    .and_then(Value::as_array_mut);
                for call in calls.into_iter().flatten() {
                    if let Some(id) = call.get_mut("id").filter(|id| id.is_string()) {
                        *id = json!("call-*");
                    }
                }
            }
            value
        })
        .collect()
}

#[tokio::test]
async fn the_corpus_records_through_the_policies() {
    let (mut checked, mut refused) = (0, 0);
    let mut failures = Vec::new();
    for case in corpus() {
        let constraint: GuidedToolConstraint =
            serde_json::from_value(case["constraint"].clone()).unwrap();
        if constraint.installs_guided_json() {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let (_, texts) = input(&case, true);
        for fold in [false, true] {
            for defer in [false, true] {
                let run_name = format!("{name} fold={fold} defer={defer}");
                let stock = run(&case, false, fold, defer).await;
                let recorded = run(&case, true, fold, defer).await;
                if stock.iter().any(|item| item.sources.is_some()) {
                    failures.push(format!("{run_name}: a stock item has a slot"));
                }
                let errors = |items: &[SourcedChunk]| -> Vec<String> {
                    items
                        .iter()
                        .filter_map(|item| item.chunk.error.clone())
                        .collect()
                };
                let stock_errors = errors(&stock);
                let new_error = errors(&recorded)
                    .into_iter()
                    .find(|error| !stock_errors.contains(error));
                if refuses_by_design(&case) {
                    if !new_error.is_some_and(|error| error.contains("cannot record sources")) {
                        failures.push(format!("{run_name}: expected a refusal"));
                    }
                    refused += 1;
                    continue;
                }
                if let Some(error) = new_error {
                    failures.push(format!("{run_name}: recording failed: {error}"));
                    continue;
                }
                if id_shape(&stock) != id_shape(&recorded) {
                    failures.push(format!("{run_name}: recording changed the output"));
                    continue;
                }
                if let Err(error) = check(&recorded, &texts) {
                    failures.push(format!("{run_name}: {error}"));
                }
                checked += 1;
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures: {:#?}",
        failures.len(),
        &failures[..failures.len().min(15)]
    );
    assert!(checked > 3600, "{checked} runs checked");
    assert!(refused > 0, "{refused} runs refused");
}

async fn split_reasoning_case() -> (Vec<SourcedChunk>, HashMap<u32, String>) {
    let case = corpus()
        .into_iter()
        .find(|case| case["name"] == "core/qwen3/reason_text/split4/none")
        .unwrap();
    let (_, texts) = input(&case, true);
    let recorded = run(&case, true, false, false).await;
    check(&recorded, &texts).unwrap();
    (recorded, texts)
}

fn move_reasoning_source(output: &mut [SourcedChunk], from: Range<usize>, to: Range<usize>) {
    let mut changed = 0;
    for item in output {
        for entry in item.sources.iter_mut().flatten() {
            for range in &mut entry.reasoning {
                if *range == from {
                    *range = to.clone();
                    changed += 1;
                }
            }
        }
    }
    assert_eq!(changed, 1);
}

#[tokio::test]
async fn the_policy_source_checker_rejects_other_bytes() {
    let (mut output, texts) = split_reasoning_case().await;
    move_reasoning_source(&mut output, 8..12, 0..4);
    assert!(check(&output, &texts).is_err());
}

#[tokio::test]
#[ignore = "F3: the checker accepts a reasoning range moved into stripped markup"]
async fn the_policy_source_checker_rejects_a_range_inside_the_opener() {
    let (mut output, texts) = split_reasoning_case().await;
    assert_eq!(&texts[&0], "<think>Think it over.</think>Answer here.");
    assert_eq!(&texts[&0][2..6], "hink");
    assert_eq!(&texts[&0][8..12], "hink");
    move_reasoning_source(&mut output, 8..12, 2..6);
    assert!(
        check(&output, &texts).is_err(),
        "a range inside the stripped <think> opener cannot source reasoning text"
    );
}
