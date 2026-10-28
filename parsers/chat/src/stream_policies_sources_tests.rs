// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source recording through the reasoning-off fold and the
//! `force_nonempty_content` deferral.
//!
//! Both policies are crate-internal, so these tests live in the crate. They
//! run every sequence of up to four chunk shapes (reasoning, whitespace,
//! visible text, a tool call, content parts, a finish, and mixes), for one
//! choice and for two interleaved choices, ending with and without a finish.
//! Each chunk's text uses bytes no other chunk uses, so a range that points
//! at the wrong chunk cannot match by accident. Each run checks:
//!
//! * Same output: with a slot, the policy emits the chunks it emits without one.
//! * Verbatim and tiling: each output choice's ranges, read from that choice's
//!   input (reasoning then content, chunk by chunk), equal its `content` and
//!   its `reasoning_content`.
//! * Causality: no range goes past the bytes fed so far.

// A list with one range is the expected value in many cases here.
#![allow(clippy::single_range_in_vec_init)]

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_parsers::chat_stream::{ChatStreamItem, SourcedChunk};
use dynamo_parsers::sources::{ChoiceSources, SourceSlot};
use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_protocols::types::{ChatCompletionMessageContent, CreateChatCompletionStreamResponse};
use futures::{Stream, StreamExt, stream};
use serde_json::{Value, json};

use crate::chat_stream::PolicyStreamItem;
use crate::nonempty_content::defer_unified_reasoning_for_nonempty_content;
use crate::reasoning_off::fold_reasoning_into_content;

impl PolicyStreamItem for SourcedChunk {
    fn has_error(&self) -> bool {
        self.chunk.error.is_some()
    }

    fn clear_id(&mut self) {
        self.chunk.id = None;
    }

    fn chunk_tokens(&self) -> usize {
        0
    }
}

/// One chunk shape for one choice.
#[derive(Clone, Copy, Debug)]
enum Shape {
    Reasoning,
    Whitespace,
    Visible,
    ReasoningAndVisible,
    ReasoningAndWhitespace,
    ToolCall,
    Parts,
    Finish,
    FinishWithText,
}

const SHAPES: &[Shape] = &[
    Shape::Reasoning,
    Shape::Whitespace,
    Shape::Visible,
    Shape::ReasoningAndVisible,
    Shape::ReasoningAndWhitespace,
    Shape::ToolCall,
    Shape::Parts,
    Shape::Finish,
    Shape::FinishWithText,
];

/// Bytes no other chunk uses: a letter per chunk and channel.
fn word(step: usize, index: u32, channel: u8) -> String {
    let letter = char::from(b'a' + ((step * 4 + index as usize * 2 + channel as usize) % 26) as u8);
    format!("{letter}{step}{index}{channel}")
}

fn choice_json(shape: Shape, step: usize, index: u32) -> Value {
    let reasoning = word(step, index, 0);
    let visible = word(step, index, 1);
    let delta = match shape {
        Shape::Reasoning => json!({"reasoning_content": reasoning}),
        Shape::Whitespace => json!({"content": " \n".repeat(step % 2 + 1)}),
        Shape::Visible => json!({"content": visible}),
        Shape::ReasoningAndVisible => json!({"reasoning_content": reasoning, "content": visible}),
        Shape::ReasoningAndWhitespace => json!({"reasoning_content": reasoning, "content": " "}),
        Shape::ToolCall => {
            json!({"tool_calls": [{"index": 0, "id": format!("call-{step}"), "type": "function", "function": {"name": "f", "arguments": "{}"}}]})
        }
        Shape::Parts => json!({"content": [{"type": "text", "text": "parts"}]}),
        Shape::Finish => json!({}),
        Shape::FinishWithText => json!({"content": visible}),
    };
    let finish = matches!(shape, Shape::Finish | Shape::FinishWithText).then_some("stop");
    json!({"index": index, "delta": delta, "finish_reason": finish})
}

fn chunk(choices: Vec<Value>) -> Annotated<CreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(
            serde_json::from_value(json!({
                "id": "sources", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": choices,
            }))
            .unwrap(),
        ),
        id: Some("event".into()),
        event: None,
        comment: None,
        error: None,
    }
}

fn text_of(content: &Option<ChatCompletionMessageContent>) -> &str {
    match content {
        Some(ChatCompletionMessageContent::Text(text)) => text,
        _ => "",
    }
}

/// Each chunk with its slot. A choice's input is its reasoning, then its
/// content, chunk by chunk, all in one offset space.
fn with_sources(chunks: &[Annotated<CreateChatCompletionStreamResponse>]) -> Vec<SourcedChunk> {
    let mut offsets: BTreeMap<u32, usize> = BTreeMap::new();
    chunks
        .iter()
        .map(|chunk| {
            let data = chunk.data.as_ref().unwrap();
            let sources = data
                .choices
                .iter()
                .map(|choice| {
                    let offset = offsets.entry(choice.index).or_default();
                    let mut take = |len: usize| {
                        let range = *offset..*offset + len;
                        *offset += len;
                        if len == 0 { vec![] } else { vec![range] }
                    };
                    let reasoning = take(
                        choice
                            .delta
                            .reasoning_content
                            .as_deref()
                            .map_or(0, str::len),
                    );
                    let content = take(text_of(&choice.delta.content).len());
                    ChoiceSources {
                        index: choice.index,
                        content,
                        reasoning,
                    }
                })
                .collect();
            SourcedChunk {
                chunk: chunk.clone(),
                sources: Some(sources),
            }
        })
        .collect()
}

fn inputs(chunks: &[Annotated<CreateChatCompletionStreamResponse>]) -> Vec<BTreeMap<u32, String>> {
    let mut input: BTreeMap<u32, String> = BTreeMap::new();
    chunks
        .iter()
        .map(|chunk| {
            for choice in &chunk.data.as_ref().unwrap().choices {
                let text = input.entry(choice.index).or_default();
                text.push_str(choice.delta.reasoning_content.as_deref().unwrap_or(""));
                text.push_str(text_of(&choice.delta.content));
            }
            input.clone()
        })
        .collect()
}

fn read(input: &str, ranges: &[Range<usize>]) -> String {
    ranges.iter().map(|range| &input[range.clone()]).collect()
}

type Policy =
    fn(stream::BoxStream<'static, SourcedChunk>) -> stream::BoxStream<'static, SourcedChunk>;

fn fold(
    stream: stream::BoxStream<'static, SourcedChunk>,
) -> stream::BoxStream<'static, SourcedChunk> {
    fold_reasoning_into_content(stream, true).boxed()
}

fn defer(
    stream: stream::BoxStream<'static, SourcedChunk>,
) -> stream::BoxStream<'static, SourcedChunk> {
    defer_unified_reasoning_for_nonempty_content(stream).boxed()
}

async fn collect_with_pulls<S>(output: S, pulled: Arc<AtomicUsize>) -> Vec<(SourcedChunk, usize)>
where
    S: Stream<Item = SourcedChunk>,
{
    output
        .map(|item| {
            let seen = pulled.load(Ordering::SeqCst);
            (item, seen)
        })
        .collect()
        .await
}

async fn check(
    policy: Policy,
    chunks: Vec<Annotated<CreateChatCompletionStreamResponse>>,
    label: &str,
) {
    let stock_input: Vec<SourcedChunk> = chunks
        .iter()
        .cloned()
        .map(|chunk| SourcedChunk {
            chunk,
            sources: None,
        })
        .collect();
    let stock: Vec<SourcedChunk> = policy(stream::iter(stock_input).boxed()).collect().await;
    let pulled = Arc::new(AtomicUsize::new(0));
    let counter = pulled.clone();
    let input = stream::iter(with_sources(&chunks))
        .inspect(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .boxed();
    let sourced = collect_with_pulls(policy(input), pulled).await;
    let shape = |items: Vec<&SourcedChunk>| -> Vec<Value> {
        items
            .into_iter()
            .map(|item| serde_json::to_value(&item.chunk).unwrap())
            .collect()
    };
    assert_eq!(
        shape(stock.iter().collect()),
        shape(sourced.iter().map(|(item, _)| item).collect()),
        "{label}: recording changed the output"
    );
    assert!(
        stock.iter().all(|item| item.sources.is_none()),
        "{label}: stock items got a slot"
    );

    let fed_after = inputs(&chunks);
    let all_input = fed_after.last().cloned().unwrap_or_default();
    for (item, pulled) in &sourced {
        let sources = item.sources().unwrap_or_else(|| panic!("{label}: no slot"));
        let Some(data) = item.chunk() else { continue };
        assert_eq!(
            sources.len(),
            data.choices.len(),
            "{label}: one entry per choice"
        );
        let fed = pulled
            .checked_sub(1)
            .and_then(|at| fed_after.get(at))
            .cloned()
            .unwrap_or_else(|| all_input.clone());
        for choice in &data.choices {
            let entry = sources
                .iter()
                .find(|entry| entry.index == choice.index)
                .unwrap_or_else(|| panic!("{label}: no entry for choice {}", choice.index));
            let input = all_input.get(&choice.index).map_or("", String::as_str);
            assert_eq!(
                read(input, &entry.content),
                text_of(&choice.delta.content),
                "{label}: content"
            );
            assert_eq!(
                read(input, &entry.reasoning),
                choice.delta.reasoning_content.as_deref().unwrap_or(""),
                "{label}: reasoning"
            );
            let fed = fed.get(&choice.index).map_or(0, String::len);
            for range in entry.content.iter().chain(&entry.reasoning) {
                assert!(
                    range.end <= fed,
                    "{label}: {range:?} is after the {fed} bytes fed"
                );
            }
        }
    }
}

/// Every sequence of `len` shapes.
fn sequences(len: usize) -> Vec<Vec<Shape>> {
    let mut out = vec![vec![]];
    for _ in 0..len {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                SHAPES.iter().map(move |shape| {
                    let mut next = prefix.clone();
                    next.push(*shape);
                    next
                })
            })
            .collect();
    }
    out
}

#[tokio::test]
async fn both_policies_on_every_short_sequence() {
    let mut runs = 0;
    for (name, policy) in [("fold", fold as Policy), ("defer", defer as Policy)] {
        for len in 1..=4 {
            for shapes in sequences(len) {
                // One choice.
                let chunks: Vec<_> = shapes
                    .iter()
                    .enumerate()
                    .map(|(step, shape)| chunk(vec![choice_json(*shape, step, 0)]))
                    .collect();
                check(policy, chunks, &format!("{name} {shapes:?}")).await;
                runs += 1;
                // Two choices: choice 1 runs the same shapes one step behind.
                if len <= 3 {
                    let mut chunks = Vec::new();
                    for (step, shape) in shapes.iter().enumerate() {
                        chunks.push(chunk(vec![choice_json(*shape, step, 0)]));
                        if step > 0 {
                            chunks.push(chunk(vec![choice_json(shapes[step - 1], step, 1)]));
                        }
                    }
                    chunks.push(chunk(vec![
                        choice_json(Shape::Finish, len, 0),
                        choice_json(*shapes.last().unwrap(), len, 1),
                    ]));
                    check(policy, chunks, &format!("{name} two choices {shapes:?}")).await;
                    runs += 1;
                }
            }
        }
    }
    println!("{runs} runs per recording mode");
}

#[tokio::test]
async fn folded_reasoning_keeps_its_ranges() {
    // The STRGT-331 case: the request turned reasoning off, and the fold moves
    // reasoning `R` into the front of content `A`.
    let chunks = vec![chunk(vec![
        json!({"index": 0, "delta": {"reasoning_content": "R", "content": "A"}}),
    ])];
    let output: Vec<SourcedChunk> = fold(stream::iter(with_sources(&chunks)).boxed())
        .collect()
        .await;
    let sources = output[0].sources().unwrap();
    assert_eq!(
        text_of(&output[0].chunk().unwrap().choices[0].delta.content),
        "RA"
    );
    assert_eq!(sources[0].content, vec![0..1, 1..2]);
    assert!(sources[0].reasoning.is_empty());
}

#[tokio::test]
async fn deferred_reasoning_released_as_content_keeps_its_ranges() {
    // A reasoning-only turn under force_nonempty_content: the reasoning comes
    // out as content at the finish, dated by its own bytes.
    let chunks = vec![
        chunk(vec![
            json!({"index": 0, "delta": {"reasoning_content": "abc"}}),
        ]),
        chunk(vec![
            json!({"index": 0, "delta": {"reasoning_content": "de"}}),
        ]),
        chunk(vec![
            json!({"index": 0, "delta": {}, "finish_reason": "stop"}),
        ]),
    ];
    let output: Vec<SourcedChunk> = defer(stream::iter(with_sources(&chunks)).boxed())
        .collect()
        .await;
    let last = output.last().unwrap();
    assert_eq!(
        text_of(&last.chunk().unwrap().choices[0].delta.content),
        "abcde"
    );
    assert_eq!(last.sources().unwrap()[0].content, vec![0..3, 3..5]);
}

#[tokio::test]
async fn text_without_ranges_gets_no_entry() {
    let chunks = vec![
        chunk(vec![
            json!({"index": 0, "delta": {"reasoning_content": "r"}}),
        ]),
        chunk(vec![
            json!({"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}),
        ]),
    ];
    let mut items = with_sources(&chunks);
    items[1].sources = Some(vec![]);
    let output: Vec<SourcedChunk> = defer(stream::iter(items).boxed()).collect().await;
    let after: Vec<_> = output.iter().skip(1).collect();
    assert!(!after.is_empty());
    for item in after {
        assert!(item.sources().unwrap().is_empty());
    }
}
