// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost of the reasoning-off fold and the `force_nonempty_content` deferral
//! on the stock path (items with no source slot), per chunk. Ignored by
//! default. It uses only items that the commit before this work also has, so
//! it runs unchanged there for a before and after comparison:
//!
//! ```text
//! cargo test --release -p dynamo-chat-parsing --lib stream_policies_cost -- --ignored --nocapture
//! ```
//!
//! The input is 64 streams of 256 one-character chunks: reasoning, one
//! whitespace chunk, then content, and a finish.

use std::time::{Duration, Instant};

use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_protocols::types::CreateChatCompletionStreamResponse;
use futures::{StreamExt, stream};

use crate::chat_stream::PolicyStreamItem;
use crate::nonempty_content::defer_unified_reasoning_for_nonempty_content;
use crate::reasoning_off::fold_reasoning_into_content;

type Item = Annotated<CreateChatCompletionStreamResponse>;

impl PolicyStreamItem for Item {
    fn has_error(&self) -> bool {
        self.error.is_some()
    }

    fn clear_id(&mut self) {
        self.id = None;
    }

    fn chunk_tokens(&self) -> usize {
        0
    }
}

fn item(delta: serde_json::Value, finish: Option<&str>) -> Item {
    Annotated {
        data: Some(
            serde_json::from_value(serde_json::json!({
                "id": "cost", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
            }))
            .unwrap(),
        ),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

fn streams() -> Vec<Vec<Item>> {
    (0..64)
        .map(|_| {
            let mut items: Vec<Item> = (0..128)
                .map(|i| {
                    item(
                        serde_json::json!({"reasoning_content": format!("{}", i % 10)}),
                        None,
                    )
                })
                .collect();
            items.push(item(serde_json::json!({"content": " "}), None));
            items.extend(
                (0..126).map(|i| item(serde_json::json!({"content": format!("{}", i % 10)}), None)),
            );
            items.push(item(serde_json::json!({}), Some("stop")));
            items
        })
        .collect()
}

const ROUNDS: usize = 15;

fn median_ns_per_chunk(chunks: usize, mut run: impl FnMut()) -> f64 {
    let mut times: Vec<Duration> = (0..ROUNDS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[ROUNDS / 2].as_nanos() as f64 / chunks as f64
}

#[test]
#[ignore = "cost guard: run with --release on a quiet host"]
fn stream_policies_cost() {
    let streams = streams();
    let chunks: usize = streams.iter().map(Vec::len).sum();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let fold = median_ns_per_chunk(chunks, || {
        runtime.block_on(async {
            for items in &streams {
                let out: Vec<Item> = fold_reasoning_into_content(stream::iter(items.clone()), true)
                    .collect()
                    .await;
                std::hint::black_box(out);
            }
        })
    });
    let defer = median_ns_per_chunk(chunks, || {
        runtime.block_on(async {
            for items in &streams {
                let out: Vec<Item> =
                    defer_unified_reasoning_for_nonempty_content(stream::iter(items.clone()))
                        .collect()
                        .await;
                std::hint::black_box(out);
            }
        })
    });
    println!("fold reasoning into content, stock: {fold:.1} ns/chunk");
    println!("force_nonempty_content deferral, stock: {defer:.1} ns/chunk");
}

/// The same streams with recording on: each chunk carries one entry with one
/// range, for its reasoning or its content.
#[allow(clippy::single_range_in_vec_init)]
fn sourced_streams() -> Vec<Vec<dynamo_parsers::chat_stream::SourcedChunk>> {
    use dynamo_parsers::chat_stream::SourcedChunk;
    use dynamo_parsers::sources::ChoiceSources;
    use dynamo_protocols::types::ChatCompletionMessageContent;
    streams()
        .into_iter()
        .map(|items| {
            let mut offset = 0;
            items
                .into_iter()
                .map(|item| {
                    let mut sources = ChoiceSources::default();
                    if let Some(choice) = item.data.as_ref().and_then(|data| data.choices.first()) {
                        if let Some(reasoning) = &choice.delta.reasoning_content {
                            sources.reasoning = vec![offset..offset + reasoning.len()];
                            offset += reasoning.len();
                        }
                        if let Some(ChatCompletionMessageContent::Text(text)) =
                            &choice.delta.content
                        {
                            sources.content = vec![offset..offset + text.len()];
                            offset += text.len();
                        }
                    }
                    SourcedChunk {
                        chunk: item,
                        sources: Some(vec![sources]),
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "cost guard: run with --release on a quiet host"]
fn stream_policies_cost_on() {
    let streams = sourced_streams();
    let chunks: usize = streams.iter().map(Vec::len).sum();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let fold = median_ns_per_chunk(chunks, || {
        runtime.block_on(async {
            for items in &streams {
                let out: Vec<_> = fold_reasoning_into_content(stream::iter(items.clone()), true)
                    .collect()
                    .await;
                std::hint::black_box(out);
            }
        })
    });
    let defer = median_ns_per_chunk(chunks, || {
        runtime.block_on(async {
            for items in &streams {
                let out: Vec<_> =
                    defer_unified_reasoning_for_nonempty_content(stream::iter(items.clone()))
                        .collect()
                        .await;
                std::hint::black_box(out);
            }
        })
    });
    println!("fold reasoning into content, recording: {fold:.1} ns/chunk");
    println!("force_nonempty_content deferral, recording: {defer:.1} ns/chunk");
}
