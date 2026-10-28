// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recording source ranges through the reasoning-off fold and the
//! `force_nonempty_content` deferral must cost few allocations per chunk.
//!
//! Each chunk arrives with one entry and one range, as a caller that feeds
//! tokens writes it. The stage pairs the entry with its choice and writes the
//! entry of its output. Pairing allocated about six times per chunk, and each
//! output list once more. The test counts the stage's allocations per chunk
//! with items that carry a slot against the same chunks without one.

// The caller's entry for one chunk is one range.
#![allow(clippy::single_range_in_vec_init)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use dynamo_parsers::chat_stream::SourcedChunk;
use dynamo_parsers::sources::ChoiceSources;
use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_protocols::types::CreateChatCompletionStreamResponse;
use futures::{StreamExt, stream};

use crate::nonempty_content::defer_unified_reasoning_for_nonempty_content;
use crate::reasoning_off::fold_reasoning_into_content;

thread_local! {
    // Per thread, so the other tests in this binary do not count.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn count() {
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
}

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

struct Counting;

// SAFETY: every call forwards to `System` with the same arguments.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Extra allocations per chunk that recording may cost in each stage.
const EXTRA_PER_CHUNK: f64 = 0.05;

fn chunk(reasoning: bool, text: &str) -> Annotated<CreateChatCompletionStreamResponse> {
    let delta = if reasoning {
        serde_json::json!({"reasoning_content": text})
    } else {
        serde_json::json!({"content": text})
    };
    Annotated {
        data: Some(
            serde_json::from_value(serde_json::json!({
                "id": "a", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
            }))
            .unwrap(),
        ),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

/// 16 streams of reasoning, then visible content, one character per chunk.
fn streams() -> Vec<Vec<(bool, String)>> {
    (0..16)
        .map(|_| (0..256).map(|i| (i < 128, format!("{}", i % 10))).collect())
        .collect()
}

fn sourced(stream: &[(bool, String)]) -> Vec<SourcedChunk> {
    let mut offset = 0;
    stream
        .iter()
        .map(|(reasoning, text)| {
            let range = offset..offset + text.len();
            offset = range.end;
            let (content, reasoning_ranges) = if *reasoning {
                (Vec::new(), vec![range])
            } else {
                (vec![range], Vec::new())
            };
            SourcedChunk {
                chunk: chunk(*reasoning, text),
                sources: Some(vec![ChoiceSources {
                    index: 0,
                    content,
                    reasoning: reasoning_ranges,
                }]),
            }
        })
        .collect()
}

fn stock(stream: &[(bool, String)]) -> Vec<Annotated<CreateChatCompletionStreamResponse>> {
    stream.iter().map(|(r, t)| chunk(*r, t)).collect()
}

#[test]
fn recording_in_the_fold_and_the_deferral_allocates_little_per_chunk() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let streams = streams();
    let chunks: usize = streams.iter().map(Vec::len).sum();
    let mut over = Vec::new();
    for stage in ["fold", "deferral"] {
        let mut counts = [0usize; 2];
        for (side, record) in [false, true].into_iter().enumerate() {
            // Inputs are built before counting, so only the stage counts.
            let stock_input: Vec<_> = streams.iter().map(|s| stock(s)).collect();
            let sourced_input: Vec<_> = streams.iter().map(|s| sourced(s)).collect();
            let before = allocations();
            runtime.block_on(async {
                if record {
                    for items in sourced_input {
                        let out: Vec<SourcedChunk> = match stage {
                            "fold" => {
                                fold_reasoning_into_content(stream::iter(items), true)
                                    .collect()
                                    .await
                            }
                            _ => {
                                defer_unified_reasoning_for_nonempty_content(stream::iter(items))
                                    .collect()
                                    .await
                            }
                        };
                        drop(out);
                    }
                } else {
                    for items in stock_input {
                        let out: Vec<_> = match stage {
                            "fold" => {
                                fold_reasoning_into_content(stream::iter(items), true)
                                    .collect()
                                    .await
                            }
                            _ => {
                                defer_unified_reasoning_for_nonempty_content(stream::iter(items))
                                    .collect()
                                    .await
                            }
                        };
                        drop(out);
                    }
                }
            });
            counts[side] = allocations() - before;
        }
        let extra = (counts[1] as f64 - counts[0] as f64) / chunks as f64;
        println!(
            "{stage}: off {}, on {}, {extra:.3} extra per chunk over {chunks} chunks",
            counts[0], counts[1]
        );
        if extra > EXTRA_PER_CHUNK {
            over.push(format!("{stage}: {extra:.3} extra allocations per chunk"));
        }
    }
    assert!(over.is_empty(), "recording costs too much: {over:?}");
}
