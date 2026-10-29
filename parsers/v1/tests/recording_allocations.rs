// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recording source ranges must cost few allocations per chunk.
//!
//! A caller that records, such as a request with a stop string, feeds every
//! token through the parsers with recording on. Each held or released string
//! keeps its input ranges, and most are one range, so a list per string
//! allocated on every chunk. The test feeds the recorded dynamo reasoning
//! streams one character per chunk, with recording off and on, and bounds the
//! extra allocations per chunk that recording makes.

// The caller's entry for one chunk is one range.
#![allow(clippy::single_range_in_vec_init)]

mod cost_harness;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use cost_harness::{chars, chunk, corpus};
use dynamo_parsers::ReasoningParser;
use dynamo_parsers::chat_stream::SourcedChunk;
use dynamo_parsers::reasoning::ReasoningParserType;
use dynamo_parsers::reasoning::stream::{
    ReasoningStreamOptions, parse_reasoning_content_from_stream,
};
use dynamo_parsers::sources::ChoiceSources;
use dynamo_parsers::tool_calling::jail::{
    apply_tool_calling_jail, apply_tool_calling_jail_sourced,
};
use futures::{StreamExt, stream};

struct Counting;

thread_local! {
    // Per thread, so tests running side by side do not count each other.
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn count() {
    let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
}

fn allocations() -> usize {
    ALLOCS.with(Cell::get)
}

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

/// Extra allocations per chunk that recording may cost in the reasoning
/// parser. Each run of released text is one range, so recording needs none
/// beyond the caller's own run list, which grows only now and then.
const REASONING_EXTRA_PER_CHUNK: f64 = 0.05;

#[test]
fn recording_in_the_reasoning_parser_allocates_little_per_chunk() {
    let streams: Vec<(String, Vec<String>)> = corpus()
        .reasoning
        .iter()
        .map(|case| (case.parser.clone(), chars(case)))
        .collect();
    let chunks: usize = streams.iter().map(|(_, chunks)| chunks.len()).sum();
    let mut runs = Vec::new();
    let mut count = |record: bool| {
        let before = allocations();
        for (name, stream) in &streams {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name(name);
            if record && !parser.record_sources() {
                continue;
            }
            for chunk in stream {
                std::hint::black_box(parser.parse_reasoning_streaming_incremental(chunk, &[]));
                parser.take_sources(&mut runs);
                runs.clear();
            }
            std::hint::black_box(parser.finish_reasoning_stream());
            parser.take_sources(&mut runs);
            runs.clear();
        }
        allocations() - before
    };
    count(true);
    let off = count(false);
    let on = count(true);
    let extra = (on as f64 - off as f64) / chunks as f64;
    println!("reasoning: off {off}, on {on}, {extra:.3} extra per chunk over {chunks} chunks");
    assert!(
        extra <= REASONING_EXTRA_PER_CHUNK,
        "recording costs {extra:.3} allocations per chunk, budget {REASONING_EXTRA_PER_CHUNK}"
    );
}

/// Extra allocations per chunk that recording may cost on the v1 route: the
/// reasoning stream, then the tool-call jail. Pairing a chunk's entries with
/// its choices allocated about five times per stage and chunk, and each
/// output entry built new lists. The reasoning stream now reuses the input
/// entry's lists; what remains is the jail's output entries, which the public
/// slot type asks for. It was 16 to 22 per chunk.
const ROUTE_EXTRA_PER_CHUNK: f64 = 2.0;

#[test]
fn recording_on_the_v1_route_allocates_little_per_chunk() {
    let corpus = corpus();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut over = Vec::new();
    for (reasoning, tool) in [
        ("qwen3", "hermes"),
        ("basic", "kimi_k2"),
        ("minimax_m3", "minimax_m3"),
    ] {
        let streams: Vec<Vec<String>> = corpus
            .tool
            .iter()
            .filter(|case| case.parser == tool)
            .map(chars)
            .collect();
        let chunks: usize = streams.iter().map(Vec::len).sum();
        let stock_input = || -> Vec<Vec<_>> {
            streams
                .iter()
                .map(|text| text.iter().map(|c| chunk(c)).collect())
                .collect()
        };
        let sourced_input = || -> Vec<Vec<SourcedChunk>> {
            streams
                .iter()
                .map(|text| {
                    let mut offset = 0;
                    text.iter()
                        .map(|c| {
                            let start = offset;
                            offset += c.len();
                            SourcedChunk {
                                chunk: chunk(c),
                                sources: Some(vec![ChoiceSources {
                                    index: 0,
                                    content: vec![start..offset],
                                    reasoning: Vec::new(),
                                }]),
                            }
                        })
                        .collect()
                })
                .collect()
        };
        // The inputs are built before counting, so only the stages count.
        let input = stock_input();
        let before = allocations();
        runtime.block_on(async {
            for items in input {
                let reasoned = parse_reasoning_content_from_stream(
                    stream::iter(items),
                    reasoning.to_string(),
                    ReasoningStreamOptions::default(),
                );
                let out: Vec<_> =
                    apply_tool_calling_jail(Some(tool.to_string()), None, None, false, reasoned)
                        .collect()
                        .await;
                drop(out);
            }
        });
        let off = allocations() - before;
        let input = sourced_input();
        let before = allocations();
        runtime.block_on(async {
            for items in input {
                let reasoned = parse_reasoning_content_from_stream(
                    stream::iter(items),
                    reasoning.to_string(),
                    ReasoningStreamOptions::default(),
                );
                let out: Vec<_> = apply_tool_calling_jail_sourced(
                    Some(tool.to_string()),
                    None,
                    None,
                    false,
                    reasoned,
                )
                .collect()
                .await;
                drop(out);
            }
        });
        let on = allocations() - before;
        let extra = (on as f64 - off as f64) / chunks as f64;
        println!(
            "{reasoning} + {tool}: off {off}, on {on}, {extra:.3} extra per chunk over {chunks} chunks"
        );
        if extra > ROUTE_EXTRA_PER_CHUNK {
            over.push(format!(
                "{reasoning} + {tool}: {extra:.3} extra allocations per chunk"
            ));
        }
    }
    assert!(over.is_empty(), "recording costs too much: {over:?}");
}
