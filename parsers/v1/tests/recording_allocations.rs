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

mod cost_harness;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use cost_harness::{chars, corpus};
use dynamo_parsers::ReasoningParser;
use dynamo_parsers::reasoning::ReasoningParserType;

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to `System` with the same arguments.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
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
        let before = ALLOCS.load(Relaxed);
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
        ALLOCS.load(Relaxed) - before
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
