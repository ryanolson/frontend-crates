// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost of the v1 route with recording on, per input chunk. Ignored by
//! default. Compare with `v1_route_cost.rs` on the same commit.
//!
//! The reasoning stream, then the sourced jail under the terminal-error
//! contract, for Rhino's three v1 families, over the recorded dynamo streams
//! fed one character per chunk. Each chunk carries one entry with one range,
//! as a caller that records writes it, and the timed pass includes copying
//! the input, as the stock row does.
//!
//! ```text
//! cargo test --release -p dynamo-parsers --test v1_route_cost_on -- --ignored --nocapture
//! ```

// The caller's entry for one chunk is one range.
#![allow(clippy::single_range_in_vec_init)]

mod cost_harness;

use cost_harness::{chars, chunk, corpus, median_ns_per_chunk};
use dynamo_parsers::chat_stream::SourcedChunk;
use dynamo_parsers::reasoning::stream::{
    ReasoningStreamOptions, parse_reasoning_content_from_stream,
};
use dynamo_parsers::sources::ChoiceSources;
use dynamo_parsers::tool_calling::jail::{
    apply_jail_terminal_error_contract, apply_tool_calling_jail_sourced,
};
use futures::{StreamExt, stream};

#[test]
#[ignore = "cost guard: run with --release on a quiet host"]
fn v1_route_cost_on() {
    let corpus = corpus();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for (reasoning, tool) in [
        ("qwen3", "hermes"),
        ("basic", "kimi_k2"),
        ("minimax_m3", "minimax_m3"),
    ] {
        let streams: Vec<Vec<SourcedChunk>> = corpus
            .tool
            .iter()
            .filter(|case| case.parser == tool)
            .map(|case| {
                let mut offset = 0;
                chars(case)
                    .iter()
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
            .collect();
        let chunks: usize = streams.iter().map(Vec::len).sum();
        let ns = median_ns_per_chunk(chunks, || {
            runtime.block_on(async {
                for items in &streams {
                    let reasoned = parse_reasoning_content_from_stream(
                        stream::iter(items.clone()),
                        reasoning.to_string(),
                        ReasoningStreamOptions::default(),
                    )
                    .boxed();
                    let parser = Some(tool.to_string());
                    let out: Vec<_> = apply_jail_terminal_error_contract(reasoned, move |s| {
                        apply_tool_calling_jail_sourced(parser, None, None, false, s)
                    })
                    .collect()
                    .await;
                    std::hint::black_box(out);
                }
            })
        });
        println!("v1 route {reasoning} + {tool}, recording: {ns:.1} ns/chunk");
    }
}
