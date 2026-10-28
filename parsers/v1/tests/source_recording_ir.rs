// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Instruction counts of the stock jail `apply` (recording off), for callgrind.
//! Ignored by default. It uses only API that the base commit also has.
//!
//! It runs `JAIL_PASSES` passes (default 1) over the recorded tool streams fed
//! one character per chunk, and prints the chunk count. Run it under callgrind
//! at two pass counts; the difference of the two "Collected" totals, divided
//! by the extra passes times the chunk count, is the Ir per chunk with setup
//! cancelled out:
//!
//! ```text
//! JAIL_PASSES=1 valgrind --tool=callgrind <test binary> --ignored --exact jail_apply_passes --test-threads=1
//! JAIL_PASSES=3 valgrind --tool=callgrind <test binary> --ignored --exact jail_apply_passes --test-threads=1
//! ```

mod cost_harness;

use cost_harness::{chars, chunk, corpus};
use dynamo_parsers::tool_calling::jail::JailedStream;
use futures::{StreamExt, stream};

#[test]
#[ignore = "instruction count: run under callgrind"]
fn jail_apply_passes() {
    let passes: usize = std::env::var("JAIL_PASSES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let streams: Vec<(String, Vec<_>)> = corpus()
        .tool
        .iter()
        .map(|case| {
            (
                case.parser.clone(),
                chars(case).iter().map(|c| chunk(c)).collect(),
            )
        })
        .collect();
    let chunks: usize = streams.iter().map(|(_, chunks)| chunks.len()).sum();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for _ in 0..passes {
        runtime.block_on(async {
            for (parser, items) in &streams {
                let jail = JailedStream::builder().tool_call_parser(parser).build();
                let out: Vec<_> = jail.apply(stream::iter(items.clone())).collect().await;
                std::hint::black_box(out);
            }
        });
    }
    println!("jail chunks per pass: {chunks}, passes: {passes}");
}
