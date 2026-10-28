// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parse cost per family, with source recording on and off.
//!
//! Each family parses every native input in its conformance corpus, four
//! characters per chunk, and the bench prints the median nanoseconds per input
//! byte over the repeats. Parser construction is outside the timed loop.
//!
//! ```text
//! cargo bench -p dynamo-parsers-v2 --bench source_recording
//! ```
//!
//! Without `--bench` (as `cargo test --benches` runs it), one pass runs as a
//! smoke check and prints no timings.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use dynamo_parsers_v2::{
    Tool, UnifiedParser, UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState,
    create_unified_parser_for_family,
};

/// The families that record, by corpus name.
const FAMILIES: &[&str] = &[
    "beam",
    "deepseek_v4",
    "deepseek_v41",
    "gemma4",
    "glm47",
    "gpt_oss",
    "kimi_k2",
    "nemotron_v3",
    "qwen3",
];
const CHARS_PER_CHUNK: usize = 4;
const REPEATS: usize = 15;

struct Case {
    chunks: Vec<String>,
    tools: Vec<Tool>,
    init: UnifiedParserInit,
}

#[derive(serde::Deserialize)]
struct Document {
    cases: BTreeMap<String, CaseDocument>,
}

#[derive(serde::Deserialize)]
struct CaseDocument {
    #[serde(default)]
    request: Option<Request>,
}

/// Read with types from the text: as an untyped value, a chunk such as `42e0`
/// is a number and not text.
#[derive(serde::Deserialize)]
struct Request {
    init: Init,
    tools: Vec<Tool>,
    chunks: Vec<Chunk>,
}

#[derive(serde::Deserialize)]
struct Init {
    starting_state: String,
    tool_output_mode: String,
}

#[derive(serde::Deserialize)]
struct Chunk {
    delta_text: String,
}

fn corpus(family: &str) -> Option<Vec<Case>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/fixtures-unified-v2/families")
        .join(family)
        .join("inputs_and_golden.yaml");
    let text = std::fs::read_to_string(&path).ok()?;
    let document: Document =
        serde_yaml::from_str(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut cases = Vec::new();
    for request in document.cases.into_values().filter_map(|case| case.request) {
        if request.init.tool_output_mode != "Native" {
            continue;
        }
        let starting_state = match request.init.starting_state.as_str() {
            "Reasoning" => UnifiedParserStartingState::Reasoning,
            "Response" => UnifiedParserStartingState::Response,
            _ => UnifiedParserStartingState::None,
        };
        let input: String = request.chunks.into_iter().map(|c| c.delta_text).collect();
        let chars: Vec<char> = input.chars().collect();
        cases.push(Case {
            chunks: chars
                .chunks(CHARS_PER_CHUNK)
                .map(|chunk| chunk.iter().collect())
                .collect(),
            tools: request.tools,
            init: UnifiedParserInit {
                starting_state,
                ..UnifiedParserInit::default()
            },
        });
    }
    Some(cases)
}

fn parsers(family: &str, cases: &[Case], record: bool) -> Vec<Box<dyn UnifiedParser>> {
    cases
        .iter()
        .map(|case| {
            let mut parser = create_unified_parser_for_family(family, &case.tools).unwrap();
            parser.initialize_request(case.init.clone()).unwrap();
            if record {
                assert!(parser.record_sources(), "{family} refused to record");
            }
            parser
        })
        .collect()
}

/// Parse every case once and return the elapsed nanoseconds.
fn pass(family: &str, cases: &[Case], record: bool) -> u128 {
    let mut parsers = parsers(family, cases, record);
    let mut output = UnifiedParserOutput::default();
    let mut ranges = Vec::new();
    let start = Instant::now();
    for (parser, case) in parsers.iter_mut().zip(cases) {
        for chunk in &case.chunks {
            output.events.clear();
            // A parse error is part of some cases; recovery is not timed here.
            if parser.parse_into(chunk, &mut output).is_err() {
                break;
            }
            ranges.clear();
            parser.take_sources(&mut ranges);
            black_box((&output, &ranges));
        }
        black_box(parser.finish().ok());
        ranges.clear();
        parser.take_sources(&mut ranges);
        black_box(&ranges);
    }
    start.elapsed().as_nanos()
}

fn median(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let bench = std::env::args().any(|arg| arg == "--bench");
    let repeats = if bench { REPEATS } else { 1 };
    if bench {
        println!("family        bytes   off ns/B   on ns/B   on/off");
    }
    for family in FAMILIES {
        let Some(cases) = corpus(family) else {
            println!("{family}: corpus not found, skipped");
            continue;
        };
        let bytes: usize = cases
            .iter()
            .flat_map(|case| &case.chunks)
            .map(String::len)
            .sum();
        let mut off = Vec::new();
        let mut on = Vec::new();
        for _ in 0..repeats {
            off.push(pass(family, &cases, false));
            on.push(pass(family, &cases, true));
        }
        if bench {
            let off = median(off) as f64 / bytes as f64;
            let on = median(on) as f64 / bytes as f64;
            println!(
                "{family:<12} {bytes:>7} {off:>10.1} {on:>9.1} {:>8.3}",
                on / off
            );
        }
    }
}
