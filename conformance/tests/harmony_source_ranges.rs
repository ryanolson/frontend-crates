// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GPT-OSS source ranges: every Text and Reasoning byte that the Harmony parser
//! emits maps to the input bytes it came from.
//!
//! The parser is fed decoded text, as dynamo-llm feeds it. Harmony encodes that
//! text again with its own tokenizer and reads content back one token at a time,
//! so the ranges it records are checked against the input, not trusted, by the
//! shared checker in `common/source_ranges.rs`.
//!
//! The corpus is every GPT-OSS conformance case, the beam fixtures (Harmony
//! syntax), dynamo's GPT-OSS streams and test literals, and targeted cases for
//! the paths that replay or make up tokens. Each stream runs at its recorded
//! split, one character per chunk, and every two-way split at a character
//! boundary. It runs under the fixture's init and under the init dynamo-llm
//! uses, which replays `<|start|>assistant` as prompt tokens.

mod common;

use std::collections::BTreeMap;
use std::ops::Range;

use common::source_ranges::{
    End, Run, Tally, all_splits, check_all, check_runs, check_splits, committed_text, joined_ranges,
};
use dynamo_parsers_v2::unified::harmony::encode_harmony;
use dynamo_parsers_v2::{
    UnifiedParser, UnifiedParserEvent, UnifiedParserInit, UnifiedParserOutput,
    UnifiedParserStartingState, UnifiedToolOutputMode, create_unified_parser_for_family,
};

const GPT_OSS_FIXTURE: &str = "fixtures-unified-v2/families/gpt_oss/inputs_and_golden.yaml";
const BEAM_GOLDENS: &str = "fixtures-beam/reference-goldens.jsonl";
const BEAM_MESSAGES: &str = "fixtures-beam/original_parser_fixtures.yaml";
const DYNAMO_STREAMS: &str = "tests/data/harmony_dynamo_streams.json";
const HARMONY_MARKERS: &[&str] = &[
    "<|start|>",
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
];

/// Cases for the paths that replay or make up tokens. Bodies avoid repeated
/// n-grams and mix 1 to 4 byte characters, so a misplaced range cannot match.
const TARGETED: &[(&str, &str)] = &[
    (
        "distinct_codepoints",
        "<|channel|>analysis<|message|>αβγ😀δεζ🌆ηθ<|end|><|start|>assistant<|channel|>final<|message|>ab¢€𐍈cd😀ef<|return|>",
    ),
    (
        "outside_text_before_header",
        "Hello there ✓ <|channel|>final<|message|>X<|return|>",
    ),
    (
        "implicit_recipient_then_text",
        "  to=functions.f<|channel|>commentary<|message|>{}<|call|>after ✓",
    ),
    ("unterminated_final_header", "<|channel|>final ¡olé! 🌆"),
    (
        "unterminated_final_header_with_start",
        "<|start|>assistant<|channel|>final tail ✓",
    ),
    (
        "quoted_control_recovery",
        "<|start|>assistant<|channel|>commentary to=functions.f<|message|>{\"a\":\"x<|end|><|start|>assistant<|channel|>final<|message|>visible ζ€😀",
    ),
    ("orphan_controls", "abc<|end|>déf<|message|>ghi😀"),
    (
        "outside_after_turn_end",
        "<|channel|>final<|message|>Aé<|return|>trailing ✓",
    ),
    (
        "held_marker_prefix_at_eof",
        "<|channel|>final<|message|>abc<|",
    ),
    (
        "unlisted_special_token",
        "<|channel|>final<|message|>x<|endoftext|>y<|return|>",
    ),
    (
        "resync_at_start",
        "<|channel|>final<|message|>aé<|start|>assistant<|channel|>final<|message|>b😀",
    ),
    (
        "repeated_ngrams",
        "<|channel|>analysis<|message|>ha ha ha ha<|end|><|start|>assistant<|channel|>final<|message|>ha ha ha<|return|>",
    ),
];

#[derive(Clone)]
struct Stream {
    id: String,
    chunks: Vec<String>,
    starting_state: UnifiedParserStartingState,
}

fn conformance_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(conformance_path(rel)).unwrap_or_else(|error| panic!("{rel}: {error}"))
}

fn starting_state(value: &serde_yaml::Value) -> UnifiedParserStartingState {
    match value.as_str() {
        Some("Reasoning") => UnifiedParserStartingState::Reasoning,
        Some("Response") => UnifiedParserStartingState::Response,
        Some("None") | None => UnifiedParserStartingState::None,
        Some(other) => panic!("unknown starting_state {other}"),
    }
}

/// GPT-OSS cases. Guided cases are returned apart: recording refuses them.
fn gpt_oss_cases() -> (Vec<Stream>, Vec<(String, UnifiedParserInit)>) {
    let document: serde_yaml::Value = serde_yaml::from_str(&read(GPT_OSS_FIXTURE)).unwrap();
    let mut native = Vec::new();
    let mut guided = Vec::new();
    for (name, case) in document["cases"].as_mapping().unwrap() {
        let name = name.as_str().unwrap().to_string();
        let request = &case["request"];
        let init = &request["init"];
        let state = starting_state(&init["starting_state"]);
        let chunks: Vec<String> = request["chunks"]
            .as_sequence()
            .unwrap()
            .iter()
            .map(|chunk| chunk["delta_text"].as_str().unwrap().to_string())
            .collect();
        match init["tool_output_mode"].as_str() {
            Some("Native") => native.push(Stream {
                id: format!("gpt_oss/{name}"),
                chunks,
                starting_state: state,
            }),
            _ => {
                let named_tool = init["tool_output_mode"]["GuidedJson"]["named_tool"]
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| init["named_tool"].as_str().map(str::to_string));
                guided.push((
                    format!("gpt_oss/{name}"),
                    UnifiedParserInit {
                        starting_state: state,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool },
                        ..UnifiedParserInit::default()
                    },
                ))
            }
        }
    }
    (native, guided)
}

fn has_harmony_marker(text: &str) -> bool {
    HARMONY_MARKERS.iter().any(|marker| text.contains(marker))
}

/// Beam fixtures use Harmony syntax with another tool grammar. Fed to GPT-OSS
/// they are adversarial streams: every byte the parser emits must still map back.
fn beam_streams() -> Vec<Stream> {
    let mut streams = Vec::new();
    for line in read(BEAM_GOLDENS)
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        let chunks: Vec<String> = record["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|chunk| chunk.as_str().unwrap().to_string())
            .collect();
        if has_harmony_marker(&chunks.concat()) {
            streams.push(Stream {
                id: format!("beam/{}", record["id"].as_str().unwrap()),
                chunks,
                starting_state: UnifiedParserStartingState::None,
            });
        }
    }
    let document: serde_yaml::Value = serde_yaml::from_str(&read(BEAM_MESSAGES)).unwrap();
    for case in document["message_cases"].as_sequence().unwrap() {
        let text = case["input"]["text"].as_str().unwrap().to_string();
        if has_harmony_marker(&text) {
            streams.push(Stream {
                id: format!("beam_message/{}", case["case_id"].as_str().unwrap()),
                chunks: vec![text],
                starting_state: UnifiedParserStartingState::None,
            });
        }
    }
    streams
}

/// dynamo's recorded GPT-OSS backend streams and Harmony test literals, copied by
/// `tests/data/harmony_dynamo_streams.json` with the dynamo commit it names.
fn dynamo_streams() -> Vec<Stream> {
    let document: serde_json::Value = serde_json::from_str(&read(DYNAMO_STREAMS)).unwrap();
    document["streams"]
        .as_array()
        .unwrap()
        .iter()
        .map(|stream| Stream {
            id: format!("dynamo/{}", stream["id"].as_str().unwrap()),
            chunks: stream["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|chunk| chunk.as_str().unwrap().to_string())
                .collect(),
            starting_state: UnifiedParserStartingState::None,
        })
        .collect()
}

fn targeted_streams() -> Vec<Stream> {
    TARGETED
        .iter()
        .map(|(name, text)| Stream {
            id: format!("targeted/{name}"),
            chunks: vec![text.to_string()],
            starting_state: UnifiedParserStartingState::None,
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum InitMode {
    /// The fixture's init, with no prompt tokens.
    Fixture,
    /// dynamo-llm's streaming init: a native request with no prefilled channel
    /// replays `<|start|>assistant` as prompt tokens.
    DynamoLlm,
}

fn init_for(stream: &Stream, mode: InitMode) -> UnifiedParserInit {
    let prompt_token_ids = match (mode, stream.starting_state) {
        (InitMode::DynamoLlm, UnifiedParserStartingState::None) => {
            encode_harmony("<|start|>assistant").unwrap()
        }
        _ => Vec::new(),
    };
    UnifiedParserInit {
        prompt_token_ids,
        starting_state: stream.starting_state,
        tool_output_mode: UnifiedToolOutputMode::Native,
        ..UnifiedParserInit::default()
    }
}

fn gpt_oss(init: &UnifiedParserInit) -> Box<dyn UnifiedParser> {
    let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
    parser.initialize_request(init.clone()).unwrap();
    parser
}

fn run(chunks: &[&str], init: &UnifiedParserInit, end: End, record: bool) -> Run {
    common::source_ranges::run(&|| gpt_oss(init), chunks, end, record)
}

fn check_stream(stream: &Stream, tally: &mut Tally) {
    let recorded: Vec<&str> = stream.chunks.iter().map(String::as_str).collect();
    for mode in [InitMode::Fixture, InitMode::DynamoLlm] {
        let init = init_for(stream, mode);
        check_splits(
            &|| gpt_oss(&init),
            &format!("{} {mode:?}", stream.id),
            &recorded,
            tally,
        );
    }
}

fn check_corpus(name: &str, streams: Vec<Stream>) -> (Tally, BTreeMap<String, usize>) {
    check_all(name, streams, check_stream)
}

fn assert_clean(name: &str, tally: &Tally, counts: &BTreeMap<String, usize>) {
    let lost: usize = counts
        .iter()
        .filter(|(why, _)| why.starts_with("harmony_source_"))
        .map(|(_, count)| count)
        .sum();
    assert!(
        tally.failures.is_empty() && lost == 0,
        "{name}: {} failing runs (first {} shown), {lost} unattributed commits\n{}",
        tally.failures.len(),
        tally.failures.len(),
        tally.failures.join("\n")
    );
}

#[test]
fn gpt_oss_cases_map_every_byte_to_its_input() {
    let (native, _) = gpt_oss_cases();
    assert_eq!(native.len(), 225, "the GPT-OSS native corpus changed size");
    let (tally, counts) = check_corpus("gpt_oss", native);
    assert_clean("gpt_oss", &tally, &counts);
}

#[test]
fn beam_streams_map_every_byte_to_its_input() {
    let streams = beam_streams();
    assert!(!streams.is_empty());
    eprintln!("beam: {} streams", streams.len());
    let (tally, counts) = check_corpus("beam", streams);
    assert_clean("beam", &tally, &counts);
}

#[test]
fn dynamo_streams_map_every_byte_to_its_input() {
    let streams = dynamo_streams();
    eprintln!("dynamo: {} streams", streams.len());
    let (tally, counts) = check_corpus("dynamo", streams);
    assert_clean("dynamo", &tally, &counts);
}

#[test]
fn targeted_streams_map_every_byte_to_its_input() {
    let (tally, counts) = check_corpus("targeted", targeted_streams());
    assert_clean("targeted", &tally, &counts);
    // Each replay and recovery path must be reached, or a pass says nothing about it.
    for why in [
        "orphan_harmony_control",
        "malformed_harmony_quoted_control",
        "unterminated_harmony_header",
        "harmony_resync_at_start",
    ] {
        assert!(
            counts.get(why).copied().unwrap_or(0) > 0,
            "no targeted run reached {why}"
        );
    }
    assert!(tally.finish_text_runs > 0 && tally.reset_text_runs > 0);
}

#[test]
fn targeted_paths_emit_the_text_they_test() {
    let init = UnifiedParserInit::default();
    let text_of = |input: &str| {
        let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
        parser.initialize_request(init.clone()).unwrap();
        assert!(parser.record_sources());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(input, &mut output).unwrap();
        output.append(&mut parser.finish().unwrap());
        let mut ranges = Vec::new();
        parser.take_sources(&mut ranges);
        let text: String = output
            .events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        (text, ranges)
    };
    let cases = [
        (
            "Hello there ✓ <|channel|>final<|message|>X<|return|>",
            "Hello there ✓ X",
        ),
        ("<|channel|>final ¡olé! 🌆", " ¡olé! 🌆"),
        (
            "<|start|>assistant<|channel|>commentary to=functions.f<|message|>{\"a\":\"x<|end|><|start|>assistant<|channel|>final<|message|>visible ζ€😀",
            "visible ζ€😀",
        ),
        (
            "<|channel|>final<|message|>Aé<|return|>trailing ✓",
            "Aétrailing ✓",
        ),
        ("<|channel|>final<|message|>abc<|", "abc<|"),
    ];
    for (input, expected) in cases {
        let (text, ranges) = text_of(input);
        assert_eq!(text, expected, "{input:?}");
        let rebuilt: String = ranges.iter().map(|range| &input[range.clone()]).collect();
        assert_eq!(rebuilt, expected, "{input:?} ranges {ranges:?}");
    }
}

#[test]
fn reset_text_keeps_the_order_it_is_returned_in() {
    // Held whitespace waits in the outside queue, and a later marker prefix waits
    // in the text queue. `reset` returns the text queue first.
    let input = "  <|chan";
    let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit::default())
        .unwrap();
    assert!(parser.record_sources());
    let mut output = UnifiedParserOutput::default();
    parser.parse_into(input, &mut output).unwrap();
    assert!(output.events.is_empty());
    let recovered = parser.reset();
    let mut ranges = Vec::new();
    parser.take_sources(&mut ranges);
    let rebuilt: String = ranges.iter().map(|range| &input[range.clone()]).collect();
    assert_eq!(rebuilt, recovered);
}

#[test]
fn guided_requests_refuse_to_record() {
    let (_, guided) = gpt_oss_cases();
    assert_eq!(guided.len(), 97, "the GPT-OSS guided corpus changed size");
    for (id, init) in guided {
        let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
        parser.initialize_request(init.clone()).unwrap();
        assert!(
            !parser.record_sources(),
            "{id}: guided mode accepted recording"
        );

        let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
        assert!(parser.record_sources());
        assert!(
            parser.initialize_request(init).is_err(),
            "{id}: guided init after recording must fail, not record nothing"
        );
    }
}

#[test]
fn recording_is_refused_after_parsing_starts() {
    let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit::default())
        .unwrap();
    parser
        .parse_into(
            "<|channel|>final<|message|>a",
            &mut UnifiedParserOutput::default(),
        )
        .unwrap();
    assert!(!parser.record_sources());
}

/// The range of the `len` bytes that start right after the `nth` (from 0)
/// occurrence of `marker` in `input`.
fn after(input: &str, marker: &str, nth: usize, len: usize) -> Range<usize> {
    let start = input.match_indices(marker).nth(nth).unwrap().0 + marker.len();
    start..start + len
}

/// Text that also appears in a header, or repeats. Byte equality cannot tell
/// these occurrences apart, so each case names its exact ranges under the
/// fixture init. The case that opens with a bare word differs under
/// dynamo-llm's init, where that word joins the prompt's open header and is
/// never emitted.
fn exact_cases() -> Vec<(&'static str, Vec<Range<usize>>)> {
    let header_word = "<|channel|>final<|message|>final";
    let both_words = "<|channel|>analysis<|message|>analysis<|end|><|start|>assistant<|channel|>final<|message|>final analysis<|return|>";
    let repeated = "<|channel|>analysis<|message|>ha ha ha ha<|end|><|start|>assistant<|channel|>final<|message|>ha ha ha<|return|>";
    let outside = "final <|channel|>final<|message|>final";
    vec![
        (header_word, vec![after(header_word, "<|message|>", 0, 5)]),
        (
            both_words,
            vec![
                after(both_words, "<|message|>", 0, 8),
                after(both_words, "<|message|>", 1, 14),
            ],
        ),
        (
            repeated,
            vec![
                after(repeated, "<|message|>", 0, 11),
                after(repeated, "<|message|>", 1, 8),
            ],
        ),
        (outside, vec![0..6, after(outside, "<|message|>", 0, 5)]),
    ]
}

#[test]
fn repeated_text_maps_to_its_own_occurrence() {
    for (input, fixture_expected) in exact_cases() {
        for mode in [InitMode::Fixture, InitMode::DynamoLlm] {
            let expected = match mode {
                InitMode::DynamoLlm if input.starts_with("final ") => {
                    fixture_expected[1..].to_vec()
                }
                _ => fixture_expected.clone(),
            };
            let stream = Stream {
                id: input.to_string(),
                chunks: vec![input.to_string()],
                starting_state: UnifiedParserStartingState::None,
            };
            let init = init_for(&stream, mode);
            for chunks in all_splits(input) {
                let on = run(&chunks, &init, End::Finish, true);
                assert_eq!(joined_ranges(&on), expected, "{mode:?} {chunks:?}");
            }
        }
    }
}

#[test]
fn range_checker_rejects_text_sourced_from_the_channel_header() {
    // The sweep's checker cannot tell a header word from the same body word:
    // both rebuild the same bytes. The exact cases can.
    let input = "<|channel|>final<|message|>final";
    let metadata = input.find("final").unwrap();
    let body = input.rfind("final").unwrap();
    assert_ne!(metadata, body);
    let (_, expected) = exact_cases()
        .into_iter()
        .find(|(case, _)| *case == input)
        .unwrap();
    let init = UnifiedParserInit::default();
    let splits = all_splits(input);
    let mut accepted = 0;
    for chunks in &splits {
        let off = run(chunks, &init, End::Finish, false);
        let mut on = run(chunks, &init, End::Finish, true);
        let mut control = Tally::default();
        check_runs("correct sources", input, chunks, &off, &on, &mut control);
        assert!(control.failures.is_empty());
        assert_eq!(joined_ranges(&on), expected);
        assert_eq!(
            on.calls.iter().map(committed_text).collect::<String>(),
            "final"
        );
        for range in on.ranges.iter_mut().flatten() {
            assert!(range.start >= body);
            range.start -= body - metadata;
            range.end -= body - metadata;
        }
        accepted += usize::from(joined_ranges(&on) == expected);
    }
    assert_eq!(
        accepted,
        0,
        "accepted header bytes for {accepted}/{} splits",
        splits.len()
    );
}
