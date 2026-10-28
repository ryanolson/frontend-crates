// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source ranges for every Unified family on the shared scanner: every Text and
//! Reasoning byte that the parser emits maps to the input bytes it came from.
//!
//! Each native case of each family runs at its recorded split, one character per
//! chunk, and every two-way split, with recording on and off, under the shared
//! checker in `common/source_ranges.rs`. A family that refuses to record still
//! runs, so the sweep shows that asking to record changes nothing for it.
//!
//! GPT-OSS has its own sweep in `harmony_source_ranges.rs`, with dynamo-llm's
//! prompt replay as a second init.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::Init;
use common::source_ranges::{Tally, check_all, check_splits};
use common::unified_capture::{InputChunk, replay_chunks};
use dynamo_parsers_v2::{
    REGISTERED_UNIFIED_FAMILIES, Tool, UnifiedParser, UnifiedParserEvent, UnifiedParserInit,
    UnifiedParserOutput, UnifiedToolOutputMode, create_unified_parser_for_family,
};

/// Families whose parser records sources. Every other built-in family must
/// refuse, so a new family cannot be added without a ruling here.
const RECORDING: &[&str] = &[
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

/// Families with their own state machine, not the shared scanner. They keep
/// the default `record_sources`, which returns false.
const REFUSING: &[&str] = &["kimi_k3", "muse_glimmer", "muse_spark"];

struct Case {
    id: String,
    family: String,
    chunks: Vec<String>,
    tools: Vec<Tool>,
    init: Init,
}

impl Case {
    fn parser(&self) -> Box<dyn UnifiedParser> {
        let mut parser = create_unified_parser_for_family(&self.family, &self.tools).unwrap();
        self.init.apply(&mut parser, &self.id);
        parser
    }

    fn guided(&self) -> bool {
        self.init.output_mode() != UnifiedToolOutputMode::Native
    }
}

fn families_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures-unified-v2/families")
}

fn corpus_families() -> Vec<String> {
    let mut families: Vec<String> = std::fs::read_dir(families_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    families.sort();
    families
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

/// Read with types, not as a `serde_yaml::Value`: a chunk such as `42e0` is a
/// number to the untyped reader and must stay text.
#[derive(serde::Deserialize)]
struct Request {
    input: String,
    init: Init,
    tools: serde_json::Value,
    chunks: Vec<InputChunk>,
}

fn cases(family: &str) -> Vec<Case> {
    let path = families_dir().join(family).join("inputs_and_golden.yaml");
    let text = std::fs::read_to_string(&path).unwrap();
    let document: Document =
        serde_yaml::from_str(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    document
        .cases
        .into_iter()
        .filter_map(|(name, case)| {
            let request = case.request?;
            Some(Case {
                id: format!("{family}/{name}"),
                family: family.to_string(),
                chunks: replay_chunks(&request.input, Some(&request.chunks)),
                tools: common::unified_tools_for_schemas(Some(&request.tools)),
                init: request.init,
            })
        })
        .collect()
}

fn check_case(case: &Case, tally: &mut Tally) {
    let recorded: Vec<&str> = case.chunks.iter().map(String::as_str).collect();
    check_splits(&|| case.parser(), &case.id, &recorded, tally);
}

fn sweep(family: &str) {
    let native: Vec<Case> = cases(family)
        .into_iter()
        .filter(|case| !case.guided())
        .collect();
    assert!(!native.is_empty(), "{family}: no native cases");
    let records = native[0].parser().record_sources();
    assert_eq!(
        records,
        RECORDING.contains(&family),
        "{family}: record_sources does not match the RECORDING list"
    );
    eprintln!("{family}: {} native cases", native.len());
    let (tally, _) = check_all(family, native, check_case);
    assert!(
        tally.failures.is_empty(),
        "{family}: {} failing runs (first {} shown)\n{}",
        tally.failures.len(),
        tally.failures.len(),
        tally.failures.join("\n")
    );
    if records {
        assert!(tally.text_bytes > 0, "{family}: no committed text checked");
        assert!(
            tally.reset_text_runs > 0,
            "{family}: no run returned reset text"
        );
    }
}

#[test]
fn every_family_is_ruled_on() {
    let corpus: BTreeSet<String> = corpus_families().into_iter().collect();
    let ruled: BTreeSet<String> = RECORDING
        .iter()
        .chain(REFUSING)
        .map(|family| family.to_string())
        .collect();
    assert_eq!(
        corpus, ruled,
        "a corpus family is missing from RECORDING or REFUSING"
    );
    for family in REGISTERED_UNIFIED_FAMILIES {
        let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        let canonical = match *family {
            "qwen3_coder" => "qwen3",
            "nemotron3" => "nemotron_v3",
            "muse_spark_open" => "muse_spark",
            "kimi-k3" => "kimi_k3",
            "harmony" => "gpt_oss",
            other => other,
        };
        assert!(
            ruled.contains(canonical),
            "{family}: registered but not ruled on"
        );
        assert_eq!(
            parser.record_sources(),
            RECORDING.contains(&canonical),
            "{family}"
        );
    }
}

#[test]
fn guided_requests_refuse_to_record() {
    let mut checked = 0;
    for family in RECORDING.iter().filter(|family| **family != "gpt_oss") {
        for case in cases(family).into_iter().filter(Case::guided) {
            let mut parser = case.parser();
            assert!(
                !parser.record_sources(),
                "{}: guided mode accepted recording",
                case.id
            );
            let mut parser = create_unified_parser_for_family(family, &case.tools).unwrap();
            assert!(parser.record_sources(), "{}", case.id);
            let init = UnifiedParserInit {
                starting_state: case.init.starting_state(),
                tool_output_mode: case.init.output_mode(),
                ..UnifiedParserInit::default()
            };
            assert!(
                parser.initialize_request(init).is_err(),
                "{}: guided init after recording must fail, not record nothing",
                case.id
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}

/// After input, recording starts only when nothing held could surface, and its
/// offsets count from there. Harmony records only before its first parse; a
/// scanner family that still holds a byte of `a ` refuses. When recording
/// starts, the next text reads back from offset 0. (A held prompt prefix is
/// `parsers/v2/tests/source_ranges_prompt_prefix.rs`.)
#[test]
fn recording_after_input_counts_from_where_it_starts() {
    let mut recorded = 0;
    for family in RECORDING {
        let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        parser
            .parse_into("a ", &mut UnifiedParserOutput::default())
            .unwrap();
        if *family == "gpt_oss" {
            assert!(!parser.record_sources(), "{family}");
            continue;
        }
        if !parser.record_sources() {
            continue;
        }
        recorded += 1;
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("bc ", &mut output).unwrap();
        output.append(&mut parser.finish().unwrap());
        let mut ranges = Vec::new();
        parser.take_sources(&mut ranges);
        let text: String = output
            .events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::Text(text) | UnifiedParserEvent::Reasoning(text) => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect();
        let read: String = ranges.iter().map(|range| &"bc "[range.clone()]).collect();
        assert_eq!(read, text, "{family}");
    }
    assert!(recorded >= 5, "{recorded} families recorded after input");
}

#[test]
fn beam_cases_map_every_byte_to_its_input() {
    sweep("beam");
}

#[test]
fn deepseek_v4_cases_map_every_byte_to_its_input() {
    sweep("deepseek_v4");
}

#[test]
fn deepseek_v41_cases_map_every_byte_to_its_input() {
    sweep("deepseek_v41");
}

#[test]
fn gemma4_cases_map_every_byte_to_its_input() {
    sweep("gemma4");
}

#[test]
fn glm47_cases_map_every_byte_to_its_input() {
    sweep("glm47");
}

#[test]
fn kimi_k2_cases_map_every_byte_to_its_input() {
    sweep("kimi_k2");
}

#[test]
fn nemotron_v3_cases_map_every_byte_to_its_input() {
    sweep("nemotron_v3");
}

#[test]
fn qwen3_cases_map_every_byte_to_its_input() {
    sweep("qwen3");
}

#[test]
fn kimi_k3_output_does_not_change_when_asked_to_record() {
    sweep("kimi_k3");
}

#[test]
fn muse_glimmer_output_does_not_change_when_asked_to_record() {
    sweep("muse_glimmer");
}

#[test]
fn muse_spark_output_does_not_change_when_asked_to_record() {
    sweep("muse_spark");
}
