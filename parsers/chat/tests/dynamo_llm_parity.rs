// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replays dynamo-llm's own output for the moved rules and checks this crate gives the
//! same answer on the same input.
//!
//! `data/dynamo_llm_parity.jsonl` was recorded by a test inside dynamo-llm, at
//! ai-dynamo/dynamo `cffc4fa` (`dynamo-parsers` 10.0.7, `dynamo-parsers-v2` 0.7.35,
//! `dynamo-protocols` 8.2.0). The recorder is `data/recorder.rs`; it ran once per
//! `DYN_PARSER_VERSION` value (unset, `1`, `2`), because dynamo-llm reads the version
//! once per process.
//!
//! Inputs: every parser name the v1 and v2 registries know at dynamo's pins, Dynamo's
//! aliases, names that only exist on newer registries, and junk names, each also as
//! `None`. Pair rules run on every (tool-call parser, reasoning parser) pair. The route
//! runs on 60 request shapes (tool choice, top-level tools, system-message tools).
//!
//! The file is compact: line 1 is a table of distinct outputs, and later lines refer to
//! it by index. Parser names are indexes into the `names` line, where index 0 is
//! `None`.

use std::collections::{HashMap, HashSet};

use dynamo_chat_parsing::{
    ParserVersion, ToolProcessingRoute, families, prefill, route, special_tokens, stop_tokens,
    tools,
};
use dynamo_parsers::tool_calling::ToolChoice;
use dynamo_protocols::types::{ChatCompletionToolChoiceOption, CreateChatCompletionRequest};
use serde_json::{Value, json};

const DATA: &str = include_str!("data/dynamo_llm_parity.jsonl");

const VERSIONS: [ParserVersion; 3] = [ParserVersion::Auto, ParserVersion::V1, ParserVersion::V2];

fn version(label: &str) -> ParserVersion {
    match label {
        "Auto" => ParserVersion::Auto,
        "V1" => ParserVersion::V1,
        "V2" => ParserVersion::V2,
        other => panic!("unknown version label {other}"),
    }
}

/// The recorder's stand-in tokenizer: FNV-1a over the bytes, and two tokens for a
/// marker whose byte length is 2 mod 3.
fn fake_encode(s: &str) -> anyhow::Result<Vec<u32>> {
    let mut h: u32 = 0x811c9dc5;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    let h = h % 100_000 + 1_000;
    Ok(if s.len() % 3 == 2 {
        vec![h, h + 1]
    } else {
        vec![h]
    })
}

fn err_json(error: &anyhow::Error) -> Value {
    json!({"error": format!("{error:#}")})
}

fn route_json(route: Result<ToolProcessingRoute, dynamo_chat_parsing::InvalidRequest>) -> Value {
    match route {
        Ok(ToolProcessingRoute::Unified(family)) => json!({"unified": family}),
        Ok(ToolProcessingRoute::LegacyJail(parser)) => json!({"legacy_jail": parser}),
        Ok(ToolProcessingRoute::PassThrough) => json!("pass_through"),
        Err(error) => json!({"invalid_argument": error.0}),
    }
}

fn bit(c: char) -> bool {
    c == '1'
}

struct Data {
    values: Vec<Value>,
    prefills: Vec<Value>,
    names: Vec<Option<String>>,
    prompts: Vec<Option<String>>,
    rows: Vec<Value>,
}

impl Data {
    fn load() -> Self {
        let mut lines = DATA
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap());
        let mut head = |key: &str| {
            let line = lines.next().unwrap();
            assert_eq!(line[0], key);
            line[1].clone()
        };
        let values = head("values").as_array().unwrap().clone();
        let prefills = head("prefills").as_array().unwrap().clone();
        let names = serde_json::from_value(head("names")).unwrap();
        let prompts = serde_json::from_value(head("prompts")).unwrap();
        Self {
            values,
            prefills,
            names,
            prompts,
            rows: lines.collect(),
        }
    }

    fn rows<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
        self.rows.iter().filter(move |row| row[0] == kind)
    }

    fn name(&self, index: &Value) -> Option<&str> {
        self.names[index.as_u64().unwrap() as usize].as_deref()
    }

    fn value(&self, index: &Value) -> &Value {
        &self.values[index.as_u64().unwrap() as usize]
    }

    fn object<'a>(&'a self, kind: &'a str) -> Vec<&'a Value> {
        self.rows(kind).map(|row| &row[1]).collect()
    }
}

/// Names that select a unified family on this branch but did not in the recording,
/// because a newer `dynamo-parsers-v2` registered them. A case that names one of them
/// may differ from dynamo-llm; [`Tally`] counts it apart, and
/// `registry_drift_is_only_new_families` pins the list.
fn drift_names(data: &Data) -> HashSet<String> {
    data.rows("name")
        .filter(|row| row[2].is_null())
        .filter_map(|row| row[1].as_str())
        .filter(|name| families::canonical_family(name).is_some())
        .map(str::to_string)
        .collect()
}

/// The names that may count as registry drift for a rule run at `version`. At
/// [`ParserVersion::V1`] no rule reads the v2 registry, so none may.
fn registry_names(version: ParserVersion, names: [Option<&str>; 2]) -> [Option<&str>; 2] {
    if version == ParserVersion::V1 {
        [None, None]
    } else {
        names
    }
}

/// Counts cases and keeps the first mismatches, so one run shows every kind that fails.
struct Tally {
    drift: HashSet<String>,
    cases: usize,
    drifted: usize,
    mismatches: Vec<String>,
}

impl Tally {
    fn new(data: &Data) -> Self {
        Self {
            drift: drift_names(data),
            cases: 0,
            drifted: 0,
            mismatches: Vec::new(),
        }
    }

    /// Compare a rule that does not read the v2 family registry. Any difference fails.
    fn check(&mut self, what: impl FnOnce() -> String, expected: &Value, actual: &Value) {
        self.cases += 1;
        if expected != actual {
            self.mismatches
                .push(format!("{}: dynamo {expected} != moved {actual}", what()));
        }
    }

    /// Compare a rule that reads the v2 family registry. A difference for a parser
    /// name the newer registry added counts as drift, not as a mismatch.
    fn check_registry(
        &mut self,
        names: [Option<&str>; 2],
        what: impl FnOnce() -> String,
        expected: &Value,
        actual: &Value,
    ) {
        if expected != actual
            && names
                .iter()
                .flatten()
                .any(|name| self.drift.contains(*name))
        {
            self.cases += 1;
            self.drifted += 1;
            return;
        }
        self.check(what, expected, actual);
    }

    fn finish(self, label: &str) -> usize {
        assert!(self.cases > 0, "{label}: no cases replayed");
        if !self.mismatches.is_empty() {
            let shown: Vec<_> = self.mismatches.iter().take(20).collect();
            panic!(
                "{label}: {} of {} cases differ from dynamo-llm, first {}:\n{}",
                self.mismatches.len(),
                self.cases,
                shown.len(),
                shown
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
        eprintln!(
            "{label}: {} cases, {} differ only by registry drift ({:?})",
            self.cases, self.drifted, self.drift
        );
        self.cases
    }
}

#[test]
fn family_selection_and_special_tokens_match_dynamo() {
    let data = Data::load();
    let mut tally = Tally::new(&data);
    for row in data.rows("selected") {
        let version = version(row[1].as_str().unwrap());
        let tool = data.name(&row[2]);
        for (r, (family, special)) in row[3]
            .as_array()
            .unwrap()
            .iter()
            .zip(row[4].as_array().unwrap())
            .enumerate()
        {
            let reasoning = data.names[r].as_deref();
            let what = || format!("{version:?} ({tool:?}, {reasoning:?})");
            tally.check_registry(
                registry_names(version, [tool, reasoning]),
                || format!("selected_family {}", what()),
                data.value(family),
                &json!(families::selected_family(tool, reasoning, version)),
            );
            let moved =
                match special_tokens::parser_requires_special_tokens(tool, reasoning, version) {
                    Ok(value) => json!(value),
                    Err(error) => err_json(&error),
                };
            tally.check_registry(
                registry_names(version, [tool, reasoning]),
                || format!("parser_requires_special_tokens {}", what()),
                data.value(special),
                &moved,
            );
        }
    }
    tally.finish("family selection and special tokens");
}

#[test]
fn pair_rules_match_dynamo() {
    let data = Data::load();
    let mut tally = Tally::new(&data);
    for row in data.rows("pair") {
        let tool = data.name(&row[1]);
        for (r, reasoning) in data.names.iter().enumerate() {
            let reasoning = reasoning.as_deref();
            let pair = || format!("({tool:?}, {reasoning:?})");
            tally.check_registry(
                [tool, reasoning],
                || format!("configured_family {}", pair()),
                data.value(&row[2][r]),
                &json!(families::configured_family(tool, reasoning)),
            );
            let is_v2 = bit(row[3].as_str().unwrap().chars().nth(r).unwrap());
            tally.check_registry(
                [tool, reasoning],
                || format!("is_v2_configured_family {}", pair()),
                &json!(is_v2),
                &json!(families::is_v2_configured_family(tool, reasoning)),
            );
            let starts = row[4][r].as_str().unwrap();
            let prefills = row[5][r].as_str().unwrap();
            for (p, prompt) in data.prompts.iter().enumerate() {
                let prompt = prompt.as_deref();
                tally.check_registry(
                    [tool, reasoning],
                    || {
                        format!(
                            "prompt_injected_reasoning_start_for_parsers {} {prompt:?}",
                            pair()
                        )
                    },
                    &json!(bit(starts.chars().nth(p).unwrap())),
                    &json!(prefill::prompt_injected_reasoning_start_for_parsers(
                        tool, reasoning, prompt
                    )),
                );
                let code = prefills.chars().nth(p).unwrap().to_digit(36).unwrap() as usize;
                let moved = prefill::prompt_reasoning_prefill_for_parsers(tool, reasoning, prompt);
                tally.check_registry(
                    [tool, reasoning],
                    || format!("prompt_reasoning_prefill_for_parsers {} {prompt:?}", pair()),
                    &data.prefills[code],
                    &json!([moved.legacy(), moved.unified(), moved.prefix()]),
                );
            }
            let modes = row[6][r].as_array().unwrap();
            for (i, version) in VERSIONS.iter().enumerate() {
                for (j, requires_v1_jail) in [false, true].into_iter().enumerate() {
                    let moved = match route::validate_tool_request_mode(
                        *version,
                        tool,
                        reasoning,
                        requires_v1_jail,
                    ) {
                        Ok(()) => Value::Null,
                        Err(error) => err_json(&error),
                    };
                    tally.check_registry(
                        registry_names(*version, [tool, reasoning]),
                        || {
                            format!(
                                "validate_tool_request_mode {version:?} {} {requires_v1_jail}",
                                pair()
                            )
                        },
                        data.value(&modes[i * 2 + j]),
                        &moved,
                    );
                }
                // `validate_parser_version` is the same check with no jail requirement.
                let moved = match route::validate_parser_version(*version, tool, reasoning) {
                    Ok(()) => Value::Null,
                    Err(error) => err_json(&error),
                };
                tally.check_registry(
                    registry_names(*version, [tool, reasoning]),
                    || format!("validate_parser_version {version:?} {}", pair()),
                    data.value(&modes[i * 2]),
                    &moved,
                );
            }
        }
    }
    tally.finish("pair rules");
}

fn route_requests(data: &Data) -> Vec<CreateChatCompletionRequest> {
    let requests = data.object("route_requests")[0]["requests"]
        .as_array()
        .unwrap();
    requests
        .iter()
        .map(|request| serde_json::from_value(request.clone()).unwrap())
        .collect()
}

#[test]
fn route_matches_dynamo() {
    let data = Data::load();
    let requests = route_requests(&data);
    assert_eq!(requests.len(), 60);
    let mut tally = Tally::new(&data);
    for row in data.rows("route") {
        let version = version(row[1].as_str().unwrap());
        let (tool, reasoning) = (data.name(&row[2]), data.name(&row[3]));
        for (i, (expected, request)) in row[4].as_array().unwrap().iter().zip(&requests).enumerate()
        {
            tally.check_registry(
                registry_names(version, [tool, reasoning]),
                || {
                    format!(
                        "tool_processing_route {version:?} ({tool:?}, {reasoning:?}) request {i}"
                    )
                },
                data.value(expected),
                &route_json(route::tool_processing_route(
                    tool, reasoning, request, version,
                )),
            );
        }
    }
    tally.finish("route");
}

#[test]
fn prompt_opener_matches_dynamo() {
    let data = Data::load();
    let mut tally = Tally::new(&data);
    for row in data.rows("prompt_start") {
        let reasoning = data.name(&row[1]);
        for (p, expected) in row[2].as_str().unwrap().chars().enumerate() {
            let prompt = data.prompts[p].as_deref();
            tally.check(
                || format!("prompt_injected_reasoning_start {reasoning:?} {prompt:?}"),
                &json!(bit(expected)),
                &json!(prefill::prompt_injected_reasoning_start(reasoning, prompt)),
            );
        }
    }
    tally.finish("prompt opener");
}

#[test]
fn tool_parser_rules_match_dynamo() {
    let data = Data::load();
    let requests = route_requests(&data);
    let choices = [
        ToolChoice::None,
        ToolChoice::Auto,
        ToolChoice::Required,
        ToolChoice::Named("lookup".to_string()),
    ];
    let options: Vec<Option<ChatCompletionToolChoiceOption>> = [
        Value::Null,
        json!("none"),
        json!("auto"),
        json!("required"),
        json!({"type": "function", "function": {"name": "lookup"}}),
    ]
    .into_iter()
    .map(|value| serde_json::from_value(value).unwrap())
    .collect();
    let mut tally = Tally::new(&data);
    for row in data.rows("tool_parser") {
        let tool = data.name(&row[1]);
        for (choice, expected) in choices.iter().zip(row[2].as_str().unwrap().chars()) {
            tally.check(
                || format!("requires_native_tool_call_format {tool:?} {choice:?}"),
                &json!(bit(expected)),
                &json!(tools::requires_native_tool_call_format(tool, choice)),
            );
        }
        let owned = tool.map(str::to_string);
        let jail = row[3].as_array().unwrap();
        for (i, option) in options.iter().enumerate() {
            for (j, has_tools) in [false, true].into_iter().enumerate() {
                tally.check(
                    || format!("should_apply_tool_jail {tool:?} {option:?} {has_tools}"),
                    data.value(&jail[i * 2 + j]),
                    &json!(route::should_apply_tool_jail(
                        owned.as_ref(),
                        option.as_ref(),
                        has_tools
                    )),
                );
            }
        }
        for hidden in row[4].as_array().unwrap() {
            let input: Vec<u32> = serde_json::from_value(hidden[0].clone()).unwrap();
            for (i, (expected, request)) in hidden[1]
                .as_array()
                .unwrap()
                .iter()
                .zip(&requests)
                .enumerate()
            {
                let mut hidden_ids = input.clone();
                let moved = match stop_tokens::remove_tool_parser_end_tokens_from_hidden_stops(
                    tool,
                    request,
                    &mut hidden_ids,
                    fake_encode,
                ) {
                    Ok(visible) => json!([visible, hidden_ids]),
                    Err(error) => err_json(&error),
                };
                tally.check(
                    || format!("remove_tool_parser_end_tokens_from_hidden_stops {tool:?} {input:?} request {i}"),
                    data.value(expected),
                    &moved,
                );
            }
        }
    }
    tally.finish("tool parser rules");
}

#[test]
fn effective_tools_match_dynamo() {
    let data = Data::load();
    let mut tally = Tally::new(&data);
    for row in data.object("tools") {
        let request: CreateChatCompletionRequest =
            serde_json::from_value(row["request"].clone()).unwrap();
        let what = || row["request"].to_string();
        tally.check(
            || format!("tool_call_parsing_enabled {}", what()),
            &row["parsing_enabled"],
            &json!(tools::tool_call_parsing_enabled(&request)),
        );
        tally.check(
            || format!("request_has_effective_tools {}", what()),
            &row["request_has_effective_tools"],
            &json!(stop_tokens::request_has_effective_tools(&request)),
        );
        // Dynamo runs its own tool validation after the merge. Every recorded
        // validation failure is a tool name, so this stand-in checks only names.
        let validate = |tools: &[dynamo_protocols::types::ChatCompletionTool]| {
            for (i, tool) in tools.iter().enumerate() {
                let name = &tool.function.name;
                if !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
                {
                    anyhow::bail!(
                        "Function at index {i} has an invalid name: \"{name}\". \
                         Only a-z, A-Z, 0-9, underscores, dashes, and dots are allowed."
                    );
                }
            }
            Ok(())
        };
        let effective = tools::effective_tools(&request, validate);
        let moved = match &effective {
            Ok(tools) => json!({"ok": serde_json::to_value(tools.as_ref()).unwrap()}),
            Err(error) => json!({"err": error.to_string()}),
        };
        tally.check(
            || format!("effective_tools {}", what()),
            &row["effective"],
            &moved,
        );
        let definitions = match &effective {
            Ok(list) => {
                json!({"ok": tools::convert_tools(list).iter().map(|t| json!({"name": t.name, "parameters": t.parameters, "strict": t.strict})).collect::<Vec<_>>()})
            }
            Err(error) => json!({"err": error.to_string()}),
        };
        tally.check(
            || format!("convert_tools(effective_tools) {}", what()),
            &row["definitions"],
            &definitions,
        );
    }
    tally.finish("effective tools");
}

#[test]
fn small_tables_match_dynamo() {
    let data = Data::load();
    let mut tally = Tally::new(&data);
    let names: Vec<&str> = families::unified_family_names().to_vec();
    // This branch's list is the recorded one plus the drift names.
    let mut expected: Vec<String> =
        serde_json::from_value(data.object("family_names")[0]["names"].clone()).unwrap();
    expected.extend(tally.drift.iter().cloned());
    expected.sort_unstable();
    tally.check_registry(
        [None, None],
        || "unified_family_names".to_string(),
        &json!(expected),
        &json!(names),
    );
    for row in data.object("version_parse") {
        let value = row["value"].as_str();
        let moved = match ParserVersion::try_from(value) {
            Ok(version) => json!(format!("{version:?}")),
            Err(error) => err_json(&error),
        };
        tally.check(
            || format!("ParserVersion::try_from({value:?})"),
            &row["out"],
            &moved,
        );
    }
    for row in data.rows("name") {
        let name = row[1].as_str().unwrap();
        tally.check_registry(
            [Some(name), None],
            || format!("canonical_family({name:?})"),
            &row[2],
            &json!(families::canonical_family(name)),
        );
    }
    for row in data.object("stripped") {
        let skip: Option<bool> = serde_json::from_value(row["skip"].clone()).unwrap();
        let requires = row["requires"].as_bool().unwrap();
        tally.check(
            || format!("special_tokens_will_be_stripped({skip:?}, {requires})"),
            &row["out"],
            &json!(special_tokens::special_tokens_will_be_stripped(
                skip, requires
            )),
        );
    }
    tally.finish("small tables");
}

/// The data file must cover every name the moved code knows on this branch, so a
/// family added to a registry here cannot be missed by the replay.
#[test]
fn recorded_names_cover_this_branch() {
    let data = Data::load();
    let recorded: HashMap<&str, ()> = data
        .names
        .iter()
        .flatten()
        .map(|name| (name.as_str(), ()))
        .collect();
    let mut missing: Vec<&str> = dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES
        .iter()
        .copied()
        .chain(families::unified_family_names().iter().copied())
        .chain(dynamo_parsers::tool_calling::parsers::get_available_tool_parsers())
        .chain(dynamo_parsers::reasoning::get_available_reasoning_parsers())
        .filter(|name| !recorded.contains_key(name))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    assert!(
        missing.is_empty(),
        "names not in the recording: {missing:?}"
    );
}

/// On this branch the only difference from the recording is `dynamo-parsers-v2`
/// registering Muse Spark. A new drift name means a registry change this crate has not
/// been checked against: record again with a dynamo that pins it.
#[test]
fn registry_drift_is_only_new_families() {
    let data = Data::load();
    let mut drift: Vec<String> = drift_names(&data).into_iter().collect();
    drift.sort_unstable();
    let expected: &[&str] = if dynamo_parsers_v2::canonical_unified_family("muse_spark").is_some() {
        &["muse_spark", "muse_spark_open"]
    } else {
        &[]
    };
    assert_eq!(drift, expected);
}

#[test]
fn tally_records_native_format_mismatches_for_existing_families() {
    let mut tally = Tally::new(&Data::load());
    tally.check(
        || "requires_native_tool_call_format kimi_k3 Required".to_string(),
        &json!(true),
        &json!(false),
    );
    assert_eq!(tally.mismatches.len(), 1);
    assert_eq!(tally.drifted, 0);
}

/// A name the newer registry added must not hide a difference in
/// a rule that never reads the registry.
#[test]
fn registry_drift_does_not_hide_native_format_mismatches() {
    let data = Data::load();
    let mut tally = Tally::new(&data);
    // Treat it as drift even where the registry does not know it yet.
    tally.drift.insert("muse_spark".to_string());
    let recorded = data
        .rows("tool_parser")
        .find(|row| data.name(&row[1]) == Some("muse_spark"))
        .unwrap();
    assert_eq!(recorded[2], "0000");
    assert!(!tools::requires_native_tool_call_format(
        Some("muse_spark"),
        &ToolChoice::Required,
    ));

    // Registration cannot change the Kimi-only native-format rule.
    tally.check(
        || "requires_native_tool_call_format muse_spark Required".to_string(),
        &json!(false),
        &json!(true),
    );
    assert_eq!(
        tally.mismatches.len(),
        1,
        "an unrelated native-format regression must fail the parity replay"
    );
}

/// A v2 registry addition cannot change v1 availability, so V1
/// rules get no drift exemption.
#[test]
fn v2_registry_drift_cannot_exempt_v1_validation() {
    let data = Data::load();
    let mut tally = Tally::new(&data);
    // Keep the check meaningful even when replaying with Dynamo's exact pins.
    tally.drift.insert("muse_spark".to_string());
    let reasoning = data
        .names
        .iter()
        .position(|name| name.as_deref() == Some("muse_spark"))
        .unwrap();
    let row = data
        .rows("pair")
        .find(|row| data.name(&row[1]).is_none())
        .unwrap();
    // Mode index 2 records V1 with no jail requirement.
    let expected = data.value(&row[6][reasoning][2]);
    assert!(
        expected["error"]
            .as_str()
            .unwrap()
            .contains("no compatible v1 parser")
    );
    assert!(route::validate_parser_version(ParserVersion::V1, None, Some("muse_spark")).is_err());

    // Replay the same comparison as pair_rules_match_dynamo, with a regression
    // that accepts this v2-only name as a v1 parser.
    tally.check_registry(
        registry_names(ParserVersion::V1, [None, Some("muse_spark")]),
        || "validate_parser_version V1 (None, Some(muse_spark))".to_string(),
        expected,
        &Value::Null,
    );
    assert_eq!(
        tally.mismatches.len(),
        1,
        "registering a v2 family cannot make an unavailable v1 parser valid"
    );
    assert_eq!(tally.drifted, 0);
}
