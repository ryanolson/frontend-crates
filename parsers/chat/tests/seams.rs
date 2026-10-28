// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The parts of the move that are not a copy: functions that take a value dynamo-llm
//! read from its own state, and the call-site code that became a function.

use dynamo_chat_parsing::{
    ParserVersion,
    prefill::{PromptReasoningPrefill, prompt_reasoning_prefill_for_parsers},
    special_tokens::resolve_skip_special_tokens,
    stop_tokens::{
        move_tool_parser_end_tokens_to_visible_stops,
        remove_tool_parser_end_tokens_from_hidden_stops,
    },
};
use dynamo_protocols::types::CreateChatCompletionRequest;
use serde_json::json;

const CALL: u32 = 42;

fn harmony_encode(input: &str) -> anyhow::Result<Vec<u32>> {
    assert_eq!(input, "<|call|>");
    Ok(vec![CALL])
}

fn with_tools(tool_choice: Option<&str>) -> CreateChatCompletionRequest {
    let mut value = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}]
    });
    if let Some(choice) = tool_choice {
        value["tool_choice"] = json!(choice);
    }
    serde_json::from_value(value).unwrap()
}

/// dynamo-llm's stop-condition code (`preprocessor.rs:3249-3270`), now one function.
#[test]
fn end_token_moves_from_hidden_to_visible_stops() {
    let request = with_tools(None);
    let mut hidden = Some(vec![CALL, 7]);
    let mut visible = None;
    move_tool_parser_end_tokens_to_visible_stops(
        Some("harmony"),
        &request,
        &mut hidden,
        &mut visible,
        harmony_encode,
    )
    .unwrap();
    assert_eq!(hidden, Some(vec![7]));
    assert_eq!(visible, Some(vec![CALL]));

    // A token already visible is not added twice.
    let mut hidden = Some(vec![CALL]);
    let mut visible = Some(vec![CALL, 9]);
    move_tool_parser_end_tokens_to_visible_stops(
        Some("harmony"),
        &request,
        &mut hidden,
        &mut visible,
        harmony_encode,
    )
    .unwrap();
    assert_eq!(hidden, Some(vec![]));
    assert_eq!(visible, Some(vec![CALL, 9]));
}

#[test]
fn nothing_moves_without_hidden_stops_or_tool_permission() {
    // No hidden list: neither list is created.
    let mut hidden = None;
    let mut visible = None;
    move_tool_parser_end_tokens_to_visible_stops(
        Some("harmony"),
        &with_tools(None),
        &mut hidden,
        &mut visible,
        harmony_encode,
    )
    .unwrap();
    assert_eq!((hidden, visible), (None, None));

    // `tool_choice: none` keeps the token hidden, and the visible list is not created.
    let mut hidden = Some(vec![CALL]);
    let mut visible = None;
    move_tool_parser_end_tokens_to_visible_stops(
        Some("harmony"),
        &with_tools(Some("none")),
        &mut hidden,
        &mut visible,
        |_: &str| -> anyhow::Result<Vec<u32>> { panic!("no encode without tool permission") },
    )
    .unwrap();
    assert_eq!((hidden, visible), (Some(vec![CALL]), None));
}

/// dynamo-llm's tokenizer call became a closure; its error keeps dynamo's context.
#[test]
fn encode_error_names_the_marker_and_parser() {
    let mut hidden = vec![CALL];
    let error = remove_tool_parser_end_tokens_from_hidden_stops(
        Some("harmony"),
        &with_tools(None),
        &mut hidden,
        |_: &str| -> anyhow::Result<Vec<u32>> { anyhow::bail!("tokenizer down") },
    )
    .unwrap_err();
    assert_eq!(
        format!("{error:#}"),
        "Failed to encode tool-call end token \"<|call|>\" for parser \"harmony\": tokenizer down"
    );
    assert_eq!(hidden, [CALL]);
}

/// A marker that encodes to two tokens cannot be a stop, so it stays where it is.
#[test]
fn multi_token_marker_is_not_moved() {
    let mut hidden = vec![CALL, CALL + 1];
    let visible = remove_tool_parser_end_tokens_from_hidden_stops(
        Some("harmony"),
        &with_tools(None),
        &mut hidden,
        |_: &str| Ok(vec![CALL, CALL + 1]),
    )
    .unwrap();
    assert!(visible.is_empty());
    assert_eq!(hidden, [CALL, CALL + 1]);
}

/// dynamo-llm's `skip_special_tokens` call site (`preprocessor.rs:3296-3324`).
#[test]
fn skip_special_tokens_default_flips_only_when_unset() {
    for (requested, requires, expected) in [
        (None, false, None),
        (None, true, Some(false)),
        (Some(false), false, Some(false)),
        (Some(false), true, Some(false)),
        (Some(true), false, Some(true)),
        // The client's explicit value wins; dynamo only warns.
        (Some(true), true, Some(true)),
    ] {
        assert_eq!(
            resolve_skip_special_tokens(requested, requires, Some("harmony"), Some("gpt_oss")),
            expected,
            "{requested:?} {requires}"
        );
    }
}

#[test]
fn prefill_accessors_and_explicit_bool() {
    let explicit = PromptReasoningPrefill::from(true);
    assert!(explicit.legacy() && explicit.unified());
    assert_eq!(explicit.prefix(), None);

    let gemma = prompt_reasoning_prefill_for_parsers(
        Some("gemma4"),
        Some("gemma4"),
        Some("x<|channel>thou"),
    );
    assert_eq!(gemma.prefix(), Some("<|channel>thou"));
    assert!(gemma.unified());
    assert!(!gemma.legacy());
}

/// The version is an argument; this crate reads no environment, so one process can
/// serve every version.
#[test]
fn every_version_in_one_process() {
    use dynamo_chat_parsing::families::selected_family;
    let pair = (Some("harmony"), Some("gpt_oss"));
    let routes: Vec<_> = [ParserVersion::Auto, ParserVersion::V1, ParserVersion::V2]
        .into_iter()
        .map(|version| selected_family(pair.0, pair.1, version))
        .collect();
    assert_eq!(routes, [Some("gpt_oss"), None, Some("gpt_oss")]);
}
