// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! With the `arbitrary-precision` feature, a number outside the `f64` range
//! (`1e400`) stays verbatim where a parser checks arguments by reading them
//! as a `serde_json::Value`. Without the feature, those checks behave as in
//! 0.7.x: the Kimi K3 typed argument becomes a string, and its JSON-object
//! form and a required guided call are dropped.
#![cfg(feature = "arbitrary-precision")]

use dynamo_parsers_v2::{
    UnifiedParserExt, UnifiedParserInit, UnifiedToolOutputMode, create_unified_parser_for_family,
    tool_arguments_raw,
};

#[test]
fn kimi_k3_keeps_large_number_tokens_in_native_arguments() {
    let actual: Vec<_> = [
        "<|open|>argument key=\"n\" type=\"number\"<|sep|>1e400<|close|>argument<|sep|>",
        "<|open|>json type=\"object\"<|sep|>{\"n\":1e400}<|close|>json<|sep|>",
    ]
    .into_iter()
    .map(|body| {
        let input = format!(
            "<|open|>tools<|sep|><|open|>call tool=\"inspect\" index=\"1\"<|sep|>{body}<|close|>call<|sep|><|close|>tools<|sep|>"
        );
        let mut parser = create_unified_parser_for_family("kimi_k3", &[]).unwrap();
        let mut events = parser.push(&input).unwrap();
        events.extend(parser.finish().unwrap().events);
        tool_arguments_raw(&events).remove(&0)
    })
    .collect();
    assert_eq!(actual, vec![Some(r#"{"n":1e400}"#.into()); 2]);
}

#[test]
fn guided_required_keeps_large_number_tokens_in_arguments() {
    let mut parser = create_unified_parser_for_family("qwen3", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ..UnifiedParserInit::default()
        })
        .unwrap();
    let mut events = parser
        .push(r#"[{"name":"inspect","arguments":{"n":1e400}}]"#)
        .unwrap();
    events.extend(parser.finish().unwrap().events);
    let raw = tool_arguments_raw(&events);
    assert_eq!(raw.get(&0).map(String::as_str), Some(r#"{"n":1e400}"#));
}
