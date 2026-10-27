// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A number outside the `f64` range (`1e400`) where a parser checks
//! arguments by reading them as a `serde_json::Value`. With the
//! `arbitrary-precision` feature it stays verbatim. Without the feature,
//! these checks behave as in 0.7.x: a Kimi K3 typed argument becomes a
//! string, its JSON-object form is dropped, and a required guided call is
//! rejected (`finish` returns an error under the default policy).

use dynamo_parsers_v2::{
    UnifiedParserExt, UnifiedParserInit, UnifiedToolOutputMode, create_unified_parser_for_family,
    tool_arguments_raw,
};

/// The Kimi K3 arguments of a typed argument and of the JSON-object form.
fn kimi_k3_arguments() -> Vec<Option<String>> {
    [
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
    .collect()
}

/// A required guided call with `arguments`: its raw arguments, or the
/// error `finish` returned.
fn guided_required(arguments: &str) -> Result<Option<String>, String> {
    let mut parser = create_unified_parser_for_family("qwen3", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ..UnifiedParserInit::default()
        })
        .unwrap();
    let mut events = parser
        .push(&format!(r#"[{{"name":"inspect","arguments":{arguments}}}]"#))
        .unwrap();
    let finished = parser.finish().map_err(|error| error.to_string())?;
    events.extend(finished.events);
    Ok(tool_arguments_raw(&events).remove(&0))
}

#[test]
fn guided_required_keeps_ordinary_numbers() {
    assert_eq!(
        guided_required(r#"{"n":1.50}"#),
        Ok(Some(r#"{"n":1.50}"#.into()))
    );
}

#[cfg(feature = "arbitrary-precision")]
#[test]
fn kimi_k3_keeps_large_number_tokens_in_native_arguments() {
    assert_eq!(kimi_k3_arguments(), vec![Some(r#"{"n":1e400}"#.into()); 2]);
}

#[cfg(feature = "arbitrary-precision")]
#[test]
fn guided_required_keeps_large_number_tokens_in_arguments() {
    assert_eq!(
        guided_required(r#"{"n":1e400}"#),
        Ok(Some(r#"{"n":1e400}"#.into()))
    );
}

#[cfg(not(feature = "arbitrary-precision"))]
#[test]
fn kimi_k3_without_the_feature_quotes_or_drops_a_large_number() {
    assert_eq!(
        kimi_k3_arguments(),
        vec![Some(r#"{"n":"1e400"}"#.into()), None]
    );
}

#[cfg(not(feature = "arbitrary-precision"))]
#[test]
fn guided_required_without_the_feature_rejects_a_large_number() {
    let result = guided_required(r#"{"n":1e400}"#);
    assert!(result.is_err(), "{result:?}");
}
