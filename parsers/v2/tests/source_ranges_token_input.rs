// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::single_range_in_vec_init)]

use dynamo_parsers_v2::unified::harmony::encode_harmony;
use dynamo_parsers_v2::{
    UnifiedParser, UnifiedParserEvent, UnifiedParserInit, UnifiedParserOutput,
    UnifiedParserStartingState, create_unified_parser_for_family,
};

fn parser(record: bool) -> Box<dyn UnifiedParser> {
    let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Response,
            ..UnifiedParserInit::default()
        })
        .unwrap();
    if record {
        assert!(parser.record_sources());
    }
    parser
}

#[test]
fn recording_rejects_token_input_before_committing_unmapped_text() {
    let tokens = encode_harmony("A").unwrap();
    let expected = vec![UnifiedParserEvent::Text("A".into())];

    let mut stock = parser(false);
    let mut output = UnifiedParserOutput::default();
    stock.parse_token_ids_into(&tokens, &mut output).unwrap();
    assert_eq!(output.events, expected);
    let mut ranges = Vec::new();
    stock.take_sources(&mut ranges);
    assert!(ranges.is_empty());

    let mut text = parser(true);
    let mut output = UnifiedParserOutput::default();
    text.parse_into("A", &mut output).unwrap();
    assert_eq!(output.events, expected);
    text.take_sources(&mut ranges);
    assert_eq!(ranges, [0..1]);

    // Recording dates only bytes fed through parse_into. Token input must fail
    // before it commits text that has no byte in that coordinate system.
    let mut recording = parser(true);
    let mut output = UnifiedParserOutput::default();
    let result = recording.parse_token_ids_into(&tokens, &mut output);
    ranges.clear();
    recording.take_sources(&mut ranges);
    assert!(
        result.is_err(),
        "accepted token input while recording: result={result:?}, events={:?}, ranges={ranges:?}",
        output.events
    );
    assert!(output.events.is_empty());
    assert!(ranges.is_empty());
}
