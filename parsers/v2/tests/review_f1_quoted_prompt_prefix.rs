// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    UnifiedParserEvent, UnifiedParserInit, UnifiedParserOutput, create_unified_parser_for_family,
};

#[test]
fn closing_a_quote_releases_the_held_channel_opener_as_text() {
    let mut parser = create_unified_parser_for_family("gemma4", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit::default())
        .unwrap();
    let mut output = UnifiedParserOutput::default();
    parser.parse_into("\"<|channel>", &mut output).unwrap();
    assert_eq!(output.events, [UnifiedParserEvent::Text("\"".into())]);
    output.events.clear();
    parser.parse_into("\"", &mut output).unwrap();
    output.append(&mut parser.finish().unwrap());
    assert_eq!(
        output.events,
        [UnifiedParserEvent::Text("<|channel>\"".into())]
    );
}

#[test]
#[ignore = "F1: a complete channel opener held inside a prose quote can surface as text"]
fn recording_refuses_a_channel_opener_held_inside_a_quote() {
    let mut parser = create_unified_parser_for_family("gemma4", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit::default())
        .unwrap();
    parser
        .parse_into("\"<|channel>", &mut UnifiedParserOutput::default())
        .unwrap();
    assert!(
        !parser.record_sources(),
        "the held <|channel> becomes text when the next byte closes the quote"
    );
}
