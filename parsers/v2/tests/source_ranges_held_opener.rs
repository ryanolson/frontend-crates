// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A complete Gemma 4 `<|channel>` held alone still holds input. Inside a quote
//! it surfaces as text when the quote closes, and `reset` returns it. Its bytes
//! were fed before recording, so the source offsets cannot count them, and
//! recording must refuse.

use dynamo_parsers_v2::{
    UnifiedParser, UnifiedParserEvent, UnifiedParserInit, UnifiedParserOutput,
    UnifiedParserStartingState, create_unified_parser_for_family,
};

fn replay(state: UnifiedParserStartingState) -> Box<dyn UnifiedParser> {
    let mut parser = create_unified_parser_for_family("gemma4", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: state,
            ..UnifiedParserInit::default()
        })
        .unwrap();
    let mut output = UnifiedParserOutput::default();
    parser.parse_into("<|channel>", &mut output).unwrap();
    assert!(output.events.is_empty());
    parser
}

fn answer(parser: &mut dyn UnifiedParser) -> Vec<UnifiedParserEvent> {
    let mut output = UnifiedParserOutput::default();
    parser.parse_into("A", &mut output).unwrap();
    output.append(&mut parser.finish().unwrap());
    output.events
}

#[test]
fn a_redundant_channel_opener_is_structural_without_recording() {
    let mut parser = replay(UnifiedParserStartingState::Reasoning);
    assert_eq!(
        answer(&mut *parser),
        [UnifiedParserEvent::Reasoning("A".into())]
    );
    let mut parser = replay(UnifiedParserStartingState::None);
    assert_eq!(parser.reset(), "<|channel>");
}

/// A held `<|channel>` was fed before recording, so the source offsets cannot
/// count it, and `reset` returns it as text. Recording refuses in both states.
#[test]
fn recording_refuses_a_held_channel_opener() {
    for state in [
        UnifiedParserStartingState::None,
        UnifiedParserStartingState::Reasoning,
    ] {
        let mut parser = replay(state);
        assert!(!parser.record_sources(), "{state:?}");
        let mut ranges = Vec::new();
        parser.take_sources(&mut ranges);
        assert!(ranges.is_empty());
    }
}

/// A refused recording leaves the stock behavior: no panic in the source math,
/// and `reset` returns the opener.
#[test]
fn a_refused_recording_keeps_the_stock_output() {
    let mut parser = replay(UnifiedParserStartingState::Reasoning);
    assert!(!parser.record_sources());
    assert_eq!(
        answer(&mut *parser),
        [UnifiedParserEvent::Reasoning("A".into())]
    );
    let mut parser = replay(UnifiedParserStartingState::None);
    assert!(!parser.record_sources());
    assert_eq!(parser.reset(), "<|channel>");
}

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
