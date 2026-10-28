// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

#[test]
fn a_channel_opener_in_the_default_state_counts_generated_bytes_only() {
    let mut parser = replay(UnifiedParserStartingState::None);
    assert!(parser.record_sources());
    assert_eq!(
        answer(&mut *parser),
        [UnifiedParserEvent::Reasoning("A".into())]
    );
    let mut ranges = Vec::new();
    parser.take_sources(&mut ranges);
    assert_eq!(ranges, [0..1]);
}

#[test]
#[ignore = "F2: a held prompt opener makes source_at subtract uncounted bytes"]
fn a_channel_opener_in_the_reasoning_state_counts_generated_bytes_only() {
    let mut parser = replay(UnifiedParserStartingState::Reasoning);
    assert!(parser.record_sources());
    assert_eq!(
        answer(&mut *parser),
        [UnifiedParserEvent::Reasoning("A".into())]
    );
    let mut ranges = Vec::new();
    parser.take_sources(&mut ranges);
    assert_eq!(ranges, [0..1]);
}

#[test]
#[ignore = "F2: reset subtracts the held prompt opener from zero recorded bytes"]
fn reset_after_accepting_a_structural_prompt_opener_does_not_panic() {
    let mut parser = replay(UnifiedParserStartingState::None);
    assert!(parser.record_sources());
    assert_eq!(parser.reset(), "<|channel>");
}
