// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact source ranges on the shared scanner, for the cases where byte
//! equality cannot tell two occurrences apart.
//!
//! The conformance sweep (`unified_source_ranges.rs`) checks that ranges rebuild
//! the committed text at every split. These cases also name the exact offsets:
//! a held `<` that a later marker deletes, `reset` text from disjoint tool
//! spans, repeated words, and offsets across `reset` and a new request.

#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;

use dynamo_parsers_v2::{
    UnifiedParser, UnifiedParserEvent, UnifiedParserInit, UnifiedParserOutput,
    UnifiedParserStartingState, create_unified_parser_for_family,
};

fn recording(family: &str, init: UnifiedParserInit) -> Box<dyn UnifiedParser> {
    let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
    parser.initialize_request(init).unwrap();
    assert!(parser.record_sources(), "{family} refused to record");
    parser
}

fn parse(parser: &mut Box<dyn UnifiedParser>, chunk: &str) -> Vec<UnifiedParserEvent> {
    let mut output = UnifiedParserOutput::default();
    parser.parse_into(chunk, &mut output).unwrap();
    output.events
}

fn taken(parser: &mut Box<dyn UnifiedParser>) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    parser.take_sources(&mut ranges);
    ranges
}

fn text(value: &str) -> UnifiedParserEvent {
    UnifiedParserEvent::Text(value.into())
}

fn reasoning(value: &str) -> UnifiedParserEvent {
    UnifiedParserEvent::Reasoning(value.into())
}

/// GLM holds a lone `<`, then a closer deletes markup around it. A wrapper that
/// matches the emitted `<` against the input guesses the wrong one in each case;
/// the parser knows which byte it kept.
#[test]
fn a_held_byte_keeps_its_own_offset_when_a_marker_follows() {
    for (second, expected) in [("</think>", 0..1), ("/think><<", 8..9)] {
        let mut parser = recording("glm47", UnifiedParserInit::default());
        assert!(parse(&mut parser, "<").is_empty());
        assert!(taken(&mut parser).is_empty());
        assert_eq!(parse(&mut parser, second), [text("<")], "{second:?}");
        assert_eq!(taken(&mut parser), [expected], "{second:?}");
    }
}

const BAD: &str = concat!(
    "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">",
    "<｜DSML｜ parameter name=\"value\" string=\"false\">{\"a\":",
    "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
);

/// DeepSeek V4.1 defers a parse error to EOF and keeps the failed block for
/// `reset`. Two failed blocks around committed text come back joined, so their
/// ranges are disjoint.
#[test]
fn reset_text_from_disjoint_tool_spans_keeps_both_ranges() {
    let input = format!("{BAD}GAP{BAD}");
    let gap = BAD.len()..BAD.len() + 3;
    let mut parser = recording("deepseek_v41", UnifiedParserInit::default());
    assert_eq!(parse(&mut parser, &input), [text("GAP")]);
    assert_eq!(taken(&mut parser), [gap.start..gap.end]);
    assert!(parser.finish().is_err());
    assert!(
        taken(&mut parser).is_empty(),
        "a failed finish returns no events, so it has no ranges"
    );
    assert_eq!(parser.reset(), format!("{BAD}{BAD}"));
    assert_eq!(taken(&mut parser), [0..gap.start, gap.end..input.len()]);
}

#[test]
fn one_deferred_block_has_contiguous_reset_input() {
    let mut parser = recording("deepseek_v41", UnifiedParserInit::default());
    assert!(parse(&mut parser, BAD).is_empty());
    assert!(parser.finish().is_err());
    assert_eq!(parser.reset(), BAD);
    assert_eq!(taken(&mut parser), [0..BAD.len()]);
}

/// The range of the `len` bytes that start right after the `nth` (from 0)
/// occurrence of `marker` in `input`.
fn after(input: &str, marker: &str, nth: usize, len: usize) -> Range<usize> {
    let start = input.match_indices(marker).nth(nth).unwrap().0 + marker.len();
    start..start + len
}

/// Repeated words in reasoning and in the answer. Each run maps to its own
/// occurrence, at every two-way split.
#[test]
fn repeated_text_maps_to_its_own_occurrence() {
    let reasoning_first = "<think>ha ha</think>ha ha";
    for family in ["glm47", "deepseek_v4", "qwen3", "kimi_k2"] {
        let input = reasoning_first;
        for split in 0..=input.len() {
            let mut parser = recording(family, UnifiedParserInit::default());
            let mut events = parse(&mut parser, &input[..split]);
            let mut ranges = taken(&mut parser);
            events.extend(parse(&mut parser, &input[split..]));
            events.extend(parser.finish().unwrap().events);
            ranges.extend(taken(&mut parser));
            let joined: String = events
                .iter()
                .map(|event| match event {
                    UnifiedParserEvent::Text(t) | UnifiedParserEvent::Reasoning(t) => t.as_str(),
                    _ => "",
                })
                .collect();
            assert_eq!(joined, "ha haha ha", "{family} split {split}: {events:?}");
            let mut merged: Vec<Range<usize>> = Vec::new();
            for range in ranges {
                match merged.last_mut() {
                    Some(last) if last.end == range.start => last.end = range.end,
                    _ => merged.push(range),
                }
            }
            assert_eq!(
                merged,
                [
                    after(input, "<think>", 0, 5),
                    after(input, "</think>", 0, 5)
                ],
                "{family} split {split}"
            );
        }
    }
}

/// Text before a tool block, and text between two blocks that repeat each other.
#[test]
fn text_around_repeated_tool_blocks_maps_to_its_own_bytes() {
    let call = "<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>";
    let input = format!("ab{call}ab{call}ab");
    let mut parser = recording("glm47", UnifiedParserInit::default());
    parse(&mut parser, &input);
    parser.finish().unwrap();
    let second = 2 + call.len();
    let third = second + 2 + call.len();
    assert_eq!(
        taken(&mut parser),
        [0..2, second..second + 2, third..third + 2]
    );
}

/// The offset counts every byte this instance received, across `reset` and a
/// new request, and ranges not yet taken survive both.
#[test]
fn offsets_and_untaken_ranges_survive_reset_and_a_new_request() {
    for reinitialize in [false, true] {
        let mut parser = recording("deepseek_v4", UnifiedParserInit::default());
        assert_eq!(parse(&mut parser, "A"), [text("A")]);
        assert_eq!(parser.reset(), "");
        if reinitialize {
            parser
                .initialize_request(UnifiedParserInit::default())
                .unwrap();
        }
        assert_eq!(parse(&mut parser, "B"), [text("B")]);
        assert_eq!(taken(&mut parser), [0..2], "reinitialize={reinitialize}");
    }
}

/// Held bytes that `reset` returns keep their range, and the next request
/// counts on from them.
#[test]
fn reset_text_keeps_its_range_and_the_next_request_counts_on() {
    let mut parser = recording("deepseek_v4", UnifiedParserInit::default());
    assert_eq!(parse(&mut parser, "ab<"), [text("ab")]);
    assert_eq!(taken(&mut parser), [0..2]);
    assert_eq!(parser.reset(), "<");
    assert_eq!(taken(&mut parser), [2..3]);
    parser
        .initialize_request(UnifiedParserInit::default())
        .unwrap();
    assert_eq!(parse(&mut parser, "c"), [text("c")]);
    assert_eq!(taken(&mut parser), [3..4]);
}

/// Reasoning opened by the prompt, then a tool call, then an answer.
#[test]
fn reasoning_from_the_prompt_and_text_after_a_call_keep_their_offsets() {
    let call = "<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>";
    let input = format!("think</think>{call}done");
    let mut parser = recording(
        "glm47",
        UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Reasoning,
            ..UnifiedParserInit::default()
        },
    );
    let events = parse(&mut parser, &input);
    assert_eq!(events.first(), Some(&reasoning("think")));
    parser.finish().unwrap();
    let done = input.len() - 4;
    assert_eq!(taken(&mut parser), [0..5, done..input.len()]);
}

/// The sweep's corpus never reaches the paths below, so each gets an exact case.
const BAD_INVOKE: &str = concat!(
    "<｜DSML｜ invoke name=\"run\">",
    "<｜DSML｜ parameter name=\"value\" string=\"false\">{\"a\":",
    "</｜DSML｜ parameter></｜DSML｜ invoke>"
);

/// `finish` flushes a held `<` and then returns the deferred error. The caller
/// gets no events from a failed finish, so the flushed byte has no range either,
/// including when its range joined one the caller has not taken yet.
#[test]
fn a_failed_finish_drops_the_ranges_of_what_it_flushed() {
    let input = format!("{BAD}x<");
    let x = BAD.len()..BAD.len() + 1;
    for take_first in [true, false] {
        let mut parser = recording("deepseek_v41", UnifiedParserInit::default());
        assert_eq!(parse(&mut parser, &input), [text("x")]);
        let mut ranges = Vec::new();
        if take_first {
            ranges = taken(&mut parser);
            assert_eq!(ranges, [x.start..x.end]);
        }
        assert!(parser.finish().is_err());
        ranges.extend(taken(&mut parser));
        assert_eq!(ranges, [x.start..x.end], "take_first={take_first}");
        assert_eq!(parser.reset(), BAD);
        assert_eq!(taken(&mut parser), [0..BAD.len()]);
    }
}

/// A bare invoke whose arguments fail to parse is held for `reset`, between
/// two runs of text.
#[test]
fn a_failed_bare_invoke_keeps_its_range_for_reset() {
    let input = format!("GAP{BAD_INVOKE}TAIL");
    let end = 3 + BAD_INVOKE.len();
    let mut parser = recording("deepseek_v41", UnifiedParserInit::default());
    assert_eq!(parse(&mut parser, &input), [text("GAPTAIL")]);
    assert_eq!(taken(&mut parser), [0..3, end..input.len()]);
    assert!(parser.finish().is_err());
    assert_eq!(parser.reset(), BAD_INVOKE);
    assert_eq!(taken(&mut parser), [3..end]);
}

/// A block whose invoke the scanner cannot find yet stays in the buffer, and
/// `reset` returns it with the buffer's range.
#[test]
fn an_unfinished_block_in_the_buffer_keeps_its_range_for_reset() {
    let held = "<｜DSML｜tool_calls><｜DSML｜invoke nme=\"run\"></｜DSML｜invoke>";
    let input = format!("ab{held}");
    let mut parser = recording("deepseek_v4", UnifiedParserInit::default());
    assert_eq!(parse(&mut parser, &input), [text("ab")]);
    assert_eq!(parser.reset(), held);
    assert_eq!(taken(&mut parser), [0..input.len()]);
}

/// An invoke the emitter skips (`Ok(None)`, here Qwen3's empty function name)
/// is consumed but held with its block opener until the block closes, so
/// `reset` returns both.
#[test]
fn a_skipped_invoke_keeps_its_range_for_reset() {
    let held = "<tool_call>\n<function=>\n</function>\n";
    let input = format!("ab{held}");
    let mut parser = recording("qwen3", UnifiedParserInit::default());
    assert_eq!(parse(&mut parser, &input), [text("ab")]);
    assert_eq!(parser.reset(), held);
    assert_eq!(taken(&mut parser), [0..input.len()]);
}
