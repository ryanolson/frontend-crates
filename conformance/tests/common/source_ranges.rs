// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The source-range checker that the per-family sweeps share.
//!
//! A run feeds one split of one stream with recording on and with it off, then
//! checks four rules:
//!
//! * Same output: recording on and off give identical events, errors, and
//!   `reset` text.
//! * Verbatim and tiling: the ranges taken after each call, read from the input,
//!   equal that call's Text and Reasoning bytes in output order, with no gap.
//! * Causality: no range reaches past the bytes fed so far.
//! * Order: ranges from parse and finish calls never go back in the input.
//!   Byte equality alone accepts a range one repeat off ("ha ha ha"); this does not.
//!   `reset` text is exempt, because `reset` returns held marker text before held
//!   outside text.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Mutex;

use dynamo_parsers_v2::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Finish,
    FinishAtEos,
    Reset,
}

/// What one parser call returned, compared between recording on and off.
#[derive(Debug, PartialEq)]
pub enum CallResult {
    Events(Vec<UnifiedParserEvent>),
    /// Events committed before the error stay committed.
    Error(String, Vec<UnifiedParserEvent>),
    Reset(String),
}

pub struct Run {
    pub calls: Vec<CallResult>,
    /// Input bytes fed before each call returned.
    pub fed: Vec<usize>,
    /// Ranges taken after each call. Empty when not recording.
    pub ranges: Vec<Vec<Range<usize>>>,
    /// Whether the parser accepted `record_sources`.
    pub recorded: bool,
}

fn events_or_error(result: anyhow::Result<()>, output: UnifiedParserOutput) -> CallResult {
    match result {
        Ok(()) => CallResult::Events(output.events),
        Err(error) => CallResult::Error(error.to_string(), output.events),
    }
}

/// Feed `chunks` to a parser from `make`, which returns it initialized.
///
/// After a parse error, or a finish error, the caller recovers held text with
/// `reset`, so the run ends with a reset in both cases.
pub fn run(
    make: &dyn Fn() -> Box<dyn UnifiedParser>,
    chunks: &[&str],
    end: End,
    record: bool,
) -> Run {
    let mut parser = make();
    let recorded = record && parser.record_sources();
    let mut out = Run {
        calls: Vec::new(),
        fed: Vec::new(),
        ranges: Vec::new(),
        recorded,
    };
    let mut fed = 0;
    let take = |parser: &mut Box<dyn UnifiedParser>, out: &mut Run, fed: usize| {
        let mut ranges = Vec::new();
        parser.take_sources(&mut ranges);
        out.fed.push(fed);
        out.ranges.push(ranges);
    };
    let mut failed = false;
    for chunk in chunks {
        let mut output = UnifiedParserOutput::default();
        let result = parser.parse_into(chunk, &mut output);
        failed = result.is_err();
        fed += chunk.len();
        out.calls.push(events_or_error(result, output));
        take(&mut parser, &mut out, fed);
        if failed {
            break;
        }
    }
    let finished = match (end, failed) {
        (End::Reset, _) | (_, true) => None,
        (End::Finish, false) => Some(parser.finish()),
        (End::FinishAtEos, false) => Some(parser.finish_at_eos()),
    };
    let reset = match finished {
        None => true,
        Some(Ok(output)) => {
            out.calls.push(CallResult::Events(output.events));
            take(&mut parser, &mut out, fed);
            false
        }
        Some(Err(error)) => {
            out.calls
                .push(CallResult::Error(error.to_string(), Vec::new()));
            take(&mut parser, &mut out, fed);
            true
        }
    };
    if reset {
        out.calls.push(CallResult::Reset(parser.reset()));
        take(&mut parser, &mut out, fed);
    }
    out
}

/// The Text and Reasoning bytes one call committed, in output order.
pub fn committed_text(call: &CallResult) -> String {
    match call {
        CallResult::Events(events) | CallResult::Error(_, events) => events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::Text(text) | UnifiedParserEvent::Reasoning(text) => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect(),
        CallResult::Reset(text) => text.clone(),
    }
}

#[derive(Default)]
pub struct Tally {
    pub runs: usize,
    pub text_bytes: usize,
    pub finish_text_runs: usize,
    pub reset_text_runs: usize,
    pub failures: Vec<String>,
}

impl Tally {
    pub fn fail(&mut self, failure: String) {
        if self.failures.len() < 40 {
            self.failures.push(failure);
        }
    }

    pub fn add(&mut self, other: Tally) {
        self.runs += other.runs;
        self.text_bytes += other.text_bytes;
        self.finish_text_runs += other.finish_text_runs;
        self.reset_text_runs += other.reset_text_runs;
        for failure in other.failures {
            self.fail(failure);
        }
    }
}

/// Run one split with recording on and off, and check every rule. When the
/// parser refuses to record, only the same-output rule applies.
pub fn check(
    make: &dyn Fn() -> Box<dyn UnifiedParser>,
    at: &str,
    chunks: &[&str],
    end: End,
    tally: &mut Tally,
) {
    let input = chunks.concat();
    let off = run(make, chunks, end, false);
    let on = run(make, chunks, end, true);
    tally.runs += 1;
    let at = format!("{at} {end:?}");
    check_runs(&at, &input, chunks, &off, &on, tally);
}

pub fn check_runs(at: &str, input: &str, chunks: &[&str], off: &Run, on: &Run, tally: &mut Tally) {
    if on.calls != off.calls {
        tally.fail(format!(
            "{at}: recording changed the output\n  off: {:?}\n  on:  {:?}",
            off.calls, on.calls
        ));
        return;
    }
    if !on.recorded {
        if let Some(ranges) = on.ranges.iter().find(|ranges| !ranges.is_empty()) {
            tally.fail(format!(
                "{at}: a parser that refused to record returned ranges {ranges:?}"
            ));
        }
        return;
    }
    let mut floor = 0;
    for (index, call) in on.calls.iter().enumerate() {
        let text = committed_text(call);
        let ranges = &on.ranges[index];
        tally.text_bytes += text.len();
        let is_last = index + 1 == on.calls.len();
        if is_last && !text.is_empty() {
            match call {
                CallResult::Reset(_) => tally.reset_text_runs += 1,
                _ => tally.finish_text_runs += 1,
            }
        }
        let fed = on.fed[index];
        let mut rebuilt = Vec::new();
        for range in ranges {
            if range.end > fed || range.start > range.end {
                tally.fail(format!(
                    "{at}: call {index} range {range:?} is past the {fed} bytes fed\n  input: {input:?}"
                ));
                return;
            }
            if !matches!(call, CallResult::Reset(_)) {
                if range.start < floor {
                    tally.fail(format!(
                        "{at}: call {index} range {range:?} goes back before {floor}\n  ranges: {ranges:?}\n  input: {input:?}"
                    ));
                    return;
                }
                floor = range.end;
            }
            rebuilt.extend_from_slice(&input.as_bytes()[range.clone()]);
        }
        if rebuilt != text.as_bytes() {
            tally.fail(format!(
                "{at}: call {index} ranges do not rebuild the committed text\n  text:    {text:?}\n  rebuilt: {:?}\n  ranges: {ranges:?}\n  input: {input:?}\n  chunks: {chunks:?}",
                String::from_utf8_lossy(&rebuilt)
            ));
            return;
        }
    }
}

pub fn char_boundaries(text: &str) -> Vec<usize> {
    let mut at: Vec<usize> = text.char_indices().map(|(at, _)| at).collect();
    at.push(text.len());
    at
}

/// The sweep for one stream: the recorded split (finish, finish at EOS, reset),
/// one character per chunk (finish, finish at EOS), and every two-way split at a
/// character boundary (finish, plus reset after the head).
pub fn check_splits(
    make: &dyn Fn() -> Box<dyn UnifiedParser>,
    id: &str,
    recorded: &[&str],
    tally: &mut Tally,
) {
    let text = recorded.concat();
    let boundaries = char_boundaries(&text);
    for end in [End::Finish, End::FinishAtEos, End::Reset] {
        check(make, &format!("{id} recorded"), recorded, end, tally);
    }
    let chars: Vec<&str> = boundaries
        .windows(2)
        .map(|pair| &text[pair[0]..pair[1]])
        .collect();
    check(make, &format!("{id} one_char"), &chars, End::Finish, tally);
    check(
        make,
        &format!("{id} one_char"),
        &chars,
        End::FinishAtEos,
        tally,
    );
    for &split in &boundaries {
        let (head, tail) = text.split_at(split);
        let label = format!("{id} split@{split}");
        check(make, &label, &[head, tail], End::Finish, tally);
        // Reset after the head recovers held text, which must map back too.
        check(make, &label, &[head], End::Reset, tally);
    }
}

/// All splits of `input` that the exact-offset cases use: whole, one character
/// per chunk, and every two-way split.
pub fn all_splits(input: &str) -> Vec<Vec<&str>> {
    let mut splits = vec![vec![input]];
    splits.push(
        char_boundaries(input)
            .windows(2)
            .map(|pair| &input[pair[0]..pair[1]])
            .collect(),
    );
    for split in char_boundaries(input) {
        splits.push(vec![&input[..split], &input[split..]]);
    }
    splits
}

/// The ranges of a run, in output order, with adjacent ones joined.
pub fn joined_ranges(run: &Run) -> Vec<Range<usize>> {
    let mut joined: Vec<Range<usize>> = Vec::new();
    for range in run.ranges.iter().flatten().cloned() {
        match joined.last_mut() {
            Some(last) if last.end == range.start => last.end = range.end,
            _ => joined.push(range),
        }
    }
    joined
}

/// Counts tracing `why` codes, to show which recovery paths a corpus reached.
pub struct WhyCounter {
    pub counts: Mutex<BTreeMap<String, usize>>,
}

struct WhyVisitor<'a>(&'a mut Option<String>);

impl tracing::field::Visit for WhyVisitor<'_> {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "why" {
            *self.0 = Some(value.to_string());
        }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "why" {
            *self.0 = Some(format!("{value:?}").trim_matches('"').to_string());
        }
    }
}

impl tracing::Subscriber for WhyCounter {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut why = None;
        event.record(&mut WhyVisitor(&mut why));
        if let Some(why) = why {
            *self.counts.lock().unwrap().entry(why).or_default() += 1;
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Check every item of `items` on a pool of threads, counting `why` codes.
pub fn check_all<T: Send>(
    name: &str,
    items: Vec<T>,
    check_item: impl Fn(&T, &mut Tally) + Sync,
) -> (Tally, BTreeMap<String, usize>) {
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let items = Mutex::new(items.into_iter());
    let tally = Mutex::new(Tally::default());
    let counter = std::sync::Arc::new(WhyCounter {
        counts: Mutex::new(BTreeMap::new()),
    });
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let counter = counter.clone();
            let (items, tally, check_item) = (&items, &tally, &check_item);
            scope.spawn(move || {
                tracing::subscriber::with_default(counter, || {
                    loop {
                        let Some(item) = items.lock().unwrap().next() else {
                            return;
                        };
                        let mut mine = Tally::default();
                        check_item(&item, &mut mine);
                        tally.lock().unwrap().add(mine);
                    }
                })
            });
        }
    });
    let tally = tally.into_inner().unwrap();
    let counts = std::mem::take(&mut *counter.counts.lock().unwrap());
    eprintln!(
        "{name}: {} runs per recording mode, {} committed text bytes, {} runs with finish text, {} with reset text",
        tally.runs, tally.text_bytes, tally.finish_text_runs, tally.reset_text_runs
    );
    eprintln!("{name}: why codes reached (both recording modes): {counts:?}");
    (tally, counts)
}
