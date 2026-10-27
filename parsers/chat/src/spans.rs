// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Caller byte ranges kept beside text that the stream adapter holds, splits,
//! and joins.
//!
//! A list of ranges "tiles" a text when the ranges' lengths add up to the
//! text's byte length, in order: byte `k` of the text came from the input byte
//! found by walking `k` bytes into the ranges.

use std::ops::Range;

/// Caller byte ranges that tile one text, in order.
pub(crate) type Spans = Vec<Range<usize>>;

/// The number of bytes the ranges cover.
pub(crate) fn spans_len(spans: &[Range<usize>]) -> usize {
    spans.iter().map(|range| range.len()).sum()
}

/// Append ranges, joining a range that starts where the last one ends.
pub(crate) fn append_spans(dst: &mut Spans, src: impl IntoIterator<Item = Range<usize>>) {
    for range in src {
        if range.is_empty() {
            continue;
        }
        match dst.last_mut() {
            Some(last) if last.end == range.start => last.end = range.end,
            _ => dst.push(range),
        }
    }
}

/// Split the ranges after byte `at`: `spans` keeps bytes `..at`, and the
/// ranges of bytes `at..` are returned.
pub(crate) fn split_spans(spans: &mut Spans, at: usize) -> Spans {
    let mut seen = 0;
    for position in 0..spans.len() {
        let len = spans[position].len();
        if seen + len > at {
            let mut tail = spans.split_off(position);
            let cut = tail[0].start + (at - seen);
            if cut > tail[0].start {
                spans.push(tail[0].start..cut);
                tail[0].start = cut;
            }
            return tail;
        }
        seen += len;
    }
    Vec::new()
}

/// Maps the input offsets of one parser instance to the caller's offsets.
///
/// The parser counts every byte it received. The adapter feeds it the prompt
/// prefix first (`origin` bytes), then text whose caller ranges it knows. A
/// parser range before the origin would date prompt bytes as output, so it is
/// refused.
pub(crate) struct FeedMap {
    origin: usize,
    end: usize,
    /// Local start offset and caller range of each fed piece, in feed order.
    pieces: Vec<(usize, Range<usize>)>,
}

impl FeedMap {
    pub(crate) fn new(origin: usize) -> Self {
        Self {
            origin,
            end: origin,
            pieces: Vec::new(),
        }
    }

    /// Record that the parser received text tiled by `spans`.
    pub(crate) fn feed(&mut self, spans: &[Range<usize>]) {
        for range in spans {
            if range.is_empty() {
                continue;
            }
            self.pieces.push((self.end, range.clone()));
            self.end += range.len();
        }
    }

    /// Append the caller ranges of the parser's local range `local` to `out`.
    pub(crate) fn map(&self, local: Range<usize>, out: &mut Spans) -> Result<(), String> {
        if local.start < self.origin || local.end > self.end || local.start > local.end {
            return Err(format!(
                "parser source range {local:?} is outside the fed input {:?}",
                self.origin..self.end
            ));
        }
        if local.is_empty() {
            return Ok(());
        }
        let mut position = self
            .pieces
            .partition_point(|(start, _)| *start <= local.start)
            - 1;
        let mut at = local.start;
        while at < local.end {
            let (start, caller) = &self.pieces[position];
            let offset = at - start;
            let take = (caller.len() - offset).min(local.end - at);
            append_spans(
                out,
                std::iter::once(caller.start + offset..caller.start + offset + take),
            );
            at += take;
            position += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    #[test]
    fn split_and_append_keep_the_tiling() {
        let mut spans = vec![0..3, 10..14];
        let tail = split_spans(&mut spans, 5);
        assert_eq!(spans, vec![0..3, 10..12]);
        assert_eq!(tail, vec![12..14]);
        let mut joined = spans.clone();
        append_spans(&mut joined, tail);
        assert_eq!(joined, vec![0..3, 10..14]);
        assert_eq!(split_spans(&mut joined, 7), Vec::<Range<usize>>::new());
        assert_eq!(split_spans(&mut vec![0..3], 3), Vec::<Range<usize>>::new());
    }

    #[test]
    fn feed_map_maps_across_pieces_and_refuses_the_prefix() {
        let mut map = FeedMap::new(4);
        map.feed(&[100..103, 200..202]);
        let mut out = Vec::new();
        map.map(5..8, &mut out).unwrap();
        assert_eq!(out, vec![101..103, 200..201]);
        assert!(map.map(3..5, &mut Vec::new()).is_err());
        assert!(map.map(8..10, &mut Vec::new()).is_err());
    }
}
