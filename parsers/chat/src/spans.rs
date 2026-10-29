// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Caller byte ranges kept beside text that the stream adapter holds, splits,
//! and joins.
//!
//! A list of ranges "tiles" a text when the ranges' lengths add up to the
//! text's byte length, in order: byte `k` of the text came from the input byte
//! found by walking `k` bytes into the ranges.

use std::ops::{Deref, DerefMut, Range};

/// Caller byte ranges that tile one text, in order. Most text comes from one
/// stretch of input, so one range is held inline: a vector per event cost
/// recording about one allocation per event in the adapter.
pub(crate) type Spans = InlineList<Range<usize>>;

/// A list that holds one item inline and behaves as a `Vec` otherwise.
#[derive(Clone, Debug)]
pub(crate) struct InlineList<T>(Repr<T>);

#[derive(Clone, Debug)]
enum Repr<T> {
    Empty,
    One(T),
    Many(Vec<T>),
}

impl<T> Default for InlineList<T> {
    fn default() -> Self {
        Self(Repr::Empty)
    }
}

impl<T> InlineList<T> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Appends `item`, as `Vec::push`.
    pub(crate) fn push(&mut self, item: T) {
        match std::mem::take(self).0 {
            Repr::Empty => self.0 = Repr::One(item),
            Repr::One(first) => self.0 = Repr::Many(vec![first, item]),
            Repr::Many(mut items) => {
                items.push(item);
                self.0 = Repr::Many(items);
            }
        }
    }

    /// Splits off the items from index `at`, as `Vec::split_off`.
    pub(crate) fn split_off(&mut self, at: usize) -> Self {
        match &mut self.0 {
            Repr::Many(items) => Self::from(items.split_off(at)),
            _ if at == 0 => std::mem::take(self),
            _ => Self::new(),
        }
    }

    /// The items as a vector, which allocates for one item.
    pub(crate) fn into_vec(self) -> Vec<T> {
        match self.0 {
            Repr::Empty => Vec::new(),
            Repr::One(item) => vec![item],
            Repr::Many(items) => items,
        }
    }
}

impl<T> From<Vec<T>> for InlineList<T> {
    fn from(mut items: Vec<T>) -> Self {
        match items.len() {
            0 => Self::new(),
            1 => Self(Repr::One(items.pop().expect("one item"))),
            _ => Self(Repr::Many(items)),
        }
    }
}

impl<T> Deref for InlineList<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match &self.0 {
            Repr::Empty => &[],
            Repr::One(item) => std::slice::from_ref(item),
            Repr::Many(items) => items,
        }
    }
}

impl<T> DerefMut for InlineList<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.0 {
            Repr::Empty => &mut [],
            Repr::One(item) => std::slice::from_mut(item),
            Repr::Many(items) => items,
        }
    }
}

/// Two lists are equal when they hold the same items, however stored.
impl<T: PartialEq> PartialEq for InlineList<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for InlineList<T> {}

impl<T> Extend<T> for InlineList<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        for item in items {
            self.push(item);
        }
    }
}

impl<T> IntoIterator for InlineList<T> {
    type Item = T;
    type IntoIter = InlineListIntoIter<T>;

    fn into_iter(self) -> InlineListIntoIter<T> {
        match self.0 {
            Repr::Empty => InlineListIntoIter::One(None),
            Repr::One(item) => InlineListIntoIter::One(Some(item)),
            Repr::Many(items) => InlineListIntoIter::Many(items.into_iter()),
        }
    }
}

/// The owned items of an [`InlineList`].
pub(crate) enum InlineListIntoIter<T> {
    One(Option<T>),
    Many(std::vec::IntoIter<T>),
}

impl<T> Iterator for InlineListIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self {
            Self::One(item) => item.take(),
            Self::Many(items) => items.next(),
        }
    }
}

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
    Spans::new()
}

/// Maps the input offsets of one parser instance to the caller's offsets.
///
/// The parser counts every byte it received. The adapter feeds it the prompt
/// prefix first (`origin` bytes), then text whose caller ranges it knows. A
/// parser range before the origin would date prompt bytes as output, so it is
/// refused. So is a range whose input bytes differ from the text it dates: the
/// map keeps the fed text to check that.
pub(crate) struct FeedMap {
    origin: usize,
    end: usize,
    /// Local start offset and caller range of each fed piece, in feed order.
    pieces: Vec<(usize, Range<usize>)>,
    /// The text fed after the origin.
    text: String,
}

impl FeedMap {
    pub(crate) fn new(origin: usize) -> Self {
        Self {
            origin,
            end: origin,
            pieces: Vec::new(),
            text: String::new(),
        }
    }

    /// Record that the parser received `text`, tiled by `spans`.
    pub(crate) fn feed(&mut self, spans: &[Range<usize>], text: &str) {
        self.text.push_str(text);
        for range in spans {
            if range.is_empty() {
                continue;
            }
            self.pieces.push((self.end, range.clone()));
            self.end += range.len();
        }
    }

    /// Append the caller ranges of the parser's local range `local`, which
    /// dates the text bytes `expected`, to `out`.
    pub(crate) fn map(
        &self,
        local: Range<usize>,
        expected: &[u8],
        out: &mut Spans,
    ) -> Result<(), String> {
        if local.start < self.origin || local.end > self.end || local.start > local.end {
            return Err(format!(
                "parser source range {local:?} is outside the fed input {:?}",
                self.origin..self.end
            ));
        }
        let fed = &self.text.as_bytes()[local.start - self.origin..local.end - self.origin];
        if fed != expected {
            return Err(format!(
                "parser source range {local:?} holds other bytes than the text it dates"
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

    fn spans(ranges: Vec<Range<usize>>) -> Spans {
        Spans::from(ranges)
    }

    #[test]
    fn split_and_append_keep_the_tiling() {
        let mut list = spans(vec![0..3, 10..14]);
        let tail = split_spans(&mut list, 5);
        assert_eq!(list, spans(vec![0..3, 10..12]));
        assert_eq!(tail, spans(vec![12..14]));
        let mut joined = list.clone();
        append_spans(&mut joined, tail);
        assert_eq!(joined, spans(vec![0..3, 10..14]));
        assert_eq!(split_spans(&mut joined, 7), Spans::new());
        assert_eq!(split_spans(&mut spans(vec![0..3]), 3), Spans::new());
        // One range is split inline too.
        let mut one = spans(vec![4..10]);
        assert_eq!(split_spans(&mut one, 2), spans(vec![6..10]));
        assert_eq!(one, spans(vec![4..6]));
        assert_eq!(split_spans(&mut one, 0), spans(vec![4..6]));
        assert_eq!(one, Spans::new());
    }

    /// One range is held inline, with `Vec` semantics.
    #[test]
    fn one_range_is_inline() {
        let mut list = Spans::new();
        list.push(1..2);
        assert!(matches!(list.0, Repr::One(_)));
        assert_eq!(
            std::mem::size_of::<Spans>(),
            std::mem::size_of::<Vec<Range<usize>>>()
        );
        list.push(2..3);
        assert_eq!(&*list, &[1..2, 2..3]);
        assert_eq!(
            list.clone().into_iter().collect::<Vec<_>>(),
            vec![1..2, 2..3]
        );
        assert_eq!(list.split_off(1), spans(vec![2..3]));
        assert_eq!(list.into_vec(), vec![1..2]);
    }

    #[test]
    fn feed_map_maps_across_pieces_and_refuses_the_prefix() {
        let mut map = FeedMap::new(4);
        map.feed(&[100..103, 200..202], "abcde");
        let mut out = Spans::new();
        map.map(5..8, b"bcd", &mut out).unwrap();
        assert_eq!(out, spans(vec![101..103, 200..201]));
        assert!(map.map(5..8, b"xyz", &mut Spans::new()).is_err());
        assert!(map.map(3..5, b"ab", &mut Spans::new()).is_err());
        assert!(map.map(8..10, b"e?", &mut Spans::new()).is_err());
    }
}
