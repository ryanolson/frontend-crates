// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The stream item that the chat stream stages read and rewrite.
//!
//! The reasoning stream stages and the jail's terminal-error contract were
//! moved from dynamo `lib/llm/src/preprocessor.rs`. There they run on
//! dynamo-runtime's `Annotated` over dynamo-llm's chunk type, which carries
//! dynamo-only fields (metrics, `nvext`, prompt logprobs) beside the OpenAI
//! chunk. This crate cannot name those types. The stages are generic over
//! [`ChatStreamItem`] instead, so dynamo keeps its own item and its fields pass
//! through each stage untouched. A concrete item type here would make dynamo
//! strip and re-attach those fields around every stage, and the end-of-stream
//! flush chunk has no input chunk to take them from.
//!
//! This crate implements the trait for its own [`Annotated`] over
//! [`CreateChatCompletionStreamResponse`], the item the jail already uses, and
//! for [`SourcedChunk`], the same item with a source slot.
//!
//! An item may carry a [`SourceSlot`]. When the input items have one, each
//! stage records where its text came from and writes, for each choice of each
//! output chunk, the input byte ranges that tile its `content` and its
//! `reasoning_content`. Items without a slot record nothing.

use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, CreateChatCompletionStreamResponse,
};

use crate::sources::{ChoiceSources, ChoiceText, SourceSlot, entry_covers, match_choice_sources};
use crate::tool_calling::jail::Annotated;

/// A stream item that may carry one OpenAI chat completion chunk.
pub trait ChatStreamItem: SourceSlot + Clone + Send + 'static {
    /// Whether this item is a terminal error from the backend.
    ///
    /// dynamo-runtime decides this from the annotation event (`event ==
    /// "error"`), not from the payload.
    fn is_error(&self) -> bool;

    /// The chat chunk, if this item carries one.
    fn chunk(&self) -> Option<&CreateChatCompletionStreamResponse>;

    /// The chat chunk, if this item carries one, for in-place rewrite.
    fn chunk_mut(&mut self) -> Option<&mut CreateChatCompletionStreamResponse>;

    /// Clear the transport and per-chunk fields of a copy that becomes a
    /// synthetic chunk.
    ///
    /// A stage that flushes held text at the end of the stream clones an
    /// earlier item as the envelope. That chunk consumed no backend data, so
    /// it must not repeat the earlier item's usage or metadata.
    fn scrub_synthetic(&mut self);
}

impl ChatStreamItem for Annotated<CreateChatCompletionStreamResponse> {
    fn is_error(&self) -> bool {
        self.event.as_deref() == Some("error")
    }

    fn chunk(&self) -> Option<&CreateChatCompletionStreamResponse> {
        self.data.as_ref()
    }

    fn chunk_mut(&mut self) -> Option<&mut CreateChatCompletionStreamResponse> {
        self.data.as_mut()
    }

    fn scrub_synthetic(&mut self) {
        self.event = None;
        self.comment = None;
        self.error = None;
        if let Some(data) = self.data.as_mut() {
            data.usage = None;
        }
    }
}

/// The text lengths of each choice, in order, for [`match_choice_sources`].
///
/// [`match_choice_sources`]: crate::sources::match_choice_sources
pub fn choice_texts(choices: &[ChatChoiceStream]) -> Vec<ChoiceText> {
    choices.iter().map(choice_text).collect()
}

/// The text lengths of one choice; see [`choice_texts`].
pub fn choice_text(choice: &ChatChoiceStream) -> ChoiceText {
    ChoiceText {
        index: choice.index,
        content_len: match &choice.delta.content {
            Some(ChatCompletionMessageContent::Text(text)) => text.len(),
            _ => 0,
        },
        reasoning_len: choice
            .delta
            .reasoning_content
            .as_ref()
            .map_or(0, String::len),
    }
}

/// The entries of `matched` that are set, in order, in `matched`'s own
/// vector: a stage that writes each choice's output entry back at its
/// position gets its output list without a new allocation.
// `filter_map` collects in place; `flatten`, which clippy suggests, does not.
#[allow(clippy::filter_map_identity)]
pub fn into_entries(matched: Vec<Option<ChoiceSources>>) -> Vec<ChoiceSources> {
    matched.into_iter().filter_map(|entry| entry).collect()
}

/// [`match_choice_sources`] over a chunk's choices.
///
/// When each entry names the choice at its own position, as a caller that
/// writes one entry per choice does, the rule pairs them position by
/// position. That case reuses the entries' vector and allocates nothing: the
/// general rule allocated about five times per chunk, in every stage that
/// records. Any other shape goes through the general rule.
///
/// [`match_choice_sources`]: crate::sources::match_choice_sources
pub fn match_chunk_sources(
    entries: Vec<ChoiceSources>,
    choices: &[ChatChoiceStream],
) -> Vec<Option<ChoiceSources>> {
    let positional = entries.len() == choices.len()
        && entries
            .iter()
            .zip(choices)
            .all(|(entry, choice)| entry.index == choice.index);
    if !positional {
        return match_choice_sources(entries, &choice_texts(choices));
    }
    entries
        .into_iter()
        .zip(choices)
        .map(|(entry, choice)| entry_covers(&entry, &choice_text(choice)).then_some(entry))
        .collect()
}

/// [`Annotated`] carries no source slot.
impl SourceSlot for Annotated<CreateChatCompletionStreamResponse> {}

/// A chat chunk with a source slot: the item to feed the v1 stream stages and
/// the jail when the caller needs to know which input bytes each output byte
/// came from.
///
/// `sources` is `Some` on every item of a recording stream, with one entry
/// per choice of `chunk` in the same order. On input, an entry gives the
/// caller's byte ranges of that choice's `content` (and `reasoning_content`, if
/// any). On output, the stages write the ranges of the text they emit. A
/// choice whose stage cannot record gets no entry, so the caller can tell that
/// its text has no source.
#[derive(Clone, Debug)]
pub struct SourcedChunk {
    pub chunk: Annotated<CreateChatCompletionStreamResponse>,
    pub sources: Option<Vec<ChoiceSources>>,
}

impl SourceSlot for SourcedChunk {
    fn sources(&self) -> Option<&[ChoiceSources]> {
        self.sources.as_deref()
    }

    fn sources_mut(&mut self) -> Option<&mut Vec<ChoiceSources>> {
        self.sources.as_mut()
    }
}

impl ChatStreamItem for SourcedChunk {
    fn is_error(&self) -> bool {
        self.chunk.is_error()
    }

    fn chunk(&self) -> Option<&CreateChatCompletionStreamResponse> {
        self.chunk.chunk()
    }

    fn chunk_mut(&mut self) -> Option<&mut CreateChatCompletionStreamResponse> {
        self.chunk.chunk_mut()
    }

    fn scrub_synthetic(&mut self) {
        self.chunk.scrub_synthetic();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::ChatCompletionStreamResponseDelta;

    fn choice(index: u32, content: usize, reasoning: usize) -> ChatChoiceStream {
        let text = |n: usize| "x".repeat(n);
        #[allow(deprecated)]
        ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                content: (content > 0).then(|| ChatCompletionMessageContent::Text(text(content))),
                function_call: None,
                tool_calls: None,
                role: None,
                refusal: None,
                reasoning_content: (reasoning > 0).then(|| text(reasoning)),
            },
            finish_reason: None,
            logprobs: None,
        }
    }

    // A one-range list is what an entry holds here.
    #[allow(clippy::single_range_in_vec_init)]
    fn entry(index: u32, content: usize, reasoning: usize) -> ChoiceSources {
        let ranges = |n: usize| if n == 0 { Vec::new() } else { vec![0..n] };
        ChoiceSources {
            index,
            content: ranges(content),
            reasoning: ranges(reasoning),
        }
    }

    /// Every shape of up to three choices and three entries, over two
    /// indices and lengths 0 to 1 (2 for an entry, which cannot match): the
    /// chunk-level match gives what the general rule gives.
    #[test]
    fn the_chunk_match_is_the_general_rule() {
        let choice_shapes: Vec<(u32, usize, usize)> = (0..2)
            .flat_map(|i| (0..2).flat_map(move |c| (0..2).map(move |r| (i, c, r))))
            .collect();
        let entry_shapes: Vec<(u32, usize, usize)> = (0..2)
            .flat_map(|i| (0..3).flat_map(move |c| (0..2).map(move |r| (i, c, r))))
            .collect();
        fn lists<T: Copy>(shapes: &[T], max: usize) -> Vec<Vec<T>> {
            let mut out = vec![Vec::new()];
            let mut level = vec![Vec::new()];
            for _ in 0..max {
                level = level
                    .iter()
                    .flat_map(|list: &Vec<T>| {
                        shapes.iter().map(move |shape| {
                            let mut next = list.clone();
                            next.push(*shape);
                            next
                        })
                    })
                    .collect();
                out.extend(level.iter().cloned());
            }
            out
        }
        let mut checked = 0;
        for choices in lists(&choice_shapes, 3) {
            let choices: Vec<ChatChoiceStream> =
                choices.iter().map(|&(i, c, r)| choice(i, c, r)).collect();
            for entries in lists(&entry_shapes, 3) {
                let entries: Vec<ChoiceSources> =
                    entries.iter().map(|&(i, c, r)| entry(i, c, r)).collect();
                let general = match_choice_sources(entries.clone(), &choice_texts(&choices));
                assert_eq!(match_chunk_sources(entries, &choices), general);
                checked += 1;
            }
        }
        assert!(checked > 100_000, "{checked} shapes");
    }
}
