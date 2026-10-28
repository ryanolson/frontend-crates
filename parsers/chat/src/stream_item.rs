// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The stream item that the v2 stream adapter reads and rewrites.
//!
//! The adapter was moved from dynamo `lib/llm`. There it runs on
//! dynamo-runtime's `Annotated` over dynamo-llm's chunk type, which carries
//! fields beside the OpenAI chunk: `nvext`, token metrics, prompt logprobs, and
//! tool-call completion evidence. The adapter moves those fields between the
//! chunks it splits and synthesizes. This crate cannot name dynamo's types, so
//! the adapter is generic over [`UnifiedStreamItem`] and [`UnifiedStreamChunk`].
//! dynamo implements them for its own types and keeps every field as it is
//! today. A concrete item type here would make dynamo strip and re-attach those
//! fields around the adapter, and a synthesized chunk has no input chunk to
//! take them from.
//!
//! This crate implements the item trait for the jail's [`Annotated`].
//!
//! # Sources
//!
//! A chunk type may carry a [`SourceSlot`]. When the input chunks have one, the
//! adapter records where each text byte came from: it turns recording on in
//! each parser, and fails the stream with an error item when a parser cannot
//! record. It overwrites the slot of every chunk it emits with one
//! [`ChoiceSources`](dynamo_parsers::sources::ChoiceSources) per choice in that
//! chunk; a choice with no text gets empty ranges. `scrub_synthetic` and the
//! `clear_*` methods do not touch the slot. A chunk type with no slot (the
//! trait defaults) records nothing and pays nothing.

use dynamo_parsers::sources::SourceSlot;
use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_protocols::types::CreateChatCompletionStreamResponse;
use serde::{Deserialize, Serialize};

/// Parser lifecycle evidence carried between serving components, never to OpenAI clients.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ToolCallCompletion {
    pub choice_index: u32,
    pub tool_index: u32,
    pub complete: bool,
}

/// One chat chunk plus the per-chunk fields that travel with it.
pub trait UnifiedStreamChunk: SourceSlot + Clone + Send + Sync + 'static {
    /// The OpenAI chunk.
    fn inner(&self) -> &CreateChatCompletionStreamResponse;

    /// The OpenAI chunk, for in-place rewrite.
    fn inner_mut(&mut self) -> &mut CreateChatCompletionStreamResponse;

    /// Tool-call completion evidence on this chunk.
    fn tool_call_completion(&self) -> &[ToolCallCompletion];

    /// Tool-call completion evidence on this chunk, for in-place rewrite.
    fn tool_call_completion_mut(&mut self) -> &mut Vec<ToolCallCompletion>;

    /// Clear the extension field. When one chunk fans out into several, only
    /// the last child keeps it.
    fn clear_nvext(&mut self);

    /// Clear the token metrics. When one chunk fans out into several, only one
    /// child keeps them.
    fn clear_llm_metrics(&mut self);

    /// Clear every per-chunk field of a copy that becomes a synthetic chunk:
    /// usage, token metrics, completion evidence, extension, and prompt
    /// logprobs. A synthetic chunk consumed no backend data, so it must not
    /// repeat them.
    fn scrub_synthetic(&mut self);
}

/// A stream item that may carry one chunk, plus its annotation.
pub trait UnifiedStreamItem: Sized + Send + 'static {
    /// The chunk type.
    type Chunk: UnifiedStreamChunk;

    /// The annotation fields (in dynamo: `id`, `event`, `comment`, `error`).
    /// The default is no annotation.
    type Annotation: Clone + Default + Send + 'static;

    /// Whether this item is a terminal error from the backend.
    fn is_error(&self) -> bool;

    /// Split the item into its chunk and its annotation, or return it
    /// unchanged when it carries no chunk.
    fn into_parts(self) -> Result<(Self::Chunk, Self::Annotation), Self>;

    /// Build an item from a chunk and an annotation.
    fn from_parts(data: Self::Chunk, annotation: Self::Annotation) -> Self;

    /// Build a terminal error item, for which [`is_error`](Self::is_error)
    /// returns `true`. The adapter ends the stream with one when it cannot
    /// record the sources that the input chunks ask for.
    fn from_error(message: String) -> Self;
}

/// The annotation fields of the jail's [`Annotated`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AnnotationFields {
    pub id: Option<String>,
    pub event: Option<String>,
    pub comment: Option<Vec<String>>,
    pub error: Option<String>,
}

impl<C: UnifiedStreamChunk> UnifiedStreamItem for Annotated<C> {
    type Chunk = C;
    type Annotation = AnnotationFields;

    fn is_error(&self) -> bool {
        self.event.as_deref() == Some("error")
    }

    fn into_parts(self) -> Result<(C, AnnotationFields), Self> {
        let Annotated {
            data,
            id,
            event,
            comment,
            error,
        } = self;
        let annotation = AnnotationFields {
            id,
            event,
            comment,
            error,
        };
        match data {
            Some(data) => Ok((data, annotation)),
            None => Err(Annotated {
                data: None,
                id: annotation.id,
                event: annotation.event,
                comment: annotation.comment,
                error: annotation.error,
            }),
        }
    }

    fn from_parts(data: C, annotation: AnnotationFields) -> Self {
        Annotated {
            data: Some(data),
            id: annotation.id,
            event: annotation.event,
            comment: annotation.comment,
            error: annotation.error,
        }
    }

    fn from_error(message: String) -> Self {
        Annotated {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: None,
            error: Some(message),
        }
    }
}
