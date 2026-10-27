// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The tool-output grammar a request installed, moved from dynamo `lib/llm/src/protocols/openai.rs`.

use serde::{Deserialize, Serialize};

/// The tool-output grammar installed for one request.
///
/// Batch and streaming consumers carry this decision forward instead of deriving it
/// again from `tool_choice`, which cannot distinguish JSON guidance from a native
/// structural tag or a family-specific prompt constraint.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidedToolConstraint {
    /// No tool-choice generation constraint was installed.
    #[default]
    None,
    /// Generation was pinned to the model family's native tool-call markup.
    StructuralTag,
    /// The model emits only the named tool's argument object.
    GuidedJsonNamed { tool_name: String },
    /// The model emits one call object or an array of `{name, parameters}` objects.
    GuidedJsonRequired,
}

impl GuidedToolConstraint {
    pub fn installs_guided_json(&self) -> bool {
        matches!(
            self,
            Self::GuidedJsonNamed { .. } | Self::GuidedJsonRequired
        )
    }

    pub fn uses_structural_tag(&self) -> bool {
        matches!(self, Self::StructuralTag)
    }
}
