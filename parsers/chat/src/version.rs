// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The parser generation a deployment selects.
//!
//! Dynamo reads it from `DYN_PARSER_VERSION` once at start-up. This crate never reads
//! the environment: the caller parses the setting with [`ParserVersion::try_from`] and
//! passes the result to every function that depends on it, so routing cannot change
//! inside a process and tests can cover every value without a new process.

/// The environment variable Dynamo reads the parser version from. Error messages name
/// it, so a client sees the same text as from Dynamo.
pub const PARSER_VERSION_ENV: &str = "DYN_PARSER_VERSION";

/// Parser generation shared by startup validation and frontend routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserVersion {
    /// A family with a unified v2 parser uses it; any other family uses v1.
    Auto,
    /// The v1 parsers only.
    V1,
    /// The unified v2 parsers only. A parser pair without one is refused.
    V2,
}

impl TryFrom<Option<&str>> for ParserVersion {
    type Error = anyhow::Error;

    /// Parse the setting's value: unset, `auto`, `1`, or `2`.
    fn try_from(value: Option<&str>) -> anyhow::Result<Self> {
        match value {
            Some("1") => Ok(Self::V1),
            Some("2") => Ok(Self::V2),
            Some("auto") | None => Ok(Self::Auto),
            Some(value) => {
                anyhow::bail!("{PARSER_VERSION_ENV} must be unset, auto, 1, or 2; got {value:?}")
            }
        }
    }
}
