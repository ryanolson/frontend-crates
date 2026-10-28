// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The serde_json features this crate turns on for the crates that depend on it.
//!
//! Cargo turns a feature on for every crate in a build. serde_json's
//! `arbitrary_precision` changes how all of them read and write numbers: a
//! number buffered by `#[serde(flatten)]` no longer reads as an `f32` or an
//! `i64`, a `Value` number serializes as a one-key map to non-JSON formats
//! such as MessagePack, and number text is kept as sent (`1.50`, not `1.5`).
//! So this crate leaves it off unless a consumer asks for it with the
//! `arbitrary-precision` feature.

use std::process::Command;

/// The serde_json features that a build of this crate turns on, with
/// `features` passed to cargo.
fn serde_json_features(features: &[&str]) -> Vec<String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut command = Command::new(cargo);
    command
        .args([
            "tree",
            "--package",
            "dynamo-parsers-v2",
            "--edges",
            "features,normal,build",
            "--invert",
            "serde_json",
            "--prefix",
            "none",
            "--offline",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    if !features.is_empty() {
        command.args(["--features", &features.join(",")]);
    }
    let output = command.output().expect("run cargo tree");
    assert!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut found: Vec<String> = String::from_utf8(output.stdout)
        .expect("utf-8")
        .lines()
        .filter_map(|line| line.strip_prefix("serde_json feature \""))
        .filter_map(|rest| rest.split('"').next())
        .map(str::to_string)
        .collect();
    found.sort();
    found.dedup();
    found
}

#[test]
fn arbitrary_precision_is_off_by_default() {
    let features = serde_json_features(&[]);
    assert!(
        features.iter().any(|f| f == "std"),
        "control: the tree lists serde_json's features: {features:?}"
    );
    assert!(
        !features.iter().any(|f| f == "arbitrary_precision"),
        "dynamo-parsers-v2 turns on arbitrary_precision by default: {features:?}"
    );
}

#[test]
fn the_arbitrary_precision_feature_turns_it_on() {
    let features = serde_json_features(&["arbitrary-precision"]);
    assert!(
        features.iter().any(|f| f == "arbitrary_precision"),
        "the arbitrary-precision feature does not reach serde_json: {features:?}"
    );
}
