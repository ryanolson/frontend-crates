// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The release configuration this crate needs before its first publish.

const RELEASE_CONFIG: &str = include_str!("../../../release-plz.toml");

fn release_override(package: &str) -> Option<bool> {
    RELEASE_CONFIG
        .split("[[package]]")
        .skip(1)
        .find_map(|section| {
            let mut name = None;
            let mut release = None;
            for line in section.lines() {
                let line = line.trim();
                if line.starts_with('[') {
                    break;
                }
                if let Some((key, value)) = line.split_once('=') {
                    match key.trim() {
                        "name" => name = Some(value.trim().trim_matches('"')),
                        "release" => release = value.trim().parse::<bool>().ok(),
                        _ => {}
                    }
                }
            }
            (name == Some(package)).then_some(release).flatten()
        })
}

#[test]
fn reads_existing_release_overrides() {
    assert_eq!(release_override("dynamo-demo-server"), Some(false));
    assert_eq!(release_override("dynamo-parsers-v2-py"), Some(false));
    assert_eq!(release_override("dynamo-protocols"), None);
}

/// RELEASING.md: a new crate keeps `release = false` until its first manual publish.
/// Delete this test in the PR that removes the override after that publish.
#[test]
fn new_crate_waits_for_its_first_manual_publish() {
    assert_eq!(
        release_override(env!("CARGO_PKG_NAME")),
        Some(false),
        "RELEASING.md requires release=false until the first manual publish"
    );
}
