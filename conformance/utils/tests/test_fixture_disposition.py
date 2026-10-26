# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import extract_fixtures  # noqa: E402
import capture_stimulus  # noqa: E402
import fixture_disposition  # noqa: E402
import package_fixtures  # noqa: E402
import stream_capture_archive  # noqa: E402
import unified_history  # noqa: E402
import capture_sources
from fixture_test_support import configure_extractor_root

_TEST_STREAM_CAPTURE_VERSION = "99.0.0"
_TEST_STREAM_CAPTURE_ROOT = f"dynamo_v2-{_TEST_STREAM_CAPTURE_VERSION}"


def test_canonicalize_unified_inputs_rejects_duplicate_scenario_owners():
    records = {
        ("gemma4", "UNIFIED.3-1"): {"scenario": "text_only"},
        ("gemma4", "UNIFIED.9-9"): {"scenario": "text_only"},
    }

    with pytest.raises(ValueError, match="duplicate Unified scenario ownership"):
        fixture_disposition.canonicalize_unified_inputs(records)


def test_canonicalize_unified_inputs_allows_one_historical_alias_rename():
    scenario = "gemma4_guided_json_visible_call_prose_before_reasoning"
    records = {
        ("gemma4", "UNIFIED.31-29"): {"scenario": scenario, "input": "same"},
        ("gemma4", "UNIFIED.g4-1"): {"scenario": scenario, "input": "same"},
    }

    canonical, aliases = fixture_disposition.canonicalize_unified_inputs(records)

    ident = (
        "gemma4",
        fixture_disposition.canonical_unified_case_key(
            "gemma4", "UNIFIED.g4-1", scenario
        ),
    )
    assert canonical == {ident: records[("gemma4", "UNIFIED.g4-1")]}
    assert aliases[("gemma4", "UNIFIED.31-29")] == ident
    assert aliases[("gemma4", "UNIFIED.g4-1")] == ident


def test_stream_input_corrections_compare_yaml_scalar_types():
    old_docs = {
        "glm47/TOOLCALLING.streamv1.7.yaml": {
            "family": "glm47",
            "mode": "streamv1",
            "cases": {"TOOLCALLING.streamv1.7.g": {"enum": [True]}},
        }
    }
    corrected_docs = {
        "glm47/TOOLCALLING.streamv1.7.yaml": {
            "family": "glm47",
            "mode": "streamv1",
            "cases": {"TOOLCALLING.streamv1.7.g": {"enum": [1]}},
        }
    }

    assert stream_capture_archive._approved_case_additions(old_docs, corrected_docs, {}, {}) is None
    assert stream_capture_archive._approved_case_additions(
        old_docs,
        corrected_docs,
        {},
        {},
        {("glm47", "TOOLCALLING.streamv1.7.g")},
    ) == (set(), {("glm47", "TOOLCALLING.streamv1.7.g")})


@pytest.mark.parametrize("records", [["missing.yaml"], [], ["a.yaml", "a.yaml"], [1]])
def test_complete_snapshot_rejects_corrupt_member_index(records):
    with pytest.raises(ValueError, match="capture snapshot"):
        fixture_disposition.capture_snapshot_members(
            json.dumps({"schema_version": 1, "records": records}).encode(), ["a.yaml"],
        )
