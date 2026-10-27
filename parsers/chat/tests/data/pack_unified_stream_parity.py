# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack the two recorder runs into one fixture.

Usage: pack_unified_stream_parity.py OUT.jsonl.gz AUTO.jsonl V2.jsonl

The fixture holds every case from the run with DYN_PARSER_VERSION unset, then
each case from the DYN_PARSER_VERSION=2 run whose output differs.
"""

import gzip
import json
import sys

out_path, auto_path, v2_path = sys.argv[1:4]
with open(auto_path) as f:
    auto = f.readlines()
with open(v2_path) as f:
    v2 = f.readlines()
lines = list(auto)
for a, b in zip(auto, v2, strict=True):
    first, second = json.loads(a), json.loads(b)
    assert first["name"] == second["name"] and first["input"] == second["input"]
    if first["output"] != second["output"]:
        lines.append(b)
with gzip.GzipFile(out_path, "wb", compresslevel=9, mtime=0) as f:
    f.write("".join(lines).encode())
