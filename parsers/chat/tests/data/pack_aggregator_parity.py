# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack the aggregator recorder's three runs into aggregator_parity.json.gz.

usage: python3 pack_aggregator_parity.py OUT.json.gz AUTO.jsonl V1.jsonl V2.jsonl

The runs hold the same cases in the same order. Inputs, options, and outputs
repeat a lot, so each is stored once in a table and a case refers to it by
index. A case is [name, options, input, output with the version unset,
output with version 1, output with version 2]; a version's output is null
when it equals the unset one. An output keeps only the response fields that
vary; the replay restores the fixed ones (see aggregator_parity.rs).
"""

import gzip
import json
import sys

FIXED = {"id": "chatcmpl-parity", "object": "chat.completion", "created": 1700000000, "model": "parity-model"}


def compact(output):
    if "ok" not in output:
        return output
    response = dict(output["ok"])
    for key, value in FIXED.items():
        if response.get(key) == value:
            del response[key]
    return {"ok": response}


def main():
    out, auto, v1, v2 = sys.argv[1:5]
    tables = {"inputs": {}, "options": {}, "outputs": {}}

    def index(table, value):
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        return tables[table].setdefault(key, len(tables[table]))

    cases = []
    with open(auto) as fa, open(v1) as f1, open(v2) as f2:
        for la, l1, l2 in zip(fa, f1, f2, strict=True):
            a, b, c = json.loads(la), json.loads(l1), json.loads(l2)
            assert a["name"] == b["name"] == c["name"]
            assert a["input"] == b["input"] == c["input"], a["name"]
            base = index("outputs", compact(a["output"]))
            cases.append([
                a["name"],
                index("options", a["options"]),
                index("inputs", a["input"]),
                base,
                None if b["output"] == a["output"] else index("outputs", compact(b["output"])),
                None if c["output"] == a["output"] else index("outputs", compact(c["output"])),
            ])
    blob = {name: [json.loads(key) for key in table] for name, table in tables.items()}
    blob["cases"] = cases
    data = json.dumps(blob, ensure_ascii=False, separators=(",", ":")).encode()
    with open(out, "wb") as raw, gzip.GzipFile("", "wb", 9, raw, mtime=0) as f:
        f.write(data)
    print(f"{len(cases)} cases, {len(blob['inputs'])} inputs, {len(blob['options'])} options, {len(blob['outputs'])} outputs")


if __name__ == "__main__":
    main()
