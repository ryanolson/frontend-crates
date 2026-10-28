# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact the recorder's JSONL into name-indexed rows.
Usage: compact.py OUT auto.jsonl 1.jsonl 2.jsonl"""
import json, sys
values, index = [], {}
def iv(v):
    key = json.dumps(v, sort_keys=True)
    if key not in index:
        index[key] = len(values); values.append(v)
    return index[key]
prefills, pindex = [], {}
DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'
def ip(v):
    key = json.dumps(v)
    if key not in pindex:
        pindex[key] = len(prefills); prefills.append(v)
    assert len(prefills) <= len(DIGITS)
    return DIGITS[pindex[key]]
def bits(xs): return ''.join('1' if x else '0' for x in xs)
recs = [json.loads(l) for p in sys.argv[2:] for l in open(p)]
names = [None] + sorted({r['r'] for r in recs if r['k'] == 'pair' and r['r'] is not None})
ni = {n: i for i, n in enumerate(names)}
by = {}
for r in recs:
    by.setdefault(r['k'], []).append(r)
out = []
for version in ['Auto', 'V1', 'V2']:
    rows = {}
    for r in by['selected']:
        if r['v'] == version:
            rows.setdefault(r['t'], {})[r['r']] = r
    for t in names:
        row = rows[t]
        out.append(['selected', version, ni[t], [iv(row[r]['family']) for r in names], [iv(row[r]['special']) for r in names]])
pairs = {}
for r in by['pair']:
    pairs.setdefault(r['t'], {})[r['r']] = r
for t in names:
    row = pairs[t]
    out.append(['pair', ni[t],
                [iv(row[r]['configured']) for r in names],
                bits(row[r]['is_v2'] for r in names),
                [bits(row[r]['start_for_parsers']) for r in names],
                [''.join(ip(p) for p in row[r]['prefill']) for r in names],
                [[iv(x) for m in row[r]['tool_request_mode'] for x in m] for r in names]])
for r in by['route']:
    out.append(['route', r['v'], ni[r['t']], ni[r['r']], [iv(x) for x in r['out']]])
for r in by['prompt_start']:
    out.append(['prompt_start', ni[r['r']], bits(r['out'])])
for r in by['tool_parser']:
    out.append(['tool_parser', ni[r['t']], bits(r['native']), [iv(x) for x in r['jail']],
                [[h['input'], [iv(x) for x in h['out']]] for h in r['hidden']]])
for r in by['name']:
    out.append(['name', r['name'], r['canonical']])
for k in ['route_requests', 'family_names', 'version_parse', 'tools', 'stripped']:
    for r in by[k]:
        out.append([k, {kk: vv for kk, vv in r.items() if kk != 'k'}])
prompts = by['prompts'][0]['prompts']
with open(sys.argv[1], 'w') as f:
    for head in (['values', values], ['prefills', prefills], ['names', names], ['prompts', prompts]):
        f.write(json.dumps(head, separators=(',', ':')) + '\n')
    for row in out:
        f.write(json.dumps(row, separators=(',', ':')) + '\n')
