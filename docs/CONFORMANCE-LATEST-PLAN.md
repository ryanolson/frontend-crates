# PR #470: latest captures and Unified-only conformance

## Baseline and preserved evidence

The base is `1b4a241fd73ee5f10795165139208fb3717956cc`. The latest store contains 77 complete versionless YAML files across 12 families. Each family/implementation file contains complete records, sealed request bindings, and original producer provenance separate from selection metadata. Stream, batch, and jail-buffer identities remain distinct; missing measurements remain explicit.

Direct Git-blob comparison preserves all 779 baseline evidence paths: 689 versioned capture YAMLs, 12 authored input/GOLDEN documents, 76 archives, and two Beam YAMLs. Existing evidence stays at its original path. Backfill, versioned-v2 removal, historical moves, archive expansion, and new compression remain deferred.

## Active workflow

Unified v2 is the only active conformance report and executor. Deprecated batch, stream, batch-on-stream, and Reasoning generators, runners, renderers, and tests are retired. Existing Dynamo v1 observations measured against Unified requests remain selectable comparisons. The retained split scenarios preserve their IDs and observations; the old archive-import writer is retired. Parser-library APIs and unrelated workspace checks are unchanged.

The viewer has seven runtime owners plus the existing DOM/detail/coloring boundaries. Development YAML stays outside production. One Node qualification CLI owns status, model, snapshot, and marker checks; six Node suites replace 25 test files. Shared Python owners cover request normalization, peer compatibility, inventory/coverage, migrations, and evidence verification. Historical readers and independent publication validation remain separate from production publishing.

## Publication contract

Default output is one self-contained `index.html`, with latest data and downloads embedded. Full history is a user option to add sidecar JSON beside byte-identical HTML. HTTP viewing automatically discovers a matching history publication. Missing sidecars leave latest usable; local files make zero external reads. Historical results load on demand, preserve request/provenance bindings, and cannot overwrite a newer selection.

Reuse complete publications before staging, then validated family fragments before conversion. Latest generation and parser tests must pass with history reads and network sockets denied, and generation must preserve capture YAML. Warm conversion reuse, independent family invalidation, historical-only isolation, and corrupt-cache recovery remain required.

## Measured consolidation

The comparison starts at delivered head `85f7287628cfd4a8bc1e5506493b313540d0214a`. Added file counts are measured against the base and exclude generated build dependencies. Removing inherited files can increase changed-path counts even while reducing maintained files and code.

| Measurement | Before | Candidate |
|---|---:|---:|
| Added JavaScript files, including tests/vendor | 65 | 20 |
| Added Python files, including tests/support | 65 | 39 |
| Standalone HTML | 8,207,592 B | 8,161,378 B |
| Default generated files | 1 | 1 |
| Full-history generated files | 899 | 899 |
| Full-history bytes | 39,390,303 B | 39,297,499 B |
| Chromium median display, three paired trials | 2.866 s | 2.945 s |

Display measurements use cleared browser/origin caches, the same 2,796-cell matrix, a 1440x1000 viewport, and 20 MB/s transfer. Rows and columns compare equal. No display-speed improvement is claimed. Candidate latest cold/warm generation measured 10.49/9.62 seconds; history measured 49.16/56.98 seconds under concurrent validation load. Warm runs reused 12 latest and 24 history family fragments with zero conversions. These are local measurements, separate from hosted CI.

## Verification status

The candidate passed 1,841 Python tests with 27 skips, 924 v2 parser unit tests, 109 Rust conformance tests, and 72 publisher tests with two ignored. Six Node suites passed 19 named cases. Full Chromium passed 111 tests. Latest publication and both latest parser/conformance cases passed with historical reads and IPv4/IPv6 sockets denied; capture hashes stayed unchanged. Independent review found no unresolved findings and compared 40,662 latest/history model records without differences. Final CI collection accounts for 1,951 unique latest tests across 1,965 assignments and 1,991 history tests across 2,026 assignments, each with eight qualification shards. The final CI-owner regression suite passed 68 tests. Exact-tree qualification, signing, delivery, and hosted current-head CI are recorded in the final delivery receipt; pending hosted checks remain pending.
