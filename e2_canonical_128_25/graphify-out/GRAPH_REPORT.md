# Graph Report - e2_canonical_128_25  (2026-10-02)

## Corpus Check
- 3 files · ~1,388 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, .pt 1)

## Summary
- 27 nodes · 34 edges · 4 communities (2 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- main
- E2 Canonical 128-to-25 Variant
- E2Canonical128To25Head

## God Nodes (most connected - your core abstractions)
1. `main()` - 9 edges
2. `E2Canonical128To25Head` - 4 edges
3. `E2 Canonical 128-to-25 Variant` - 3 edges
4. `parse_args()` - 2 edges
5. `set_seed()` - 2 edges
6. `load_embedding_manifest()` - 2 edges
7. `load_split()` - 2 edges
8. `make_loader()` - 2 edges
9. `train_one_epoch()` - 2 edges
10. `predict()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `E2Canonical128To25Head`  [EXTRACTED]
  e2_canonical_128_25/scripts/train_e2_128_25.py → e2_canonical_128_25/scripts/train_e2_128_25.py  _Bridges community 3 → community 1_

## Import Cycles
- None detected.

## Communities (4 total, 2 thin omitted)

### Community 1 - "main"
Cohesion: 0.25
Nodes (8): load_embedding_manifest(), load_split(), main(), make_loader(), parse_args(), predict(), set_seed(), train_one_epoch()

### Community 2 - "E2 Canonical 128-to-25 Variant"
Cohesion: 0.50
Nodes (3): Architecture, E2 Canonical 128-to-25 Variant, Purpose

## Knowledge Gaps
- **2 isolated node(s):** `Architecture`, `Purpose`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 15 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `E2Canonical128To25Head` connect `E2Canonical128To25Head` to `train_e2_128_25.py`, `main`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `main()` connect `main` to `train_e2_128_25.py`, `E2Canonical128To25Head`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **What connects `Architecture`, `Purpose` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._