# Graph Report - e1_temporal  (2026-10-02)

## Corpus Check
- 3 files · ~2,742 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 57 nodes · 77 edges · 8 communities (4 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- E1 Temporal Encoder Comparison — Run Commands
- RecurrentEncoder
- main
- build_split
- load_window
- load_stats_splits

## God Nodes (most connected - your core abstractions)
1. `main()` - 9 edges
2. `E1 Temporal Encoder Comparison — Run Commands` - 9 edges
3. `RecurrentEncoder` - 5 edges
4. `TCN` - 5 edges
5. `load_window()` - 4 edges
6. `build_split()` - 4 edges
7. `build_model()` - 4 edges
8. `find_sample_dir()` - 3 edges
9. `main()` - 3 edges
10. `load_stats_splits()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `build_split()` --calls--> `load_window()`  [EXTRACTED]
  e1_temporal/scripts/build_sequence_cache.py → e1_temporal/scripts/build_sequence_cache.py  _Bridges community 6 → community 5_
- `main()` --calls--> `build_model()`  [EXTRACTED]
  e1_temporal/scripts/train_temporal_e1.py → e1_temporal/scripts/train_temporal_e1.py  _Bridges community 3 → community 4_
- `main()` --calls--> `load_stats_splits()`  [EXTRACTED]
  e1_temporal/scripts/train_temporal_e1.py → e1_temporal/scripts/train_temporal_e1.py  _Bridges community 7 → community 4_

## Import Cycles
- None detected.

## Communities (8 total, 4 thin omitted)

### Community 2 - "E1 Temporal Encoder Comparison — Run Commands"
Cohesion: 0.20
Nodes (9): E1 Temporal Encoder Comparison — Run Commands, Environment, Outputs, Reading the result, RUN OWNERSHIP, Step 1: build the sequence cache (user executes), Step 2: train and compare (user executes), The control gate (+1 more)

### Community 3 - "RecurrentEncoder"
Cohesion: 0.25
Nodes (3): build_model(), RecurrentEncoder, TCN

### Community 4 - "main"
Cohesion: 0.29
Nodes (7): load_seq_splits(), main(), make_loader(), parse_args(), predict(), run_epoch(), set_seed()

### Community 5 - "build_split"
Cohesion: 0.40
Nodes (4): build_split(), find_sample_dir(), main(), parse_args()

## Knowledge Gaps
- **8 isolated node(s):** `RUN OWNERSHIP`, `Environment`, `What the source data actually looks like`, `Step 1: build the sequence cache (user executes)`, `Step 2: train and compare (user executes)` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 29 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RecurrentEncoder` connect `RecurrentEncoder` to `train_temporal_e1.py`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Why does `TCN` connect `RecurrentEncoder` to `train_temporal_e1.py`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Why does `load_window()` connect `load_window` to `build_sequence_cache.py`, `build_split`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **What connects `RUN OWNERSHIP`, `Environment`, `What the source data actually looks like` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._