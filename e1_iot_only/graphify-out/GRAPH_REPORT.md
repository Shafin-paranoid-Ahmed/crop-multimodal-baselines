# Graph Report - e1_iot_only  (2026-10-02)

## Corpus Check
- 14 files · ~3,911 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 49 file(s) not represented in the graph (top: .csv 41, .pt 8)

## Summary
- 57 nodes · 86 edges · 7 communities (5 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- E1: IoT-Only Baseline
- generate_shared_splits.py
- evaluate_e1_checkpoint.py
- extract_iot_features.py
- main
- README.md

## God Nodes (most connected - your core abstractions)
1. `E1: IoT-Only Baseline` - 16 edges
2. `main()` - 9 edges
3. `main()` - 6 edges
4. `select_by_split()` - 4 edges
5. `make_loader()` - 4 edges
6. `predict()` - 4 edges
7. `make_feature_row()` - 3 edges
8. `extract_split()` - 3 edges
9. `main()` - 3 edges
10. `main()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `make_loader()`  [EXTRACTED]
  e1_iot_only/scripts/evaluate_e1_checkpoint.py → e1_iot_only/scripts/train_e1_iot_mlp.py
- `main()` --calls--> `predict()`  [EXTRACTED]
  e1_iot_only/scripts/evaluate_e1_checkpoint.py → e1_iot_only/scripts/train_e1_iot_mlp.py
- `main()` --calls--> `select_by_split()`  [EXTRACTED]
  e1_iot_only/scripts/evaluate_e1_checkpoint.py → e1_iot_only/scripts/train_e1_iot_mlp.py

## Import Cycles
- None detected.

## Communities (7 total, 2 thin omitted)

### Community 0 - "E1: IoT-Only Baseline"
Cohesion: 0.12
Nodes (16): Current Status Note, Dataset Audit Summary, Dataset Inputs, E1 Folder Structure, E1: IoT-Only Baseline, Feature Extraction Strategy, Interpretation, Label Format (+8 more)

### Community 2 - "generate_shared_splits.py"
Cohesion: 0.32
Nodes (3): main(), parse_args(), save_label_counts()

### Community 3 - "evaluate_e1_checkpoint.py"
Cohesion: 0.52
Nodes (6): load_csv(), main(), parse_args(), make_loader(), predict(), select_by_split()

### Community 4 - "extract_iot_features.py"
Cohesion: 0.60
Nodes (5): clean_numeric_data(), extract_split(), main(), make_feature_row(), parse_args()

### Community 5 - "main"
Cohesion: 0.33
Nodes (6): load_csv(), main(), parse_args(), save_predictions(), set_seed(), train_one_epoch()

## Knowledge Gaps
- **16 isolated node(s):** `Current Status Note`, `Purpose`, `Dataset Inputs`, `Label Format`, `Dataset Audit Summary` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 26 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `main` to `train_e1_iot_mlp.py`, `evaluate_e1_checkpoint.py`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **What connects `Current Status Note`, `Purpose`, `Dataset Inputs` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `E1: IoT-Only Baseline` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._