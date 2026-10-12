# Graph Report - e2_fusion_strategies  (2026-10-02)

## Corpus Check
- 8 files · ~4,275 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 20 file(s) not represented in the graph (top: .csv 20)

## Summary
- 51 nodes · 66 edges · 7 communities (3 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- build_head
- E2 Fusion-Strategy Comparison — Run Commands
- main
- AttentionFusion
- BilinearFusion
- GatedFusion

## God Nodes (most connected - your core abstractions)
1. `build_head()` - 8 edges
2. `main()` - 8 edges
3. `E2 Fusion-Strategy Comparison — Run Commands` - 7 edges
4. `ConcatMLP` - 5 edges
5. `GatedFusion` - 5 edges
6. `AttentionFusion` - 5 edges
7. `BilinearFusion` - 5 edges
8. `FiLMFusion` - 5 edges
9. `set_seed()` - 2 edges
10. `load_split()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `build_head()` --calls--> `GatedFusion`  [EXTRACTED]
  e2_fusion_strategies/scripts/train_fusion_strategies.py → e2_fusion_strategies/scripts/train_fusion_strategies.py  _Bridges community 6 → community 1_
- `build_head()` --calls--> `AttentionFusion`  [EXTRACTED]
  e2_fusion_strategies/scripts/train_fusion_strategies.py → e2_fusion_strategies/scripts/train_fusion_strategies.py  _Bridges community 4 → community 1_
- `build_head()` --calls--> `BilinearFusion`  [EXTRACTED]
  e2_fusion_strategies/scripts/train_fusion_strategies.py → e2_fusion_strategies/scripts/train_fusion_strategies.py  _Bridges community 5 → community 1_
- `main()` --calls--> `build_head()`  [EXTRACTED]
  e2_fusion_strategies/scripts/train_fusion_strategies.py → e2_fusion_strategies/scripts/train_fusion_strategies.py  _Bridges community 1 → community 3_

## Import Cycles
- None detected.

## Communities (7 total, 4 thin omitted)

### Community 1 - "build_head"
Cohesion: 0.22
Nodes (3): build_head(), ConcatMLP, FiLMFusion

### Community 2 - "E2 Fusion-Strategy Comparison — Run Commands"
Cohesion: 0.25
Nodes (7): E2 Fusion-Strategy Comparison — Run Commands, Environment: use `.venv-e1`, Official runs (user executes), Outputs, Reading the result honestly, RUN OWNERSHIP, What this compares

### Community 3 - "main"
Cohesion: 0.29
Nodes (7): load_split(), main(), make_loader(), parse_args(), predict(), run_epoch(), set_seed()

## Knowledge Gaps
- **6 isolated node(s):** `RUN OWNERSHIP`, `Environment: use `.venv-e1``, `What this compares`, `Official runs (user executes)`, `Outputs` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 31 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ConcatMLP` connect `build_head` to `train_fusion_strategies.py`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `GatedFusion` connect `GatedFusion` to `train_fusion_strategies.py`, `build_head`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `AttentionFusion` connect `AttentionFusion` to `train_fusion_strategies.py`, `build_head`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **What connects `RUN OWNERSHIP`, `Environment: use `.venv-e1``, `What this compares` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_fusion_strategies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._