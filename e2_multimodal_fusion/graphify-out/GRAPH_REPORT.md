# Graph Report - e2_multimodal_fusion  (2026-10-02)

## Corpus Check
- 8 files · ~3,073 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 16 file(s) not represented in the graph (top: .csv 11, (none) 2, .npz 2)

## Summary
- 62 nodes · 91 edges · 11 communities (5 shown, 6 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- E2 Multimodal Fusion
- main
- main
- SplitImageDataset
- extract_image_embeddings
- extract_iot_embeddings
- E2FusionHead
- canonical/README.md

## God Nodes (most connected - your core abstractions)
1. `main()` - 10 edges
2. `main()` - 9 edges
3. `E2 Multimodal Fusion` - 9 edges
4. `SplitImageDataset` - 6 edges
5. `extract_image_embeddings()` - 5 edges
6. `extract_iot_embeddings()` - 4 edges
7. `E2FusionHead` - 4 edges
8. `load_manifest()` - 3 edges
9. `parse_args()` - 2 edges
10. `set_seed()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `extract_image_embeddings()` --calls--> `SplitImageDataset`  [EXTRACTED]
  e2_multimodal_fusion/scripts/extract_e2_embeddings.py → e2_multimodal_fusion/scripts/extract_e2_embeddings.py  _Bridges community 5 → community 6_
- `main()` --calls--> `load_manifest()`  [EXTRACTED]
  e2_multimodal_fusion/scripts/extract_e2_embeddings.py → e2_multimodal_fusion/scripts/extract_e2_embeddings.py  _Bridges community 7 → community 4_
- `main()` --calls--> `extract_image_embeddings()`  [EXTRACTED]
  e2_multimodal_fusion/scripts/extract_e2_embeddings.py → e2_multimodal_fusion/scripts/extract_e2_embeddings.py  _Bridges community 6 → community 4_
- `main()` --calls--> `E2FusionHead`  [EXTRACTED]
  e2_multimodal_fusion/scripts/train_e2_fusion.py → e2_multimodal_fusion/scripts/train_e2_fusion.py  _Bridges community 8 → community 3_

## Import Cycles
- None detected.

## Communities (11 total, 6 thin omitted)

### Community 1 - "E2 Multimodal Fusion"
Cohesion: 0.20
Nodes (9): Canonical Artifacts, Data Split, E2 Multimodal Fusion, Final Run, Fusion Head, Interpretation, Selected Encoders, Workflow (+1 more)

### Community 3 - "main"
Cohesion: 0.25
Nodes (8): load_embedding_manifest(), load_split(), main(), make_loader(), parse_args(), predict(), set_seed(), train_one_epoch()

### Community 4 - "main"
Cohesion: 0.29
Nodes (7): build_e0_model(), load_classes(), main(), parse_args(), save_split(), set_seed(), validate_manifest_counts()

### Community 6 - "extract_image_embeddings"
Cohesion: 0.67
Nodes (3): e0_embedding(), extract_image_embeddings(), prepare_images()

### Community 7 - "extract_iot_embeddings"
Cohesion: 0.67
Nodes (3): extract_iot_embeddings(), load_manifest(), select_iot_features()

## Knowledge Gaps
- **9 isolated node(s):** `Selected Encoders`, `Data Split`, `Workflow`, `Fusion Head`, `Final Run` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 23 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SplitImageDataset` connect `SplitImageDataset` to `extract_e2_embeddings.py`, `extract_image_embeddings`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `E2FusionHead` connect `E2FusionHead` to `train_e2_fusion.py`, `main`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **What connects `Selected Encoders`, `Data Split`, `Workflow` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._