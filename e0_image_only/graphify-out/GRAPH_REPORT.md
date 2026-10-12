# Graph Report - e0_image_only  (2026-10-02)

## Corpus Check
- 37 files · ~12,092 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .csv 18, .pth 7)

## Summary
- 150 nodes · 305 edges · 13 communities (9 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evaluate_ood_corruptions.py
- before/split_train_val.py
- mobilenetv3.py
- augment_train_images.py
- train.py
- check_split_distribution.py
- transform_images.py
- train_epoch
- Path
- SplitCSVImageDataset
- save_predictions
- canonical/README.md
- legacy/README.md

## God Nodes (most connected - your core abstractions)
1. `main()` - 17 edges
2. `MobileNetV3Classifier` - 12 edges
3. `train_epoch()` - 10 edges
4. `predict()` - 9 edges
5. `SplitCSVImageDataset` - 7 edges
6. `TestImageDataset` - 6 edges
7. `main()` - 6 edges
8. `save_predictions()` - 6 edges
9. `main()` - 6 edges
10. `main()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `MobileNetV3Classifier`  [EXTRACTED]
  e0_image_only/learning/evaluate_ood_corruptions.py → e0_image_only/model/mobilenetv3.py
- `main()` --calls--> `MobileNetV3Classifier`  [EXTRACTED]
  e0_image_only/learning/train.py → e0_image_only/model/mobilenetv3.py

## Import Cycles
- None detected.

## Communities (13 total, 4 thin omitted)

### Community 0 - "evaluate_ood_corruptions.py"
Cohesion: 0.12
Nodes (9): corrupt(), evaluate(), find_image_path(), main(), normalize(), parse_args(), prepare(), read_manifest() (+1 more)

### Community 1 - "before/split_train_val.py"
Cohesion: 0.16
Nodes (12): copy_item(), detect_columns(), main(), read_csv(), stratified_split(), write_csv(), copy_item(), detect_columns() (+4 more)

### Community 3 - "augment_train_images.py"
Cohesion: 0.23
Nodes (8): augment_image(), collect_images(), has_transformed_suffix(), is_image_file(), main(), parse_suffixes(), process_image(), write_image()

### Community 4 - "train.py"
Cohesion: 0.24
Nodes (8): build_label_map(), configure_cuda_runtime(), count_total_params(), main(), parse_args(), read_manifest(), set_seed(), unfreeze_last_feature_blocks()

### Community 5 - "check_split_distribution.py"
Cohesion: 0.23
Nodes (6): detect_label_field(), main(), print_comparison(), read_counts(), main(), read_counts()

### Community 6 - "transform_images.py"
Cohesion: 0.30
Nodes (6): collect_images(), is_image_file(), main(), output_path(), process_image(), write_image()

### Community 7 - "train_epoch"
Cohesion: 0.31
Nodes (7): apply_gpu_augmentation(), class_weights_from_rows(), format_progress(), normalize_batch(), predict(), prepare_batch(), train_epoch()

### Community 8 - "Path"
Cohesion: 0.53
Nodes (4): count_missing_images(), find_image_path(), save_data_audit(), write_csv_rows()

### Community 10 - "save_predictions"
Cohesion: 0.50
Nodes (3): label_to_stage(), summarize(), save_predictions()

## Knowledge Gaps
- **2 isolated node(s):** `Canonical E0 image-only run`, `E0 Legacy Archive`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 34 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MobileNetV3Classifier` connect `mobilenetv3.py` to `evaluate_ood_corruptions.py`, `train.py`?**
  _High betweenness centrality (0.148) - this node is a cross-community bridge._
- **Why does `SplitCSVImageDataset` connect `SplitCSVImageDataset` to `Path`, `save_predictions`, `train.py`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **What connects `Canonical E0 image-only run`, `E0 Legacy Archive` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `evaluate_ood_corruptions.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11666666666666667 - nodes in this community are weakly interconnected._
- **Should `mobilenetv3.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13071895424836602 - nodes in this community are weakly interconnected._