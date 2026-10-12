# Extraction changes

- New repository entrypoints, dependency records, attribution, source manifests, artifact manifest/fetcher and standalone verifier. Raw datasets, caches, credentials, runtime logs, unrelated tracks and smoke/benchmark-only outputs excluded.

- Small required helpers are frozen copies; metrics, splits, checkpoint tensors, prediction CSVs and embedding arrays are preserved byte-for-byte. Missing historical Markdown links resolve to source-commit permalinks when their targets exist in the parent repository.

- Large frozen embedding arrays are local-only and distributed once through the private baseline embeddings-v1 release. Each consumer checks its required file hashes.

- Classical matched-head trainer extracted from train_bottleneck.py with the quantum class/branch/import removed; classical algorithms and checkpoint keys unchanged. Raw E0/image-extraction data paths configurable with CROP_DATA_ROOT; fixed split paths remain local. Legacy E0 preprocessing prototypes are excluded.

- Retained the original matched-4 loader separately in e2_matched/legacy_four.py because its projection.proj parameter names differ from the generalized matched trainer. Checkpoint bytes were not changed.
