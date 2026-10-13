# Implementation notes

- Small shared helpers are versioned copies; metrics, splits, checkpoint tensors, prediction CSVs and embedding arrays are kept byte-for-byte as produced by the recorded runs.

- Large frozen embedding arrays are kept in a local bundle (`.artifacts/embeddings-v1.zip`); no release is published. Each consumer checks its required file hashes.

- The classical matched-head trainer is `train_bottleneck.py` with the quantum class, branch and import removed; classical algorithms and checkpoint keys are unchanged. Raw E0 image data paths are configurable with `CROP_DATA_ROOT`; fixed split paths remain local. Legacy E0 preprocessing prototypes are not included.

- The original matched-4 loader is kept separately in `e2_matched/legacy_four.py` because its `projection.proj` parameter names differ from the generalized matched trainer. Checkpoint bytes were not changed.
