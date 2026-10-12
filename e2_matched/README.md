# Classical matched bottlenecks

`scripts/train_matched.py` is the classical-only extraction of the generalized
bottleneck trainer. It supports dimensions 4 and 8 without PennyLane.

The frozen original matched-4 checkpoint uses `projection.proj.*` parameter keys.
Load it with `E2MatchedHead` in `legacy_four.py`, copied from the original Spec-B
model. The later generalized trainer and matched-8 checkpoint use `projection.net.*`.
The preserved checkpoints have not been renamed internally or rewritten.

The paired quantum repository retains the original matched-4 training command
alongside E3-4. Baseline `train_matched.py --bottleneck 4` is the later generalized
training implementation; it is not a claim of an exact rerun of the original run.
