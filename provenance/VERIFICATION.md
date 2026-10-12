# Extraction verification — 2026-10-12

Local standalone checks passed. The user selected local-only retention; no remote is configured.

- Canonical E0/E1/E2 checkpoints load and synthetic forward checks pass.
- Classical matched-4/8 checkpoints and backward checks pass; their post-bottleneck parameter counts are 16 and 64.
- Canonical saved Macro-F1/accuracy recomputed for all three baselines.
- All 5,767 canonical E2 TEST predictions reproduced from frozen embeddings.
- Classical checks passed with PennyLane imports deliberately blocked.
- Eight selected training/extraction entrypoints passed --help.

All copied source-file hashes were checked against the current parent checkout.
Scientific CSV/JSON/NPZ/checkpoint artifacts remain byte-identical. The extraction
manifest lists deliberate code/documentation adaptations. Python syntax and fixed
split counts passed. All local Markdown file targets resolve. A targeted credential
pattern scan found no matches; caches and environment/session files are excluded.

Checks used the existing Python 3.10.20 research interpreter, torch 2.11.0+cu128,
torchvision 0.26.0+cu128, scikit-learn 1.7.2 and PennyLane 0.42.3. Source imports were
restricted to this checkout; the original thesis code is not a runtime dependency.
No fresh dependency installation, full retraining, raw-data transfer, full corruption
inference matrix or new scientific experiment was performed. Synthetic quantum checks
used CPU default.qubit; they are not a simulator performance benchmark.

Git preserves original file bytes, including historical line endings. Whitespace
checks cover newly written root documentation/scripts rather than rewriting frozen
research CSVs. Large embeddings are checksummed local artifacts and a local archive, excluded from Git.

Fresh local-clone and local artifact-restore checks are the final acceptance step.
