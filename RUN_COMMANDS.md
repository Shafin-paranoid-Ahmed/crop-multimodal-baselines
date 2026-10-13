# Current standalone runbook

Run from this repository root in native Windows PowerShell. The commands below
are examples for manual use; verification does not launch training or regenerate data.

## Environment

Create a Python 3.10 environment and select an appropriate PyTorch wheel for your machine.
The source environment used torch 2.11.0+cu128 and torchvision 0.26.0+cu128;
the portable requirements omit the CUDA suffix. The original pins are preserved in
`provenance/source-requirements.txt`.

```powershell
py -3.10 -m venv .venv
$PY = Join-Path $PWD '.venv/Scripts/python.exe'
& $PY -m pip install -r requirements.txt
& $PY scripts/verify.py
```

## Frozen embedding inputs

The Git checkout includes small checkpoints and CSV evidence. Large NPZ inputs
are excluded from Git and no embedding release is published. A fresh clone needs
the local embedding bundle and can restore it with:

```powershell
& $PY scripts/fetch_artifacts.py --from-archive '../crop-multimodal-baselines/.artifacts/embeddings-v1.zip'
& $PY scripts/verify.py --artifacts
```

Alternatively, pass `--from-directory` with the path to the prepared baseline checkout.
The helper refuses to overwrite differing artifacts and validates SHA-256. It copies
only paths listed in ARTIFACTS.json. The archive path is an example for this machine;
the archive and repositories can be relocated. Raw datasets are not in the bundle.
For quantum model checks, add `--quantum` to the verifier command after installing
the quantum requirements. `--download` remains disabled because embedding artifacts have not been published.

## Reproduction examples

Use new run names and output directories when rerunning. Do not overwrite archived
results or manually promote a new model to canonical without an explicit decision.
The validation split selects checkpoints; the test set is not a tuning target.

To use raw images, set `CROP_DATA_ROOT` to a directory containing `train/`, `val/`,
and `test/`. Fixed split manifests remain in this checkout's `Data/`.

```powershell
$env:CROP_DATA_ROOT = 'D:/datasets/crop-data'
& $PY e0_image_only/learning/train.py --help
& $PY e1_iot_only/scripts/train_e1_iot_mlp.py --help
& $PY e2_multimodal_fusion/scripts/extract_e2_embeddings.py --help
& $PY e2_multimodal_fusion/scripts/train_e2_fusion.py --help
& $PY e2_matched/scripts/train_matched.py --model e2_matched --bottleneck 8 --run-name reproduction_e2_matched_8 --help
& $PY e2_canonical_128_25/scripts/train_e2_128_25.py --help
& $PY e2_fusion_strategies/scripts/train_fusion_strategies.py --help
```

E1 feature extraction takes an explicit `--base-dir` for raw per-sample sensor files;
its existing feature CSVs are included. Temporal cache generation takes `--data-dir`;
that input directory must contain the approved manifests and per-sample files.
Neither feature/cache generation nor official temporal training was performed here.
The original model-local runbooks document scientific hyperparameters and should be
read before choosing actual training commands.
