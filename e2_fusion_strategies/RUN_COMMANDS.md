# E2 Fusion-Strategy Comparison — Run Commands

Answers iCACCESS 2026 reviewer point 3: *"The proposed fusion method (concatenation +
MLP) is a relatively simple architecture, and no comparison against stronger fusion
strategies is provided to establish its competitiveness."*

## RUN OWNERSHIP

**The agent never launches official runs.** Everything below is executed by the user.
The agent has run only the smoke check (wiring validation, 2,048 rows, 1 epoch), whose
output is written to `fusion_strategy_summary_SMOKE.csv` and is **not reportable**.

> No official fusion-strategy run was launched by the agent.

## Environment: use `.venv-e1`

Verified 2026-09-26:

| Environment | Python | torch | CUDA | sklearn | Usable |
|---|---|---|---|---|---|
| `.venv-e1` | 3.10.20 | 2.11.0+cu128 | yes (RTX 5070) | 1.7.2 | **yes** |
| `.venv-gpu` | 3.12.10 | 2.11.0+cu128 | yes | **missing** | no |

`.venv-gpu` also has no `Activate.ps1` — its `Scripts\` folder contains only
`python.exe`, so it cannot be activated the usual way. `common/thesis_eval.py` imports
scikit-learn, so `.venv-gpu` cannot run this script as-is. Use `.venv-e1`, which already
has CUDA.

```powershell
cd "G:\FOR THESIS\MAIN"
.\.venv-e1\Scripts\Activate.ps1
```

Or skip activation and call the interpreter directly, which works from any shell state:

```powershell
& "G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe" <script> <args>
```

## What this compares

All heads consume the identical frozen embeddings (1280-d image from E0, 64-d IoT from
E1), the identical fixed split, weighted cross-entropy, Adam, LR schedule, early-stopping
rule and seed. The only thing that varies is the fusion operator.

| Head | Operator |
|---|---|
| `concat_mlp` | concatenate → 128 → 25 (the paper's E2-128-25; **control**) |
| `concat_b8` | concatenate → 128 → 8 → 8 → 25 (the paper's E2-matched-8; **control**) |
| `gated` | Gated Multimodal Unit (Arevalo et al., ICLR-W 2017) |
| `attention` | cross-modal multi-head attention over two modality tokens |
| `bilinear` | low-rank bilinear pooling (signed-sqrt + L2) |
| `film` | sensor branch modulates image branch (FiLM, Perez et al. 2018) |

The two controls must reproduce the published numbers (0.9734 Macro F1 for `concat_mlp`,
0.9734 for `concat_b8`). **If they do not, stop** — the comparison is not trustworthy and
nothing from the run should go in the paper.

## Official runs (user executes)

Pass 1 — like-for-like, every head at fusion width 128:

```powershell
cd "G:\FOR THESIS\MAIN"
.\.venv-e1\Scripts\Activate.ps1
python e2_fusion_strategies\scripts\train_fusion_strategies.py
```

Pass 2 — parameter-matched. At width 128 the gated head costs 347,929 parameters and
attention 241,817, against 175,641 for concatenation, so a win in pass 1 could just be
extra capacity. Re-run the two heavy heads narrow enough to land near 176k:

```powershell
python e2_fusion_strategies\scripts\train_fusion_strategies.py `
    --heads gated attention --dim 56 --tag _matched
```

Check the printed parameter counts and adjust `--dim` until they sit within roughly
±10% of 175,641. Report pass 2 as the headline comparison and pass 1 as the
unconstrained-capacity variant.

Runtime: these are small heads over cached embeddings — minutes per head on the RTX 5070.
Add `--cpu` to force CPU if the GPU is busy.

## Outputs

Written to `e2_fusion_strategies/results/`:

- `fusion_strategy_summary.csv` — one row per head: params, accuracy, Macro F1,
  early-stage recall, severe-stage recall, ECE
- `fusion_<head>_metrics.json` — full metric block including `per_class`
- `fusion_<head>_test_predictions.csv`, `confusion_matrices/…`

## Reading the result honestly

- Compare on **Macro F1**, not accuracy — the dataset is severely imbalanced.
- Severe-stage recall rests on 56 test samples, so one sample moves it 1.79 points.
  Do not rank heads on severe-stage recall.
- If a stronger operator wins at a matched budget, the paper's claim changes from
  "fusion beats unimodal" to "fusion beats unimodal, and operator choice matters" —
  which is a better paper, not a worse one.
- If concatenation holds up, that is the direct answer to the reviewer, and it should be
  stated with the competing numbers in the table rather than asserted.
