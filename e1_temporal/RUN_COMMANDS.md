# E1 Temporal Encoder Comparison — Run Commands

Pre-empts the anticipated reviewer question: *the IoT stream is a time series, so
why 54 summary statistics and an MLP rather than an LSTM?*

The paper now argues the case in Section III. This experiment settles it with a number.

## RUN OWNERSHIP

**The agent never launches official runs.** Everything below is executed by the user.
The agent ran only smoke checks (1,500 rows, 1 epoch) and deleted their outputs.

> No official temporal-encoder run was launched by the agent.

## Environment

Use `.venv-e1` (torch 2.11.0+cu128 with CUDA, sklearn, pandas). `.venv-gpu` lacks
scikit-learn and has no activation script.

```powershell
cd "G:\FOR THESIS\MAIN"
.\.venv-e1\Scripts\Activate.ps1
```

## What the source data actually looks like

Verified from the raw per-sample files, 2026-09-27:

|Property|Value|
|---|---|
|Variables|9 (temp, humidity, dew point; avg/max/min each)|
|Sampling interval|10 minutes|
|Window length|mostly 294 (~49 h) or 588 steps; full range 1 to 588|
|Missing cells|0.01%|
|Row order in file|**newest first** (descending timestamp)|

Two consequences the cache builder handles explicitly, and which are the reason a
naive LSTM on these files would be misleading:

- Feeding file order to a recurrent model reads time **backwards**. Windows are
  sorted to chronological order before caching.
- Window length is not independent of the label (healthy samples skew shorter:
  10th percentile 125 steps against 294 for early/mid). Windows are right-aligned
  on the most recent reading and front-padded, so length is not visible as a feature.
  Note that the published E1 already excludes length: `n_rows` is in `META_COLS`,
  not in the 54 features.

## Step 1: build the sequence cache (user executes)

```powershell
python e1_temporal\scripts\build_sequence_cache.py
```

Reads all 57,673 `<id>_final.csv` files and writes `cache/{train,val,test}_seq_L294.npz`
plus `classes.json`. One-off, roughly 10 to 30 minutes. Cleaning matches
`e1_iot_only/scripts/extract_iot_features.py` exactly, so the temporal models and the
control see the same underlying values.

At L=294 only 0.2% of train and 13.5% of test windows are truncated, and truncation
keeps the **most recent** readings.

## Step 2: train and compare (user executes)

```powershell
python e1_temporal\scripts\train_temporal_e1.py
```

|Model|Encoder|Params at default width|
|---|---|---|
|`mlp_stats`|54 summary statistics, the published E1 (**control**)|17,305|
|`gru`|GRU over the raw (294, 9) window|16,025|
|`lstm`|LSTM over the raw window|20,825|
|`tcn`|dilated 1D convolutions|45,657|

GRU and LSTM land naturally within range of the control. The TCN does not at the
shared default, so for a parameter-matched reading run it separately:

```powershell
python e1_temporal\scripts\train_temporal_e1.py --models tcn --hidden 36 --tag _matched
```

Same split, loss, optimiser, schedule, early-stopping rule and seed as the published
E1 run. Only the encoder changes.

## The control gate

`mlp_stats` must reproduce the published E1 Macro F1 of **0.883370**; the script
prints the delta and flags a mismatch. **If it does not reproduce, stop.** The
comparison is not trustworthy and nothing from the run belongs in the paper.

## Reading the result

- Compare on **Macro F1**. The dataset is severely imbalanced and accuracy hides it.
- Ignore severe-stage recall for ranking: 56 test samples, so one sample moves it
  1.79 points.
- **If temporal ≈ statistics**, the paper gains one sentence with a number and the
  question is closed permanently. This is the most likely outcome and the most useful.
- **If temporal clearly wins**, that is worth knowing before a reviewer finds it.
  It would not invalidate the fusion, bottleneck or robustness conclusions, which are
  about the fusion interface and hold for any sensor encoder, but E1 and every
  downstream embedding would need rebuilding, so weigh that against the deadline.
- Single seed. If the margin is small either way, rerun with `--seed 13 --tag _s13`
  and `--seed 73 --tag _s73` before making a claim about it.

## Outputs

Written to `e1_temporal/results/`:

- `temporal_summary.csv` — one row per encoder: params, accuracy, Macro F1,
  early-stage recall, severe-stage recall, ECE
- `e1_<model>_metrics.json` — full metric block including `per_class`
- `e1_<model>_test_predictions.csv`, `confusion_matrices/…`
