"""Does a temporal encoder beat the 54 summary statistics on the IoT channel?

The E1 baseline in the paper summarises each sensor window into 54 fixed
statistics and classifies them with a small MLP. Since the underlying stream is
a regular 10-minute time series, a reviewer can reasonably ask why it was not
modelled recurrently. This script answers with a number.

Every model sees the same split, loss, optimiser, schedule, early-stopping rule
and seed as the published E1 run; only the sensor encoder changes.

  mlp_stats  54 summary statistics -> 128 -> 64 -> 25   (the paper's E1; CONTROL)
  gru        GRU over the raw (294, 9) window
  lstm       LSTM over the raw window
  tcn        dilated 1D convolutions over the raw window

The control must reproduce the published E1 Macro F1 of 0.883370. If it does
not, the comparison is not trustworthy and nothing from the run belongs in the
paper.

Runs are launched by the user. See e1_temporal/RUN_COMMANDS.md.
"""
import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.models import E1MLP  # noqa: E402
from common.thesis_eval import (  # noqa: E402
    count_trainable_params,
    metrics_dict,
    per_class_metrics,
    save_confusion_matrix,
    save_predictions,
)

CACHE_DIR = EXPERIMENT_ROOT / "cache"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"
STATS_DIR = PROJECT_ROOT / "e1_iot_only" / "features"
DATA_DIR = PROJECT_ROOT / "Data"
META_COLS = ["image", "label", "crop", "disease", "stage", "n_rows"]
N_CHANNELS = 9
PUBLISHED_E1_MACRO_F1 = 0.8833699043245444
SEQ_MODELS = ("gru", "lstm", "tcn")
ALL_MODELS = ("mlp_stats",) + SEQ_MODELS


# ------------------------------------------------------------------ encoders
class RecurrentEncoder(nn.Module):
    """Windows are front-padded and right-aligned, so the final step is always
    real data and the last hidden state is a valid summary."""

    def __init__(self, kind="gru", hidden=64, layers=1, num_classes=25):
        super().__init__()
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn(N_CHANNELS, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class TCN(nn.Module):
    """Dilated causal-ish 1D convolutions, then global average pooling."""

    def __init__(self, channels=32, num_classes=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_CHANNELS, channels, 5, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=8, dilation=4), nn.ReLU(),
        )
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return self.head(z.mean(dim=2))


def build_model(name, num_classes, hidden, layers):
    if name == "mlp_stats":
        # the shared definition, so the control cannot drift from published E1
        return E1MLP(input_dim=54, num_classes=num_classes, hidden1=128, hidden2=64)
    if name in ("gru", "lstm"):
        return RecurrentEncoder(name, hidden, layers, num_classes)
    if name == "tcn":
        return TCN(hidden, num_classes)
    raise ValueError("unknown model: %s" % name)


# ------------------------------------------------------------------ data
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_stats_splits(classes):
    """Rebuild the exact E1 inputs from the cached 54-feature tables."""
    train_all = pd.read_csv(STATS_DIR / "train_iot_features.csv")
    test_df = pd.read_csv(STATS_DIR / "test_iot_features.csv")
    feature_cols = [c for c in train_all.columns if c not in META_COLS]
    if len(feature_cols) != 54:
        raise RuntimeError("expected 54 feature columns, found %d" % len(feature_cols))

    want = {}
    for split in ("train", "val"):
        ids = pd.read_csv(DATA_DIR / (split + ".csv"))["image"].astype(str)
        want[split] = set(ids)
    train_all["image"] = train_all["image"].astype(str)
    test_df["image"] = test_df["image"].astype(str)

    tr = train_all[train_all["image"].isin(want["train"])].reset_index(drop=True)
    va = train_all[train_all["image"].isin(want["val"])].reset_index(drop=True)

    scaler = StandardScaler().fit(tr[feature_cols])
    index = {c: i for i, c in enumerate(classes)}
    out = {}
    for name, df in (("train", tr), ("val", va), ("test", test_df)):
        x = scaler.transform(df[feature_cols]).astype(np.float32)
        y = df["label"].map(index).to_numpy(dtype=np.int64)
        meta = pd.DataFrame({"image": df["image"], "label": df["label"],
                             "stage": df["stage"].astype(str), "label_idx": y})
        out[name] = (x, y, meta)
    return out


def load_seq_splits(cache_dir, length, classes):
    out, stats = {}, None
    for split in ("train", "val", "test"):
        p = cache_dir / ("%s_seq_L%d.npz" % (split, length))
        if not p.exists():
            raise FileNotFoundError("%s missing; run build_sequence_cache.py first" % p)
        d = np.load(p, allow_pickle=False)
        X, mask, y = d["X"].astype(np.float32), d["mask"], d["y"].astype(np.int64)
        if split == "train":
            valid = X[mask]                       # real steps only
            stats = (valid.mean(axis=0), valid.std(axis=0) + 1e-6)
        X = (X - stats[0]) / stats[1]
        X[~mask] = 0.0                            # padding stays neutral after scaling
        labels = [classes[i] for i in y]
        meta = pd.DataFrame({"image": d["image_id"].astype(str), "label": labels,
                             "stage": [l.split("_")[-1] for l in labels], "label_idx": y})
        out[split] = (X, y, meta)
    return out


def make_loader(x, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(x), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        tot += loss.item() * len(y)
        n += len(y)
    return tot / max(n, 1)


def predict(model, loader, criterion, device):
    model.eval()
    tot = n = 0
    T, P, R = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tot += criterion(logits, y).item() * len(y)
            n += len(y)
            probs = torch.softmax(logits, dim=1)
            T.append(y.cpu().numpy())
            P.append(probs.argmax(dim=1).cpu().numpy())
            R.append(probs.cpu().numpy())
    return {"loss": tot / max(n, 1), "targets": np.concatenate(T),
            "preds": np.concatenate(P), "probs": np.concatenate(R)}


# ------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser(description="Temporal vs summary-statistic IoT encoders")
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    p.add_argument("--cache-dir", default=str(CACHE_DIR))
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    p.add_argument("--length", type=int, default=294)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler-patience", type=int, default=3)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--tag", default="")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch: wiring check only, never a reportable result")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir, checkpoint_dir = Path(args.results_dir), Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    with io.open(cache_dir / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    epochs = 1 if args.smoke else args.epochs

    needs_seq = any(m in SEQ_MODELS for m in args.models)
    seq = load_seq_splits(cache_dir, args.length, classes) if needs_seq else None
    stats = load_stats_splits(classes) if "mlp_stats" in args.models else None

    summary = []
    for name in args.models:
        data = stats if name == "mlp_stats" else seq
        set_seed(args.seed)
        run_name = "e1_%s%s" % (name, args.tag)

        counts = np.bincount(data["train"][1], minlength=len(classes))
        weights = torch.tensor(len(data["train"][1]) / (len(classes) * np.maximum(counts, 1)),
                               dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        train_loader = make_loader(*data["train"][:2], args.batch_size, True)
        val_loader = make_loader(*data["val"][:2], args.batch_size, False)
        test_loader = make_loader(*data["test"][:2], args.batch_size, False)

        model = build_model(name, len(classes), args.hidden, args.layers).to(device)
        n_params = count_trainable_params(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.scheduler_factor,
            patience=args.scheduler_patience)

        ckpt = checkpoint_dir / ("best_%s.pt" % run_name)
        best, best_epoch, waited, history = -1.0, 0, 0, []
        print("\n=== %s (%d trainable params, device=%s) ===" % (name, n_params, device))

        for epoch in range(1, epochs + 1):
            tl = run_epoch(model, train_loader, criterion, optimizer, device)
            vm = metrics_dict(predict(model, val_loader, criterion, device), classes)
            scheduler.step(vm["macro_f1"])
            history.append({"epoch": epoch, "train_loss": float(tl),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"]})
            print("  epoch %3d  train_loss %.4f  val_macro_f1 %.4f" % (epoch, tl, vm["macro_f1"]))
            if vm["macro_f1"] > best:
                best, best_epoch, waited = vm["macro_f1"], epoch, 0
                torch.save(model.state_dict(), ckpt)
            else:
                waited += 1
                if waited >= args.patience:
                    print("  early stop at epoch %d" % epoch)
                    break

        model.load_state_dict(torch.load(ckpt, map_location=device))
        out = predict(model, test_loader, criterion, device)
        tm = metrics_dict(out, classes)
        tm["per_class"] = per_class_metrics(out, classes)

        pd.DataFrame(history).to_csv(results_dir / ("%s_history.csv" % run_name), index=False)
        save_predictions(results_dir / ("%s_test_predictions.csv" % run_name),
                         data["test"][2], out, classes)
        save_confusion_matrix(results_dir / "confusion_matrices" /
                              ("%s_test_confusion_matrix.csv" % run_name), out, classes)
        payload = {"run_name": run_name, "model": name, "smoke": bool(args.smoke),
                   "seed": args.seed, "device": str(device), "best_epoch": best_epoch,
                   "trainable_params": n_params, "optimizer": "Adam",
                   "loss": "weighted_cross_entropy", "lr": args.lr,
                   "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                   "window_length": args.length if name != "mlp_stats" else None,
                   "classes": classes, "test": tm}
        with io.open(results_dir / ("%s_metrics.json" % run_name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if name == "mlp_stats" and not args.smoke:
            delta = tm["macro_f1"] - PUBLISHED_E1_MACRO_F1
            flag = "OK" if abs(delta) < 5e-3 else "MISMATCH - investigate before reporting"
            print("  CONTROL: published %.6f, rerun %.6f, delta %+.6f  [%s]"
                  % (PUBLISHED_E1_MACRO_F1, tm["macro_f1"], delta, flag))

        summary.append({"model": name, "params": n_params,
                        "test_accuracy": tm["accuracy"], "test_macro_f1": tm["macro_f1"],
                        "early_stage_recall": tm["early_stage_recall"],
                        "severe_stage_recall": tm["per_stage_recall"].get("3"),
                        "test_ece": tm["ece"]})
        print("  TEST  acc %.4f  macroF1 %.4f  ECE %.4f" % (tm["accuracy"], tm["macro_f1"], tm["ece"]))

    df = pd.DataFrame(summary).sort_values("test_macro_f1", ascending=False)
    out_csv = results_dir / ("temporal_summary%s%s.csv"
                             % (args.tag, "_SMOKE" if args.smoke else ""))
    df.to_csv(out_csv, index=False)
    print("\n" + df.to_string(index=False))
    print("\nwrote %s" % out_csv)
    if args.smoke:
        print("SMOKE RUN - wiring check only. These numbers are NOT reportable.")


if __name__ == "__main__":
    sys.exit(main())
