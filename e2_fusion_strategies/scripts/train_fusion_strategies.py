"""Compare fusion operators on the frozen E0/E1 embeddings.

Reviewer point (iCACCESS 2026): "The proposed fusion method (concatenation + MLP)
is a relatively simple architecture, and no comparison against stronger fusion
strategies is provided to establish its competitiveness."

This trains several fusion heads on the SAME frozen embeddings, the SAME fixed
split, the SAME weighted-CE loss, optimiser, schedule and seed, so the only thing
that varies is the fusion operator. Every head is also reported with its trainable
parameter count, because a fusion operator that wins by spending more parameters
has not beaten concatenation on fusion.

Heads:
  concat_mlp  1344 -> 128 -> 25                      (the paper's E2-128-25; control)
  concat_b8   1344 -> 128 -> 8 -> 8 -> 25            (the paper's E2-matched-8; control)
  gated       Gated Multimodal Unit (Arevalo et al., ICLR-W 2017)
  attention   cross-modal multi-head attention over the two modality tokens
  bilinear    low-rank bilinear pooling of the two projected modalities
  film        sensor branch modulates the image branch (FiLM, Perez et al. 2018)

Usage is in e2_fusion_strategies/RUN_COMMANDS.md. Runs are launched by the user.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.thesis_eval import (  # noqa: E402
    count_trainable_params,
    metrics_dict,
    per_class_metrics,
    save_confusion_matrix,
    save_predictions,
)

FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"
EXPECTED_SPLIT_COUNTS = {"train": 46713, "val": 5193, "test": 5767}
IMG_DIM = 1280
IOT_DIM = 64


# --------------------------------------------------------------------------- heads
class ConcatMLP(nn.Module):
    """The paper's unconstrained head: concatenate, then one hidden layer."""

    def __init__(self, hidden=128, num_classes=25, bottleneck=None):
        super().__init__()
        layers = [
            nn.Linear(IMG_DIM + IOT_DIM, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        ]
        if bottleneck is None:
            layers.append(nn.Linear(hidden, num_classes))
        else:
            layers += [
                nn.Linear(hidden, bottleneck),
                nn.Tanh(),
                nn.Linear(bottleneck, bottleneck, bias=False),
                nn.Tanh(),
                nn.Linear(bottleneck, num_classes),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, img, iot):
        return self.net(torch.cat([img, iot], dim=1))


class GatedFusion(nn.Module):
    """Gated Multimodal Unit: a learned per-dimension gate mixes the two modalities."""

    def __init__(self, dim=128, num_classes=25):
        super().__init__()
        self.hi = nn.Linear(IMG_DIM, dim)
        self.ht = nn.Linear(IOT_DIM, dim)
        self.gate = nn.Linear(IMG_DIM + IOT_DIM, dim)
        self.norm = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(dim, num_classes)

    def forward(self, img, iot):
        hi = torch.tanh(self.hi(img))
        ht = torch.tanh(self.ht(iot))
        z = torch.sigmoid(self.gate(torch.cat([img, iot], dim=1)))
        return self.out(self.drop(self.norm(z * hi + (1.0 - z) * ht)))


class AttentionFusion(nn.Module):
    """Each modality becomes one token; multi-head attention lets them attend."""

    def __init__(self, dim=128, heads=4, num_classes=25):
        super().__init__()
        self.pi = nn.Linear(IMG_DIM, dim)
        self.pt = nn.Linear(IOT_DIM, dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(dim, num_classes)

    def forward(self, img, iot):
        tok = torch.stack([self.pi(img), self.pt(iot)], dim=1)   # (B, 2, dim)
        att, _ = self.attn(tok, tok, tok)
        return self.out(self.drop(self.norm(tok + att).mean(dim=1)))


class BilinearFusion(nn.Module):
    """Low-rank bilinear pooling: elementwise product of the two projections."""

    def __init__(self, dim=128, num_classes=25):
        super().__init__()
        self.pi = nn.Linear(IMG_DIM, dim)
        self.pt = nn.Linear(IOT_DIM, dim)
        self.norm = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(dim, num_classes)

    def forward(self, img, iot):
        z = self.pi(img) * self.pt(iot)
        z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8)      # signed sqrt
        z = F.normalize(z, dim=1)
        return self.out(self.drop(self.norm(z)))


class FiLMFusion(nn.Module):
    """The sensor branch predicts a scale and shift applied to the image branch."""

    def __init__(self, dim=128, num_classes=25):
        super().__init__()
        self.pi = nn.Linear(IMG_DIM, dim)
        self.film = nn.Linear(IOT_DIM, 2 * dim)
        self.norm = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(dim, num_classes)

    def forward(self, img, iot):
        gamma, beta = self.film(iot).chunk(2, dim=1)
        z = F.relu((1.0 + gamma) * self.pi(img) + beta)
        return self.out(self.drop(self.norm(z)))


def build_head(name, num_classes, dim=128, heads=4):
    """dim is the fusion width. Leave it at 128 for the like-for-like comparison;
    lower it for the heavier operators to match the ~175k parameter budget."""
    if name == "concat_mlp":
        return ConcatMLP(dim, num_classes, bottleneck=None)
    if name == "concat_b8":
        return ConcatMLP(dim, num_classes, bottleneck=8)
    if name == "gated":
        return GatedFusion(dim, num_classes)
    if name == "attention":
        return AttentionFusion(dim, heads, num_classes)
    if name == "bilinear":
        return BilinearFusion(dim, num_classes)
    if name == "film":
        return FiLMFusion(dim, num_classes)
    raise ValueError("unknown head: %s" % name)


HEADS = ["concat_mlp", "concat_b8", "gated", "attention", "bilinear", "film"]


# --------------------------------------------------------------------------- data
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(feature_dir, split_name, smoke=False):
    path = feature_dir / ("%s_embeddings.npz" % split_name)
    if not path.exists():
        raise FileNotFoundError("Missing %s; run extract_e2_embeddings.py first" % path)
    data = np.load(path, allow_pickle=False)
    img = data["image_embedding"].astype(np.float32)
    iot = data["iot_embedding"].astype(np.float32)
    y = data["y"].astype(np.int64)
    meta = pd.DataFrame({
        "image": data["image_id"].astype(str),
        "label": data["label"].astype(str),
        "stage": data["stage"].astype(str),
        "label_idx": y,
    })
    expected = EXPECTED_SPLIT_COUNTS[split_name]
    if len(y) != expected:
        raise RuntimeError("%s row count %d != expected %d" % (split_name, len(y), expected))
    if img.shape[1] != IMG_DIM or iot.shape[1] != IOT_DIM:
        raise RuntimeError("%s embedding dims %d/%d != expected %d/%d"
                           % (split_name, img.shape[1], iot.shape[1], IMG_DIM, IOT_DIM))
    if np.isnan(img).any() or np.isnan(iot).any():
        raise RuntimeError("%s embeddings contain NaN values" % split_name)
    if smoke:
        n = min(2048, len(y))
        img, iot, y, meta = img[:n], iot[:n], y[:n], meta.iloc[:n].reset_index(drop=True)
    return img, iot, y, meta


def make_loader(img, iot, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(img), torch.tensor(iot), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot, n = 0.0, 0
    for img, iot, y in loader:
        img, iot, y = img.to(device), iot.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(img, iot), y)
        loss.backward()
        optimizer.step()
        tot += loss.item() * len(y)
        n += len(y)
    return tot / max(n, 1)


def predict(model, loader, criterion, device):
    model.eval()
    tot, n = 0.0, 0
    T, P, R = [], [], []
    with torch.no_grad():
        for img, iot, y in loader:
            img, iot, y = img.to(device), iot.to(device), y.to(device)
            logits = model(img, iot)
            tot += criterion(logits, y).item() * len(y)
            n += len(y)
            probs = torch.softmax(logits, dim=1)
            T.append(y.cpu().numpy())
            P.append(probs.argmax(dim=1).cpu().numpy())
            R.append(probs.cpu().numpy())
    return {"loss": tot / max(n, 1), "targets": np.concatenate(T),
            "preds": np.concatenate(P), "probs": np.concatenate(R)}


# --------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description="Compare fusion operators on frozen embeddings")
    p.add_argument("--heads", nargs="+", default=HEADS, choices=HEADS)
    p.add_argument("--feature-dir", default=str(FEATURE_DIR))
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    p.add_argument("--dim", type=int, default=128,
                   help="fusion width; lower it for gated/attention to match ~175k params")
    p.add_argument("--heads-attn", type=int, default=4, help="attention heads")
    p.add_argument("--tag", default="", help="suffix for run names, e.g. _matched")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler-patience", type=int, default=3)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="2048 rows, 1 epoch: wiring check only, never a reportable result")
    return p.parse_args()


def main():
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(feature_dir / "e2_embedding_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    classes = manifest["classes"]

    epochs = 1 if args.smoke else args.epochs
    tr = load_split(feature_dir, "train", args.smoke)
    va = load_split(feature_dir, "val", args.smoke)
    te = load_split(feature_dir, "test", args.smoke)

    counts = np.bincount(tr[2], minlength=len(classes))
    weights = torch.tensor(len(tr[2]) / (len(classes) * np.maximum(counts, 1)),
                           dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader = make_loader(tr[0], tr[1], tr[2], args.batch_size, True)
    val_loader = make_loader(va[0], va[1], va[2], args.batch_size, False)
    test_loader = make_loader(te[0], te[1], te[2], args.batch_size, False)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    summary = []
    for head in args.heads:
        set_seed(args.seed)                      # identical init stream per head
        run_name = "fusion_%s%s" % (head, args.tag)
        model = build_head(head, len(classes), args.dim, args.heads_attn).to(device)
        n_params = count_trainable_params(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.scheduler_factor,
            patience=args.scheduler_patience)

        ckpt = checkpoint_dir / ("best_%s.pt" % run_name)
        best_f1, best_epoch, waited, history = -1.0, 0, 0, []
        print("\n=== %s (%d trainable params, device=%s) ===" % (head, n_params, device))

        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
            vm = metrics_dict(predict(model, val_loader, criterion, device), classes)
            scheduler.step(vm["macro_f1"])
            history.append({"epoch": epoch, "train_loss": float(train_loss),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "val_loss": vm["loss"], "val_accuracy": vm["accuracy"],
                            "val_macro_f1": vm["macro_f1"], "val_ece": vm["ece"]})
            print("  epoch %3d  train_loss %.4f  val_macro_f1 %.4f" % (epoch, train_loss, vm["macro_f1"]))
            if vm["macro_f1"] > best_f1:
                best_f1, best_epoch, waited = vm["macro_f1"], epoch, 0
                torch.save(model.state_dict(), ckpt)
            else:
                waited += 1
                if waited >= args.patience:
                    print("  early stop at epoch %d" % epoch)
                    break

        model.load_state_dict(torch.load(ckpt, map_location=device))
        test_out = predict(model, test_loader, criterion, device)
        tm = metrics_dict(test_out, classes)
        tm["per_class"] = per_class_metrics(test_out, classes)

        pd.DataFrame(history).to_csv(results_dir / ("%s_history.csv" % run_name), index=False)
        save_predictions(results_dir / ("%s_test_predictions.csv" % run_name), te[3], test_out, classes)
        save_confusion_matrix(
            results_dir / "confusion_matrices" / ("%s_test_confusion_matrix.csv" % run_name),
            test_out, classes)
        payload = {"run_name": run_name, "head": head, "dim": args.dim, "smoke": bool(args.smoke),
                   "seed": args.seed, "device": str(device), "best_epoch": best_epoch,
                   "trainable_params": n_params, "optimizer": "Adam",
                   "loss": "weighted_cross_entropy", "lr": args.lr,
                   "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                   "train_rows": int(len(tr[2])), "val_rows": int(len(va[2])),
                   "test_rows": int(len(te[2])), "classes": classes, "test": tm}
        with open(results_dir / ("%s_metrics.json" % run_name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        summary.append({"head": head, "params": n_params, "test_accuracy": tm["accuracy"],
                        "test_macro_f1": tm["macro_f1"],
                        "early_stage_recall": tm["early_stage_recall"],
                        "severe_stage_recall": tm["per_stage_recall"].get("3"),
                        "test_ece": tm["ece"]})
        print("  TEST  acc %.4f  macroF1 %.4f  ECE %.4f" % (tm["accuracy"], tm["macro_f1"], tm["ece"]))

    df = pd.DataFrame(summary).sort_values("test_macro_f1", ascending=False)
    out = results_dir / ("fusion_strategy_summary%s%s.csv"
                         % (args.tag, "_SMOKE" if args.smoke else ""))
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print("\nwrote %s" % out)
    if args.smoke:
        print("SMOKE RUN - wiring check only. These numbers are NOT reportable.")


if __name__ == "__main__":
    main()
