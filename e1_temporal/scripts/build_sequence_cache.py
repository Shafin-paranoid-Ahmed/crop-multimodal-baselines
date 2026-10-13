"""Cache the raw IoT sequences so temporal models can be trained on them.

Answers the anticipated reviewer question: the sensor stream is a time series,
so why summary statistics and an MLP rather than a recurrent encoder?

Three properties of the source files matter and are handled explicitly here:

1. Rows are stored NEWEST FIRST. A recurrent model fed the file order would read
   time backwards, so every window is sorted to chronological order.
2. Window length varies (1 to 588 readings). Windows are right-aligned on the
   most recent reading and front-padded, with a mask marking real steps, so a
   model never sees padding as data and never sees length as a feature.
3. Cleaning matches e1_iot_only/scripts/extract_iot_features.py exactly
   (coerce to numeric, forward fill, back fill, median, then zero), so the
   temporal models and the E1 control see the same underlying values.

Runs are launched by the author. See e1_temporal/RUN_COMMANDS.md.
"""
import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "Data"
CACHE_DIR = EXPERIMENT_ROOT / "cache"

NUMERIC_COLS = [
    "int_temp_1_avg", "int_temp_1_max", "int_temp_1_min",
    "int_hum_1_avg", "int_hum_1_max", "int_hum_1_min",
    "int_dew_point_avg", "int_dew_point_max", "int_dew_point_min",
]
EXPECTED = {"train": 46713, "val": 5193, "test": 5767}


def parse_args():
    p = argparse.ArgumentParser(description="Cache IoT sequences for temporal models")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out-dir", default=str(CACHE_DIR))
    p.add_argument("--length", type=int, default=294,
                   help="steps kept, right-aligned on the most recent reading "
                        "(294 x 10 min = 49 h, the dominant window length)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None, help="smoke: first N rows only")
    return p.parse_args()


def find_sample_dir(data_dir, image_id):
    """val ids live under Data/train/ because val was split out of the train set."""
    for sub in ("train", "test"):
        d = data_dir / sub / image_id
        if d.is_dir():
            return d
    return None


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def load_window(path, length):
    """Return (length, 9) chronological array plus a validity mask."""
    with io.open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stamped = []
    for r in rows:
        try:
            t = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, KeyError, TypeError):
            continue
        stamped.append((t, r))
    stamped.sort(key=lambda x: x[0])           # oldest -> newest
    rows = [r for _, r in stamped]

    cols = []
    for c in NUMERIC_COLS:
        raw = []
        for r in rows:
            v = r.get(c, "")
            try:
                raw.append(float(v))
            except (TypeError, ValueError):
                raw.append(None)
        # forward fill, then back fill, then median, then zero
        last = None
        for i, v in enumerate(raw):
            if v is None:
                raw[i] = last
            else:
                last = v
        nxt = None
        for i in range(len(raw) - 1, -1, -1):
            if raw[i] is None:
                raw[i] = nxt
            else:
                nxt = raw[i]
        present = [v for v in raw if v is not None]
        fill = median(present) if present else 0.0
        cols.append([fill if v is None else v for v in raw])

    n = len(rows)
    x = np.zeros((length, len(NUMERIC_COLS)), dtype=np.float32)
    mask = np.zeros((length,), dtype=bool)
    if n:
        keep = min(n, length)
        block = np.asarray(cols, dtype=np.float32).T[-keep:]   # most recent `keep`
        x[length - keep:] = block                              # right-aligned
        mask[length - keep:] = True
    return x, mask, n


def build_split(data_dir, out_dir, split, length, limit):
    manifest = list(csv.DictReader(io.open(data_dir / (split + ".csv"), encoding="utf-8")))
    if limit:
        manifest = manifest[:limit]
    total = len(manifest)
    if not limit and total != EXPECTED[split]:
        raise RuntimeError("%s manifest has %d rows, expected %d"
                           % (split, total, EXPECTED[split]))

    labels = sorted({r["label"] for r in csv.DictReader(
        io.open(data_dir / "train.csv", encoding="utf-8"))})
    index = {l: i for i, l in enumerate(labels)}

    X = np.zeros((total, length, len(NUMERIC_COLS)), dtype=np.float32)
    M = np.zeros((total, length), dtype=bool)
    y = np.zeros((total,), dtype=np.int64)
    ids, raw_len = [], []

    print("[%s] %d samples" % (split, total))
    for i, item in enumerate(manifest):
        image_id = str(item["image"])
        d = find_sample_dir(data_dir, image_id)
        if d is None:
            raise FileNotFoundError("no sample directory for %s" % image_id)
        X[i], M[i], n = load_window(d / (image_id + "_final.csv"), length)
        y[i] = index[item["label"]]
        ids.append(image_id)
        raw_len.append(n)
        if (i + 1) % 5000 == 0 or i + 1 == total:
            print("[%s] %d/%d" % (split, i + 1, total))

    if not np.isfinite(X).all():
        raise RuntimeError("%s cache contains non-finite values" % split)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("%s_seq_L%d.npz" % (split, length))
    np.savez_compressed(out, X=X, mask=M, y=y,
                        image_id=np.array(ids),
                        raw_length=np.array(raw_len, dtype=np.int32))
    print("[%s] wrote %s  X%s  median raw length %d  truncated %d (%.1f%%)"
          % (split, out, X.shape, int(np.median(raw_len)),
             int((np.array(raw_len) > length).sum()),
             100.0 * (np.array(raw_len) > length).mean()))
    return labels


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    labels = None
    for split in args.splits:
        labels = build_split(data_dir, out_dir, split, args.length, args.limit)
    with io.open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print("wrote %s (%d classes)" % (out_dir / "classes.json", len(labels)))


if __name__ == "__main__":
    sys.exit(main())
