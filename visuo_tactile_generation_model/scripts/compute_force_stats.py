"""Compute per-channel mean/std for the force/tactile sequences referenced in
a metadata file (CSV/JSON/JSONL). Saves an .npz with keys `mean` and `std`,
both shape (D,), float32.

The output stats file is consumed by:
  * `WanTrainingModule(tactile_stats_path=...)` — normalizes `tactile_input`
    with `(x - mean) / std` right before the joint-denoise loss.
  * `TactileVideoInferenceDataset(stats_path=...)` — normalizes `tactile_init`
    and `tactile_gt` so validation runs in the same normalized space.

Channels with std < `min_std` (default 1e-6, e.g. dead/constant channels) are
clamped to `min_std` to avoid divide-by-zero downstream.

Usage:
    python3 scripts/compute_force_stats.py \\
        --metadata /path/to/metadata.csv \\
        --base /path/to/data_root \\
        --tactile-key force_sequence \\
        --output /path/to/force_stats.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np


def _load_metadata(path: str):
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in json metadata: {path}")
        return data
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve(base: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(base, value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--base", required=True, help="Base path to which relative npy paths in metadata are resolved")
    p.add_argument("--tactile-key", default="force_sequence")
    p.add_argument("--output", required=True, help="Output .npz file path")
    p.add_argument("--min-std", type=float, default=1e-6)
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick estimates")
    args = p.parse_args()

    rows = _load_metadata(args.metadata)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]

    # Streaming Welford-ish accumulators (sum + sumsq + count) per channel.
    n_total = 0
    sum_vec = None
    sumsq_vec = None
    skipped = 0
    used = 0

    for i, row in enumerate(rows):
        rel = row.get(args.tactile_key)
        if rel is None or (isinstance(rel, float) and math.isnan(rel)):
            skipped += 1
            continue
        path = _resolve(args.base, str(rel))
        if not os.path.exists(path):
            print(f"[warn] missing {path}, skip")
            skipped += 1
            continue
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[:, None]
        arr = arr.astype(np.float64)
        if sum_vec is None:
            D = arr.shape[1]
            sum_vec = np.zeros(D, dtype=np.float64)
            sumsq_vec = np.zeros(D, dtype=np.float64)
        elif arr.shape[1] != sum_vec.shape[0]:
            raise ValueError(
                f"channel-dim mismatch: file {path} has D={arr.shape[1]}, expected D={sum_vec.shape[0]}"
            )
        sum_vec += arr.sum(axis=0)
        sumsq_vec += (arr ** 2).sum(axis=0)
        n_total += arr.shape[0]
        used += 1
        if (i + 1) % 200 == 0:
            print(f"[progress] {i + 1}/{len(rows)} used={used} skipped={skipped} frames={n_total}")

    if n_total == 0 or sum_vec is None:
        raise RuntimeError("No usable tactile/force sequences found.")

    mean = sum_vec / n_total
    var = (sumsq_vec / n_total) - mean ** 2
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    std = np.clip(std, args.min_std, None)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, mean=mean, std=std, n_frames=np.int64(n_total), n_files=np.int64(used))
    print(f"[done] saved stats to {args.output}")
    print(f"  used files: {used}, total frames: {n_total}, channels: {mean.shape[0]}")
    print(f"  mean: {np.array2string(mean, precision=4)}")
    print(f"  std : {np.array2string(std, precision=4)}")


if __name__ == "__main__":
    main()
