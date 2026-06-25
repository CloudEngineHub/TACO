"""Inject `tactile_input` into existing video-only cache .pth files in place,
so they become equivalent to a joint-denoise cache without re-running T5 / VAE.

Mapping logic mirrors `accelerate.prepare(dataloader)` with `shuffle=False` +
default DistributedSampler:
  total_size       = ceil(N / world_size) * world_size
  padded_indices   = list(range(N)) + list(range(total_size - N))
  rank R sees      padded_indices[R::world_size]

For each pth file `<rank>/<i>.pth`, we therefore look up
`records[padded_indices[R::world_size][i]]` from the expanded
`context_metadata_*.json` to find the source `.npy` path.

Tactile resampling uses the same `np.linspace + round` scheme as
`visuo_tactile_world_model/world_model/dataset/operators.py:LoadNumpyArray`, so the injected tensor is
byte-identical to what `sft_joint_denoise:data_process` would have written.

Usage:
    python3 scripts/inject_tactile_into_cache.py \\
        --cache-root  /path/to/latent_cache_ti2v_5b_joint_nf37_s1 \\
        --data-base   /path/to/wan_worldrl_stride5_192x256 \\
        --world-size  8 \\
        --num-frames  37
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Optional

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True, help="Directory containing rank subdirs (0/, 1/, ...) and .context_window/")
    p.add_argument("--data-base", required=True, help="Base path to which `force_sequence` paths in metadata are relative")
    p.add_argument("--world-size", type=int, required=True, help="World size used at cache time (number of rank subdirs)")
    p.add_argument("--num-frames", type=int, default=37, help="Target tactile length (must match training num_frames)")
    p.add_argument("--tactile-key", default="force_sequence", help="Field name in metadata records pointing to .npy")
    p.add_argument("--dry-run", action="store_true", help="Only print mapping summary, do not modify any pth")
    p.add_argument("--overwrite", action="store_true", help="Re-inject even if `tactile_input` already present")
    p.add_argument("--drop-missing", action="store_true",
                   help="Delete pth files whose source record has a missing/NaN tactile path")
    return p.parse_args()


def _slice_window(arr, context_start, context_window_size, pad_last):
    """Mirror LoadVideo / LoadNumpyArray window slicing semantics."""
    total = arr.shape[0]
    if context_window_size is None or total <= 0:
        return arr
    size = int(context_window_size)
    start = max(0, int(context_start))
    if start >= total:
        start = total - 1
    end = start + size
    if end <= total:
        return arr[start:end]
    if pad_last:
        head = arr[start:total]
        pad = np.repeat(arr[total - 1:total], end - total, axis=0)
        return np.concatenate([head, pad], axis=0)
    fallback_start = max(0, total - size)
    return arr[fallback_start:total]


def load_tactile(
    npy_path: str,
    num_frames: int,
    context_start: int = 0,
    context_window_size: Optional[int] = None,
    pad_last: bool = False,
) -> torch.Tensor:
    arr = np.load(npy_path)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = _slice_window(arr, context_start, context_window_size, pad_last)
    if arr.shape[0] != num_frames:
        total = arr.shape[0]
        idx = np.linspace(0, max(total - 1, 0), num=num_frames).round().astype(np.int64)
        idx = np.clip(idx, 0, total - 1)
        arr = arr[idx]
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
    if t.ndim == 2:
        t = t.unsqueeze(0)  # (1, T, D)
    return t


def find_records_json(cache_root: str) -> str:
    candidates = glob.glob(os.path.join(cache_root, ".context_window", "context_metadata_*.json"))
    if not candidates:
        sys.exit(f"[FATAL] No context_metadata_*.json under {cache_root}/.context_window/")
    if len(candidates) > 1:
        sys.exit(f"[FATAL] Multiple context_metadata_*.json found, ambiguous: {candidates}")
    return candidates[0]


def main():
    args = parse_args()

    json_path = find_records_json(args.cache_root)
    with open(json_path) as f:
        records = json.load(f)
    N = len(records)
    W = args.world_size
    total_size = math.ceil(N / W) * W
    padded = list(range(N)) + list(range(total_size - N))
    print(f"[Info] Loaded {N} records from {json_path}")
    print(f"[Info] world_size={W}, total_size={total_size}, padding={total_size - N}")

    grand_injected = 0
    grand_skipped_existing = 0
    grand_missing_npy = 0
    grand_dropped = 0
    grand_total = 0

    for rank in range(W):
        rank_indices = padded[rank::W]
        rank_dir = os.path.join(args.cache_root, str(rank))
        if not os.path.isdir(rank_dir):
            sys.exit(f"[FATAL] Missing rank dir: {rank_dir}")
        pth_files = sorted(
            [p for p in os.listdir(rank_dir) if re.fullmatch(r"\d+\.pth", p)],
            key=lambda x: int(x.split(".")[0]),
        )
        if len(pth_files) != len(rank_indices):
            sys.exit(
                f"[FATAL] rank {rank}: {len(pth_files)} pth files but expected {len(rank_indices)}.\n"
                "  This usually means some samples were skipped at cache time (broken/invalid).\n"
                "  Round-robin index recovery is no longer reliable; please re-run cache."
            )

        injected = skipped_existing = missing_npy = dropped = 0
        for i, fname in enumerate(pth_files):
            pth_path = os.path.join(rank_dir, fname)
            record = records[rank_indices[i]]
            sample = torch.load(pth_path, map_location="cpu", weights_only=False)
            inputs_shared = sample[0]

            if (not args.overwrite) and ("tactile_input" in inputs_shared):
                skipped_existing += 1
                continue

            npy_field = record.get(args.tactile_key)
            # Detect "no tactile for this sample" — None, NaN (CSV empty cell),
            # or any non-string/non-dict value. Drop the pth if --drop-missing.
            def _is_missing(v):
                if v is None:
                    return True
                if isinstance(v, float) and math.isnan(v):
                    return True
                if isinstance(v, str) and v.strip() == "":
                    return True
                return not isinstance(v, (str, dict))

            if _is_missing(npy_field):
                if args.drop_missing and not args.dry_run:
                    os.remove(pth_path)
                    dropped += 1
                else:
                    missing_npy += 1
                continue

            # Side-stream may be a bare path (legacy) or a context-window dict
            # (new schema_version=2). Pull window info either from the field
            # itself or from the sibling `video` dict.
            if isinstance(npy_field, dict):
                npy_rel = npy_field["path"]
                ctx_start = int(npy_field.get("context_start", 0))
                ctx_size = npy_field.get("context_window_size", None)
                pad_last = bool(npy_field.get("pad_last", False))
            else:
                npy_rel = npy_field
                video_field = record.get("video")
                if isinstance(video_field, dict):
                    ctx_start = int(video_field.get("context_start", 0))
                    ctx_size = video_field.get("context_window_size", None)
                    pad_last = bool(video_field.get("pad_last", False))
                else:
                    ctx_start, ctx_size, pad_last = 0, None, False

            npy_path = os.path.join(args.data_base, npy_rel)
            if not os.path.isfile(npy_path):
                if args.drop_missing and not args.dry_run:
                    os.remove(pth_path)
                    dropped += 1
                else:
                    missing_npy += 1
                    if missing_npy <= 5:
                        print(f"[WARN][rank{rank}] Missing npy on disk: {npy_path}")
                continue

            tactile = load_tactile(
                npy_path,
                args.num_frames,
                context_start=ctx_start,
                context_window_size=ctx_size,
                pad_last=pad_last,
            )
            inputs_shared["tactile_input"] = tactile

            if not args.dry_run:
                torch.save(sample, pth_path)
            injected += 1

            if (injected + skipped_existing + missing_npy) % 500 == 0:
                print(f"[rank{rank}] progress {i + 1}/{len(pth_files)} (injected={injected}, kept={skipped_existing}, missing={missing_npy})")

        print(
            f"[rank{rank}] DONE total={len(pth_files)} "
            f"injected={injected} kept_existing={skipped_existing} "
            f"missing_npy={missing_npy} dropped={dropped}"
        )
        grand_injected += injected
        grand_skipped_existing += skipped_existing
        grand_missing_npy += missing_npy
        grand_dropped += dropped
        grand_total += len(pth_files)

    print("=" * 60)
    print(f"[Summary] dry_run={args.dry_run}")
    print(f"[Summary] total pth        = {grand_total}")
    print(f"[Summary] injected tactile = {grand_injected}")
    print(f"[Summary] kept existing    = {grand_skipped_existing}")
    print(f"[Summary] missing npy      = {grand_missing_npy}")
    print(f"[Summary] dropped pth      = {grand_dropped}")


if __name__ == "__main__":
    main()
