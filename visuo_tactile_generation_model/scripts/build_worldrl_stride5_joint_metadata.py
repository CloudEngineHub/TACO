#!/usr/bin/env python3
"""Build TACO joint-denoise metadata CSV for World-RL stride5 layout.

Each episode directory must contain: video.mp4 and force.npy (shape (T, 12)).
Paths in the CSV are relative to dataset_base_path (the stride5 root).

The training config expects:
  * force_sequence as a relative .npy path
  * num_frames = full-episode length (used by context-window expansion; matches twist_beat CSV)
  * video relative path

This layout has no first-frame image.jpg; training/cache yaml should set
  data_file_keys: video,force_sequence
  extra_inputs: input_image,force_sequence
because WanTrainingModule maps input_image from data[\"video\"][0].

Usage:
  python3 scripts/build_worldrl_stride5_joint_metadata.py \\
    --stride5-root <DATA_ROOT> \\
    --output <METADATA_CSV>
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import struct
import sys


def _npy_shape(path: str) -> tuple[int, ...]:
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"not a .npy file: {path}")
        major, minor = f.read(2)
        if major == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
        elif major == 2:
            hlen = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"unsupported numpy format version {major}.{minor}")
        hdr = f.read(hlen)
    s = hdr.decode("latin1").strip("\x00 \n")
    if s.endswith(","):
        s = s[:-1]
    meta = ast.literal_eval(s)
    return tuple(meta["shape"])


TASK_PROMPTS = {
    "beat_xylophone": "beat xylophone",
    "pick_bread_twice": "pick bread twice",
    "twist_bottle_cap": "twist bottle cap",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stride5-root",
        required=True,
        help="Directory containing task subfolders (beat_xylophone, ...).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <stride5-root>/metadata_taco_joint.csv).",
    )
    p.add_argument(
        "--tasks",
        default="beat_xylophone,pick_bread_twice,twist_bottle_cap",
        help="Comma-separated task folder names under stride5-root.",
    )
    args = p.parse_args()
    root = os.path.abspath(args.stride5_root)
    out = args.output or os.path.join(root, "metadata_taco_joint.csv")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    rows: list[dict] = []
    skipped = 0

    for task in tasks:
        task_dir = os.path.join(root, task)
        if not os.path.isdir(task_dir):
            print(f"[warn] missing task dir: {task_dir}", file=sys.stderr)
            continue
        prompt = TASK_PROMPTS.get(task, task.replace("_", " "))
        for name in sorted(os.listdir(task_dir), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            ep_dir = os.path.join(task_dir, name)
            if not os.path.isdir(ep_dir):
                continue
            vid = os.path.join(ep_dir, "video.mp4")
            frc = os.path.join(ep_dir, "force.npy")
            if not (os.path.isfile(vid) and os.path.isfile(frc)):
                skipped += 1
                continue
            try:
                fshape = _npy_shape(frc)
            except Exception as e:
                print(f"[skip] bad force {frc}: {e}", file=sys.stderr)
                skipped += 1
                continue
            if len(fshape) != 2 or fshape[1] != 12:
                print(f"[skip] force not (T,12): {frc} shape={fshape}", file=sys.stderr)
                skipped += 1
                continue
            num_frames = int(fshape[0])
            demo_id = f"{task}_{name}"
            rel = f"{task}/{name}"
            rows.append(
                {
                    "demo_id": demo_id,
                    "task": task,
                    "prompt": prompt,
                    "camera_key": "cam_front",
                    "video": f"{rel}/video.mp4",
                    "force_sequence": f"{rel}/force.npy",
                    "num_frames": num_frames,
                }
            )

    if not rows:
        print("no rows; abort", file=sys.stderr)
        return 1

    fieldnames = [
        "demo_id",
        "task",
        "prompt",
        "camera_key",
        "video",
        "force_sequence",
        "num_frames",
    ]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out} (skipped {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
