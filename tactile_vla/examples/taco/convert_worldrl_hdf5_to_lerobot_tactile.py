"""Convert raw WorldRL HDF5 data to the TACO tactile-aware LeRobot format.

Output convention:
  state   = pose(6) + continuous_gripper(1)
  actions = pose(6) + continuous_gripper(1)
  force_history = 8 x 12 left/right force-torque history by default
  advantage = scalar success/preference label, defaulting to 1.0
  pose(6) = xyz(3) + Euler/RPY(3), copied unchanged from puppet/pose.
"""

import shutil
from pathlib import Path

import h5py
import numpy as np
import tyro

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    import cv2
except ImportError:
    cv2 = None


FORCE_NORM = np.array([20.0, 20.0, 20.0, 2.0, 2.0, 2.0], dtype=np.float32)


def _resize_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Keep aspect ratio, then pad with black to target size."""
    if image.shape[:2] == (target_height, target_width):
        return image

    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / target_width, cur_height / target_height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
    pad_top = (target_height - resized_height) // 2
    pad_left = (target_width - resized_width) // 2
    padded[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return padded


def _decode_and_resize(img_bytes: bytes, target_height: int, target_width: int) -> np.ndarray:
    assert cv2 is not None, "Need opencv-python: pip install opencv-python"
    arr = np.frombuffer(bytes(img_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed.")
    img_bgr = _resize_with_pad(img_bgr, target_height, target_width)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def normalize_wrench(wrench: np.ndarray) -> np.ndarray:
    return wrench / FORCE_NORM


def build_force_history(force_left: np.ndarray, force_right: np.ndarray, history_len: int) -> np.ndarray:
    """Build sliding-window force history for every timestep."""
    fl = normalize_wrench(force_left.astype(np.float32))
    fr = normalize_wrench(force_right.astype(np.float32))
    force_concat = np.concatenate([fl, fr], axis=-1)
    pad = np.zeros((history_len - 1, 12), dtype=np.float32)
    padded = np.concatenate([pad, force_concat], axis=0)
    return np.stack([padded[t : t + history_len] for t in range(len(force_concat))], axis=0)


def main(
    data_dir: str,
    repo_name: str = "taco_tactile_ki_advantage_example",
    output_dir: str | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    task: str = "",
    force_history_len: int = 8,
    stride: int = 5,
    use_front_as_wrist: bool = True,
    image_height: int = 192,
    image_width: int = 256,
    fps: int = 6,
    advantage_value: float = 1.0,
):
    root = Path(output_dir) / repo_name if output_dir else None
    if root is not None and root.exists():
        if overwrite:
            shutil.rmtree(root)
        else:
            raise FileExistsError(f"Output directory exists: {root}. Use --overwrite to replace.")

    features = {
        "image": {
            "dtype": "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
        "force_history": {
            "dtype": "float32",
            "shape": (force_history_len, 12),
            "names": ["force_history"],
        },
    }
    features["advantage"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["advantage"],
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=fps,
        root=root,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    hdf5_files = sorted(Path(data_dir).glob("*.hdf5"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}.")

    print(f"Found {len(hdf5_files)} episode files.")
    print(f"Task: {task!r}")
    print(f"Force history len: {force_history_len}")
    print(f"Stride: {stride}")

    for hdf5_path in hdf5_files:
        print(f"Processing: {hdf5_path.name}")
        try:
            with h5py.File(hdf5_path, "r") as f:
                joints = f["puppet/joint"][:].astype(np.float32)
                poses = f["puppet/pose"][:].astype(np.float32)
                grippers = f["puppet/gripper"][:].astype(np.float32)
                front_imgs = f["observations/images/cam_front"][:]
                high_imgs = front_imgs if use_front_as_wrist else f["observations/images/cam_high"][:]

                if "tactile/force_left" not in f or "tactile/force_right" not in f:
                    raise KeyError("Expected tactile/force_left and tactile/force_right in each HDF5 file.")
                force_left = f["tactile/force_left"][:].astype(np.float32)
                force_right = f["tactile/force_right"][:].astype(np.float32)
        except Exception as e:
            print(f"  Load failed, skipping: {e}")
            continue

        indices = np.arange(0, len(joints), stride, dtype=np.int64)
        if len(indices) < 2:
            print(f"  [SKIP] {hdf5_path.name}: too few sampled frames ({len(indices)})")
            continue

        joints = joints[indices]
        poses = poses[indices]
        grippers = grippers[indices]
        front_imgs = front_imgs[indices]
        high_imgs = high_imgs[indices]
        force_left = force_left[indices]
        force_right = force_right[indices]

        force_hist = build_force_history(force_left, force_right, force_history_len)
        frame_count = 0
        for i in range(len(indices)):
            continuous_gripper = np.array([float(grippers[i][0])], dtype=np.float32)
            state = np.concatenate([poses[i], continuous_gripper]).astype(np.float32)
            actions = np.concatenate([poses[i], continuous_gripper]).astype(np.float32)

            try:
                image = _decode_and_resize(front_imgs[i], image_height, image_width)
                wrist_image = _decode_and_resize(high_imgs[i], image_height, image_width)
            except Exception as e:
                print(f"  Frame {i} image decode failed, skipping: {e}")
                continue

            frame = {
                "image": image,
                "wrist_image": wrist_image,
                "state": state,
                "actions": actions,
                "force_history": force_hist[i],
                "task": task,
            }
            frame["advantage"] = np.array([advantage_value], dtype=np.float32)

            dataset.add_frame(frame)
            frame_count += 1

        if frame_count == 0:
            print(f"  Skipping empty episode: {hdf5_path.name}")
            continue

        dataset.save_episode()
        print(f"  Saved {frame_count} frames.")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
