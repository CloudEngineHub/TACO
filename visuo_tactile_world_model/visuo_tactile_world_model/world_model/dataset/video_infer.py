import os

import numpy as np
import torch

from .libero_infer import LiberoInferenceDataset


class VideoInferenceDataset(LiberoInferenceDataset):
    pass


class TactileVideoInferenceDataset(VideoInferenceDataset):
    """Video inference dataset with tactile/force frame-0 anchor loading."""

    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        video_key: str = "video",
        prompt_key: str = "prompt",
        demo_id_key: str = "demo_id",
        camera_key_col: str = "camera_key",
        max_samples: int | None = None,
        force_key: str = "force_sequence",
        stats_path: str | None = None,
    ):
        super().__init__(
            base_path=base_path,
            metadata_path=metadata_path,
            video_key=video_key,
            prompt_key=prompt_key,
            demo_id_key=demo_id_key,
            camera_key_col=camera_key_col,
            max_samples=max_samples,
        )
        self.force_key = force_key
        self.tactile_mean = None
        self.tactile_std = None
        if stats_path:
            stats = np.load(stats_path)
            self.tactile_mean = torch.from_numpy(stats["mean"]).float().view(1, -1)
            self.tactile_std = torch.from_numpy(stats["std"]).float().clamp_min(1e-6).view(1, -1)

    def _resolve_force_path(self, value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.join(self.base_path, value)

    def _load_force(self, path: str) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[:, None]
        return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        rel_force = self.rows[idx].get(self.force_key)
        if rel_force is None:
            return item
        if isinstance(rel_force, float) and np.isnan(rel_force):
            return item
        force_path = self._resolve_force_path(str(rel_force))
        force_full = self._load_force(force_path)
        if self.tactile_mean is not None:
            force_full = (force_full - self.tactile_mean) / self.tactile_std
        item["tactile_init"] = force_full[:1].clone()
        item["tactile_gt"] = force_full
        item["force_path"] = force_path
        return item
