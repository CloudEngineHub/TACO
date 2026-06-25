import os
import json
from pathlib import Path
import pandas
import imageio
from PIL import Image


class LiberoInferenceDataset:
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        video_key: str = "video",
        prompt_key: str = "prompt",
        demo_id_key: str = "demo_id",
        camera_key_col: str = "camera_key",
        max_samples: int | None = None,
    ):
        self.base_path = base_path
        self.video_key = video_key
        self.prompt_key = prompt_key
        self.demo_id_key = demo_id_key
        self.camera_key_col = camera_key_col
        metadata = self._load_metadata(metadata_path)
        if max_samples is not None:
            metadata = metadata[: int(max_samples)]
        self.rows = metadata

    @staticmethod
    def _is_missing(value) -> bool:
        return value is None or (isinstance(value, float) and pandas.isna(value))

    def _resolve_demo_id(self, row: dict, fallback_idx: int) -> str:
        primary = row.get(self.demo_id_key)
        if not self._is_missing(primary):
            return str(primary)
        for alt_key in ("source_demo", "episode_id", "demo", "demo_id"):
            alt = row.get(alt_key)
            if not self._is_missing(alt):
                return str(alt)
        rel_video = row.get(self.video_key)
        if not self._is_missing(rel_video):
            parts = Path(str(rel_video)).parts
            if parts:
                return str(parts[0])
        return str(fallback_idx)

    def _resolve_camera_key(self, row: dict) -> str:
        primary = row.get(self.camera_key_col)
        if not self._is_missing(primary):
            return str(primary)
        rel_video = row.get(self.video_key)
        if not self._is_missing(rel_video):
            return Path(str(rel_video)).stem or "unknown"
        return "unknown"

    def _load_metadata(self, metadata_path: str):
        if metadata_path.endswith(".jsonl"):
            rows = []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a list in json metadata: {metadata_path}")
            return data
        metadata = pandas.read_csv(metadata_path)
        return [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __len__(self):
        return len(self.rows)

    def _resolve_video_path(self, value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.join(self.base_path, value)

    def _load_first_frame(self, video_path: str) -> Image.Image:
        reader = imageio.get_reader(video_path)
        try:
            frame = reader.get_data(0)
            return Image.fromarray(frame).convert("RGB")
        finally:
            reader.close()

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        prompt = str(row.get(self.prompt_key, ""))
        rel_video = str(row.get(self.video_key, ""))
        video_path = self._resolve_video_path(rel_video)
        first_frame = self._load_first_frame(video_path)
        demo_id = self._resolve_demo_id(row, idx)
        camera_key = self._resolve_camera_key(row)
        return {
            "row_id": idx,
            "prompt": prompt,
            "video_path": video_path,
            "input_image": first_frame,
            "demo_id": demo_id,
            "camera_key": camera_key,
        }
