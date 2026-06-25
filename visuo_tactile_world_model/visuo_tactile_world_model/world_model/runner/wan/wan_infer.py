import gc
import inspect
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from visuo_tactile_world_model.world_model.dataset.operators import ImageCropAndResize
from visuo_tactile_world_model.world_model.runner.runner_util.instantiation import instantiate_component_from_section
from visuo_tactile_world_model.world_model.runner.runner_util.wan_runtime import build_wan_i2v_pipeline_from_params
from visuo_tactile_world_model.world_model.utils.data import save_video


class WanInferRunner:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _load_train_ckpt(pipe, ckpt_path: str):
        import torch
        from safetensors.torch import load_file
        print(f"[WanInferRunner] loading training ckpt: {ckpt_path}")
        sd = load_file(ckpt_path)
        groups = {"dit.": pipe.dit,
                  "tactile_tokenizer.": getattr(pipe, "tactile_tokenizer", None),
                  "tactile_head.": getattr(pipe, "tactile_head", None)}
        for prefix, module in groups.items():
            if module is None:
                continue
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not sub:
                print(f"  [skip] no keys with prefix '{prefix}' in ckpt")
                continue
            missing, unexpected = module.load_state_dict(sub, strict=False)
            target_dtype = next(module.parameters()).dtype
            target_device = next(module.parameters()).device
            module.to(dtype=target_dtype, device=target_device)
            print(f"  [load] '{prefix}' -> {len(sub)} keys; missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"    missing(head 3): {list(missing)[:3]}")
            if unexpected:
                print(f"    unexpected(head 3): {list(unexpected)[:3]}")

    @staticmethod
    def _denormalize_tactile(tactile_pred, dataset):
        mean = getattr(dataset, "tactile_mean", None)
        std = getattr(dataset, "tactile_std", None)
        if mean is None or std is None:
            return None

        tactile_np = tactile_pred.detach().float().cpu().numpy()
        if hasattr(mean, "detach"):
            mean = mean.detach().float().cpu().numpy()
        if hasattr(std, "detach"):
            std = std.detach().float().cpu().numpy()

        mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        return tactile_np * std + mean

    def run(self):
        full_config = self.config.full_config
        dataset, _ = instantiate_component_from_section(
            full_config.dataset,
            full_config,
            section_name="dataset",
        )
        model_params = (
            full_config.model.params.to_dict()
            if hasattr(full_config.model.params, "to_dict")
            else dict(full_config.model.params)
        )
        model_params["pipeline_class_path"] = full_config.model.class_path
        model = build_wan_i2v_pipeline_from_params(model_params)
        model_call_params = inspect.signature(model.__call__).parameters
        supports_context = "context" in model_call_params

        # Optional: load a training checkpoint (dit + tactile_tokenizer + tactile_head
        # merged, saved by ModelLogger with `remove_prefix_in_ckpt: pipe.`) on top of
        # the base pipeline. This avoids the hash-based auto-detect path which only
        # knows base DiT/T5/VAE shards.
        train_ckpt = getattr(self.config, "train_ckpt", None)
        if train_ckpt:
            self._load_train_ckpt(model, str(train_ckpt))

        text_embeddings = {}
        text_emb_path = getattr(self.config, "text_embeddings_path", None)
        if text_emb_path:
            emb_npz = np.load(str(text_emb_path), allow_pickle=False)
            for prompt in emb_npz.files:
                text_embeddings[str(prompt)] = torch.from_numpy(emb_npz[prompt])
            print(f"[WanInferRunner] loaded {len(text_embeddings)} cached text embeddings from {text_emb_path}")

        output_dir = getattr(self.config, "output_dir", "./outputs/libero_infer")
        os.makedirs(output_dir, exist_ok=True)

        fps = int(getattr(self.config, "fps", 16))
        quality = int(getattr(self.config, "quality", 5))
        seed_base = int(getattr(self.config, "seed", 0))
        infer_kwargs = dict(getattr(self.config, "infer_kwargs", {}))
        input_image_resize_mode = getattr(self.config, "input_image_resize_mode", "stretch")
        conditioning_frame_fractions = self._sanitize_conditioning_fractions(
            getattr(self.config, "conditioning_frame_fractions", [0.0])
        )
        window_size = int(infer_kwargs.get("num_frames", 81))
        target_height = infer_kwargs.get("height", None)
        target_width = infer_kwargs.get("width", None)
        input_image_resizer = None
        if target_height is not None and target_width is not None:
            input_image_resizer = ImageCropAndResize(
                height=int(target_height),
                width=int(target_width),
                max_pixels=None,
                height_division_factor=16,
                width_division_factor=16,
                resize_mode=input_image_resize_mode,
            )

        for item in tqdm(dataset, total=len(dataset), desc="Infer"):
            conditioning_inputs = self._build_conditioning_inputs(item, conditioning_frame_fractions, window_size)
            for cond_idx, (frame_fraction, frame_index, input_image, compare_start) in enumerate(conditioning_inputs):
                if input_image_resizer is not None:
                    input_image = input_image_resizer(input_image)
                call_kwargs = dict(infer_kwargs)
                # tactile joint-denoise conditioning
                tactile_init = item.get("tactile_init")
                if tactile_init is not None:
                    call_kwargs.setdefault("tactile_init", tactile_init)
                prompt = str(item["prompt"])
                context = None
                if text_embeddings:
                    if prompt not in text_embeddings:
                        raise KeyError(
                            f"No cached text embedding for prompt={prompt!r}. "
                            f"Available: {sorted(text_embeddings)}"
                        )
                    context = text_embeddings[prompt]
                model_kwargs = dict(
                    prompt=prompt,
                    input_image=input_image,
                    seed=seed_base + int(item["row_id"]) + int(cond_idx),
                    **call_kwargs,
                )
                if supports_context and context is not None:
                    model_kwargs["context"] = context
                result = model(**model_kwargs)
                if isinstance(result, tuple):
                    video, tactile_pred = result
                else:
                    video, tactile_pred = result, None

                cond_tag = self._condition_tag(frame_fraction, frame_index)
                name = f"{item['row_id']:06d}__{item['demo_id']}__{item['camera_key']}__{cond_tag}.mp4"
                save_path = str(Path(output_dir) / name)
                save_video(video, save_path, fps=fps, quality=quality)

                if tactile_pred is not None:
                    tactile_path = str(Path(save_path).with_suffix("")) + "_tactile.npy"
                    tactile_denorm = self._denormalize_tactile(tactile_pred, dataset)
                    np.save(tactile_path, tactile_denorm if tactile_denorm is not None
                            else tactile_pred.detach().float().cpu().numpy())
                gt_video_path = item.get("video_path")
                if gt_video_path and os.path.exists(gt_video_path):
                    compare_frames = self._build_compare_frames(
                        video,
                        gt_video_path,
                        gt_start_frame=compare_start,
                    )
                    compare_path = str(Path(save_path).with_suffix("")) + "_compare.mp4"
                    save_video(compare_frames, compare_path, fps=fps, quality=quality)

                # Release intermediate tensors to avoid VRAM accumulation across samples.
                del video, result
                if tactile_pred is not None:
                    del tactile_pred
                if state_pred is not None:
                    del state_pred
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _sanitize_conditioning_fractions(self, fractions) -> list:
        """Returns a list of floats and/or the sentinel string "tail"."""
        if fractions is None:
            return [0.0]
        if isinstance(fractions, str):
            fractions = [x.strip() for x in fractions.split(",") if x.strip()]
        if not isinstance(fractions, (list, tuple)):
            fractions = [fractions]
        float_vals = []
        has_tail = False
        for value in fractions:
            if isinstance(value, str) and value.strip().lower() == "tail":
                has_tail = True
                continue
            try:
                v = float(value)
            except Exception:
                continue
            float_vals.append(max(0.0, min(1.0, v)))
        result = sorted(set(float_vals))
        if has_tail:
            result.append("tail")
        if len(result) == 0:
            return [0.0]
        return result

    def _get_total_video_frames(self, video_path: str) -> int:
        reader = imageio.get_reader(video_path)
        try:
            try:
                total = int(reader.count_frames())
                if total > 0:
                    return total
            except Exception:
                pass
            try:
                metadata = reader.get_meta_data()
                nframes = int(metadata.get("nframes", 0))
                if nframes > 0:
                    return nframes
            except Exception:
                pass
            return 1
        finally:
            reader.close()

    def _read_video_frame(self, video_path: str, frame_index: int):
        reader = imageio.get_reader(video_path)
        try:
            cur = int(max(0, frame_index))
            while cur >= 0:
                try:
                    frame = reader.get_data(cur)
                    return Image.fromarray(frame).convert("RGB"), cur
                except Exception:
                    cur -= 1
            return None, None
        finally:
            reader.close()

    def _build_compare_frames(self, gen_frames, gt_video_path: str, gt_start_frame: int = 0):
        gen_pil = [
            f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f))
            for f in gen_frames
        ]
        target_size = gen_pil[0].size
        n = len(gen_pil)
        gt_pil = []
        reader = imageio.get_reader(gt_video_path)
        try:
            for i in range(n):
                try:
                    frame = reader.get_data(max(0, int(gt_start_frame) + i))
                except (IndexError, RuntimeError):
                    if len(gt_pil) == 0:
                        frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                    else:
                        gt_pil.append(gt_pil[-1].copy())
                        continue
                gt_pil.append(Image.fromarray(frame).convert("RGB").resize(target_size, Image.BILINEAR))
        finally:
            reader.close()
        w, h = target_size
        compare = []
        for gt_img, gen_img in zip(gt_pil, gen_pil):
            canvas = Image.new("RGB", (w * 2, h))
            canvas.paste(gt_img, (0, 0))
            canvas.paste(gen_img, (w, 0))
            compare.append(canvas)
        return compare

    def _condition_tag(self, frame_fraction, frame_index: int | None) -> str:
        if frame_fraction == "tail":
            frame_text = "u" if frame_index is None else str(int(frame_index))
            return f"f{frame_text}_tail"
        pct = int(round(float(frame_fraction) * 100.0))
        frame_text = "u" if frame_index is None else str(int(frame_index))
        return f"f{frame_text}_p{pct:03d}"

    def _build_conditioning_inputs(self, item: dict, fractions: list, window_size: int = 81):
        """Returns list of (frame_fraction, cond_frame_index, input_image, compare_start).

        compare_start is the GT video frame to start the side-by-side compare from.
        For "tail" this is always max(0, N-window_size) regardless of whether
        _read_video_frame fell back to an earlier frame.

        When the dataset pre-locks the conditioning frame (e.g. strided inference),
        it sets item["compare_start"] to the exact GT frame index. In that case
        item["input_image"] is used directly so we never re-read the wrong frame,
        while video_path is still available for compare-video generation.
        """
        if "compare_start" in item and item.get("input_image") is not None:
            locked_start = int(item["compare_start"])
            return [(0.0, locked_start, item["input_image"], locked_start)]

        video_path = item.get("video_path")
        if video_path and os.path.exists(video_path):
            total_frames = self._get_total_video_frames(video_path)
            outputs = []
            for frac in fractions:
                if frac == "tail":
                    target_idx = max(0, total_frames - window_size)
                    compare_start = target_idx
                else:
                    target_idx = int(round(float(frac) * max(total_frames - 1, 0)))
                    compare_start = None  # filled below from actual_idx
                frame, actual_idx = self._read_video_frame(video_path, target_idx)
                if frame is None:
                    continue
                if compare_start is None:
                    compare_start = actual_idx
                outputs.append((frac, actual_idx, frame, compare_start))
            if len(outputs) > 0:
                return outputs
        # Fallback to dataset-provided first frame.
        return [(0.0, 0, item["input_image"], 0)]
