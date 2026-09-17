from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

from visuo_tactile_world_model.world_model.dataset.operators import ImageCropAndResize
from visuo_tactile_world_model.world_model.utils.data import save_video


class PeriodicWanVideoValidator:
    def __init__(
        self,
        dataset,
        output_root: str,
        every_steps: int = 1000,
        extra_steps: list[int] | None = None,
        num_samples: int = 3,
        fps: int = 16,
        quality: int = 5,
        seed_base: int = 0,
        infer_kwargs: dict | None = None,
        anchor_frame_fractions: list[float] | None = None,
        input_image_resize_mode: str = "stretch",
        wandb_run=None,
        wandb_log_video: bool = True,
        video_metrics: list[str] | None = None,
    ):
        self.dataset = dataset
        self.output_root = output_root
        self.every_steps = max(int(every_steps), 0)
        self.extra_steps = sorted({int(step) for step in (extra_steps or []) if int(step) > 0})
        self.num_samples = max(int(num_samples), 0)
        self.fps = int(fps)
        self.quality = int(quality)
        self.seed_base = int(seed_base)
        self.infer_kwargs = {} if infer_kwargs is None else dict(infer_kwargs)
        self.anchor_frame_fractions = self._sanitize_anchor_fractions(anchor_frame_fractions)
        self.input_image_resize_mode = input_image_resize_mode
        self.wandb_run = wandb_run
        self.wandb_log_video = wandb_log_video
        # Metrics to compute per validation video: psnr, ssim, lpips, dino.
        # lpips requires the `lpips` package; dino requires the local DINOv2 checkpoint.
        self.video_metrics: list[str] = list(video_metrics) if video_metrics is not None else ["psnr", "ssim"]
        # Lazy-loaded heavy models (LPIPS / DINOv2), instantiated on first use.
        self._lpips_model = None
        self._vm_mod = None  # compute_video_metrics module, loaded lazily for DINOv2

        self.output_dir = os.path.join(self.output_root, "validation")
        target_height = self.infer_kwargs.get("height", None)
        target_width = self.infer_kwargs.get("width", None)
        self.input_image_resizer = None
        if target_height is not None and target_width is not None:
            self.input_image_resizer = ImageCropAndResize(
                height=int(target_height),
                width=int(target_width),
                max_pixels=None,
                height_division_factor=16,
                width_division_factor=16,
                resize_mode=input_image_resize_mode,
            )

    def enabled(self) -> bool:
        return (self.every_steps > 0 or len(self.extra_steps) > 0) and self.num_samples > 0 and len(self.dataset) > 0

    def _step_dir(self, global_step: int) -> Path:
        return Path(self.output_dir) / f"step_{int(global_step)}"

    def maybe_run(self, accelerator, model, global_step: int):
        should_run = False
        if self.every_steps > 0 and global_step % self.every_steps == 0:
            should_run = True
        if global_step in self.extra_steps:
            should_run = True
        if not self.enabled() or not should_run:
            return

        active_ranks = min(int(self.num_samples), int(accelerator.num_processes), len(self.dataset))
        is_active_rank = int(accelerator.process_index) < active_ranks
        accelerator.wait_for_everyone()
        if not is_active_rank:
            accelerator.wait_for_everyone()
            return

        training_module = accelerator.unwrap_model(model)
        pipe = training_module.pipe
        validation_units = getattr(training_module, "validation_pipe_units", None) or pipe.units
        previous_units = pipe.units
        prev_scheduler_steps = len(getattr(pipe.scheduler, "timesteps", []))
        prev_scheduler_training = bool(getattr(pipe.scheduler, "training", False))
        sample_idx = self._sample_index_for_rank(global_step, accelerator.process_index, active_ranks)

        try:
            pipe.units = validation_units
            with torch.inference_mode():
                item = self.dataset[sample_idx]
                gt_video_path = item.get("video_path")
                anchor_inputs = self._build_anchor_inputs(item)
                for anchor_idx, (frame_fraction, frame_index, input_image) in enumerate(anchor_inputs):
                    if self.input_image_resizer is not None:
                        input_image = self.input_image_resizer(input_image)
                    call_kwargs = dict(self.infer_kwargs)
                    tactile_init = item.get("tactile_init")
                    if tactile_init is not None:
                        call_kwargs.setdefault("tactile_init", tactile_init)
                    result = pipe(
                        prompt=item["prompt"],
                        input_image=input_image,
                        seed=(
                            self.seed_base
                            + int(global_step)
                            + int(item.get("row_id", sample_idx))
                            + int(anchor_idx)
                        ),
                        progress_bar_cmd=lambda x: x,
                        **call_kwargs,
                    )
                    if isinstance(result, tuple):
                        video, tactile_pred = result
                    else:
                        video, tactile_pred = result, None

                    anchor_tag = self._anchor_tag(frame_fraction, frame_index)
                    save_path = self._video_path(
                        global_step,
                        accelerator.process_index,
                        item,
                        anchor_tag=anchor_tag,
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    save_video(video, save_path, fps=self.fps, quality=self.quality)

                    if gt_video_path and os.path.exists(gt_video_path):
                        compare_frames = self._build_compare_frames(
                            video,
                            gt_video_path,
                            gt_start_frame=frame_index,
                        )
                        compare_path = str(Path(save_path).with_suffix("")) + "_compare.mp4"
                        save_video(compare_frames, compare_path, fps=self.fps, quality=self.quality)
                        del compare_frames

                    if gt_video_path and os.path.exists(gt_video_path) and self.video_metrics:
                        vm = self._compute_video_metrics(video, gt_video_path, gt_start_frame=frame_index)
                        if vm:
                            vm_path = str(Path(save_path).with_suffix("")) + "_video_metrics.json"
                            with open(vm_path, "w", encoding="utf-8") as _vmf:
                                json.dump(vm, _vmf)

                    if tactile_pred is not None:
                        tactile_path = str(Path(save_path).with_suffix("")) + "_tactile.npy"
                        np.save(tactile_path, tactile_pred.detach().float().cpu().numpy())
                        tactile_gt = item.get("tactile_gt")
                        if tactile_gt is not None:
                            plot_path = str(Path(save_path).with_suffix("")) + "_tactile_compare.png"
                            ds_mean = getattr(self.dataset, "tactile_mean", None)
                            ds_std = getattr(self.dataset, "tactile_std", None)
                            mean_np = ds_mean.detach().cpu().numpy().reshape(-1) if ds_mean is not None else None
                            std_np = ds_std.detach().cpu().numpy().reshape(-1) if ds_std is not None else None
                            self._save_tactile_compare_plot(
                                tactile_gt.detach().float().cpu().numpy(),
                                tactile_pred.detach().float().cpu().numpy(),
                                plot_path,
                                mean=mean_np,
                                std=std_np,
                            )
                    caption = self._caption(
                        item,
                        frame_fraction=frame_fraction,
                        frame_index=frame_index,
                    )
                    self._write_sidecar(save_path, caption, accelerator.process_index, sample_idx)
                    self._write_rank_manifest(global_step, accelerator.process_index, sample_idx, save_path)
                    print(
                        f"[Validation][rank{accelerator.process_index}] "
                        f"step={global_step} anchor={anchor_tag} saved {save_path}"
                    )
                    del video, input_image, result, tactile_pred
                del item
        finally:
            pipe.units = previous_units
            if prev_scheduler_steps > 0:
                pipe.scheduler.set_timesteps(prev_scheduler_steps, training=prev_scheduler_training)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            saved_items = self._collect_saved_items(global_step)
            missing_ranks = self._find_missing_ranks(global_step, active_ranks)
            print(
                f"[Validation] step={global_step} generated {len(saved_items)} video(s) "
                f"at {self._step_dir(global_step)}"
            )
            if len(missing_ranks) > 0:
                print(f"[Validation] step={global_step} missing rank outputs: {missing_ranks}")
            self._log_to_wandb(saved_items, global_step)

    def _sample_index_for_rank(self, global_step: int, process_index: int, num_processes: int) -> int:
        sample_count = min(max(num_processes, self.num_samples), len(self.dataset))
        generator = torch.Generator(device="cpu").manual_seed(self.seed_base + int(global_step))
        indices = torch.randperm(len(self.dataset), generator=generator)[:sample_count].tolist()
        return indices[process_index % len(indices)]

    def _video_path(self, global_step: int, process_index: int, item: dict, anchor_tag: str = "f0p") -> str:
        row_id = int(item.get("row_id", process_index))
        demo_id = str(item.get("demo_id", row_id))
        camera_key = str(item.get("camera_key", "unknown"))
        name = f"row_{row_id:06d}__{demo_id}__{camera_key}__{anchor_tag}.mp4"
        return str(self._step_dir(global_step) / f"rank{int(process_index)}" / name)

    def _sidecar_path(self, save_path: str) -> str:
        return str(Path(save_path).with_suffix(".json"))

    def _rank_manifest_path(self, global_step: int, process_index: int) -> str:
        return str(self._step_dir(global_step) / f"rank{int(process_index)}" / "_rank_manifest.json")

    def _write_sidecar(self, save_path: str, caption: str, process_index: int, sample_idx: int):
        sidecar = {
            "caption": caption,
            "process_index": int(process_index),
            "sample_idx": int(sample_idx),
            "video_path": save_path,
        }
        with open(self._sidecar_path(save_path), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False)

    def _write_rank_manifest(self, global_step: int, process_index: int, sample_idx: int, save_path: str):
        manifest = {
            "process_index": int(process_index),
            "sample_idx": int(sample_idx),
            "video_path": save_path,
        }
        manifest_path = self._rank_manifest_path(global_step, process_index)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)

    def _collect_saved_items(self, global_step: int) -> list[tuple[int, str, str, str]]:
        step_dir = self._step_dir(global_step)
        saved_items = []
        for sidecar_path in sorted(step_dir.glob("rank*/*.json")):
            if sidecar_path.name == "_rank_manifest.json":
                continue
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
            save_path = sidecar.get("video_path")
            if not save_path or not os.path.exists(save_path):
                continue
            process_index = int(sidecar.get("process_index", -1))
            caption = str(sidecar.get("caption", ""))
            key_suffix = Path(save_path).stem
            saved_items.append((process_index, save_path, caption, key_suffix))
        return saved_items

    def _find_missing_ranks(self, global_step: int, num_processes: int) -> list[int]:
        step_dir = self._step_dir(global_step)
        present = set()
        for manifest_path in step_dir.glob("rank*/_rank_manifest.json"):
            rank_name = manifest_path.parent.name
            if rank_name.startswith("rank"):
                try:
                    present.add(int(rank_name[4:]))
                except Exception:
                    pass
        return [rank for rank in range(int(num_processes)) if rank not in present]

    def _build_compare_frames(self, gen_frames, gt_video_path: str, gt_start_frame: int = 0):
        """Build side-by-side [GT | generated] frames. GT frames are taken from
        the source video starting at `gt_start_frame`; if the GT clip is shorter
        than the generated sequence, the last GT frame is repeated."""
        gen_pil = [
            f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f))
            for f in gen_frames
        ]
        target_size = gen_pil[0].size  # (W, H)
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

    def _compute_video_metrics(self, gen_frames, gt_video_path: str, gt_start_frame: int = 0) -> dict:
        """Compute PSNR / SSIM / LPIPS / DINOv2 between gen_frames and GT video.

        gen_frames: list of PIL images produced by the pipeline.
        Returns a dict with ``<metric>_mean`` and ``<metric>_std`` keys for each
        successfully computed metric in ``self.video_metrics``.
        """
        if not self.video_metrics:
            return {}

        # Convert generated PIL frames → uint8 numpy (T, H, W, 3).
        gen_np = np.stack([np.asarray(f.convert("RGB")) for f in gen_frames])
        T = len(gen_np)

        # Load GT frames from the source video file.
        gt_list = []
        try:
            reader = imageio.get_reader(gt_video_path)
            for i in range(T):
                try:
                    frame = reader.get_data(max(0, gt_start_frame + i))
                    gt_list.append(np.asarray(frame))
                except (IndexError, RuntimeError):
                    if gt_list:
                        gt_list.append(gt_list[-1].copy())
                    break
            reader.close()
        except Exception as exc:
            print(f"[Validation][video_metrics] failed to load GT frames: {exc}")
            return {}

        if not gt_list:
            return {}

        gt_np = np.stack(gt_list)
        T = min(len(gen_np), len(gt_np))
        gen_np, gt_np = gen_np[:T], gt_np[:T]

        # Resize GT to match generated resolution if they differ.
        if gt_np.shape[1:3] != gen_np.shape[1:3]:
            H, W = int(gen_np.shape[1]), int(gen_np.shape[2])
            gt_np = np.stack([
                np.asarray(Image.fromarray(gt_np[i]).resize((W, H), Image.BILINEAR))
                for i in range(T)
            ])

        result = {}

        if "psnr" in self.video_metrics:
            try:
                from skimage.metrics import peak_signal_noise_ratio as _ski_psnr
                scores = [float(_ski_psnr(gt_np[i], gen_np[i], data_range=255)) for i in range(T)]
                result["psnr_mean"] = float(np.mean(scores))
                result["psnr_std"] = float(np.std(scores))
            except Exception as exc:
                print(f"[Validation][video_metrics] PSNR failed: {exc}")

        if "ssim" in self.video_metrics:
            try:
                from skimage.metrics import structural_similarity as _ski_ssim
                scores = [float(_ski_ssim(gt_np[i], gen_np[i], channel_axis=-1, data_range=255)) for i in range(T)]
                result["ssim_mean"] = float(np.mean(scores))
                result["ssim_std"] = float(np.std(scores))
            except Exception as exc:
                print(f"[Validation][video_metrics] SSIM failed: {exc}")

        if "lpips" in self.video_metrics:
            try:
                import torch as _torch
                import lpips as _lpips_pkg
                if self._lpips_model is None:
                    self._lpips_model = _lpips_pkg.LPIPS(net="alex").eval()
                    if _torch.cuda.is_available():
                        self._lpips_model = self._lpips_model.cuda()
                _dev = next(self._lpips_model.parameters()).device

                def _to_lpips(arr):
                    t = _torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 127.5 - 1.0
                    return t.to(_dev)

                scores = []
                bs = 16
                with _torch.no_grad():
                    for s in range(0, T, bs):
                        g_t = _to_lpips(gt_np[s:s + bs])
                        p_t = _to_lpips(gen_np[s:s + bs])
                        v = self._lpips_model(g_t, p_t).squeeze()
                        scores.extend(v.cpu().tolist() if v.ndim > 0 else [float(v)])
                result["lpips_mean"] = float(np.mean(scores))
                result["lpips_std"] = float(np.std(scores))
            except Exception as exc:
                print(f"[Validation][video_metrics] LPIPS failed: {exc}")

        if "dino" in self.video_metrics:
            try:
                if self._vm_mod is None:
                    import importlib.util as _ilu
                    _script = Path(__file__).parents[3] / "scripts" / "compute_video_metrics.py"
                    _spec = _ilu.spec_from_file_location("_visuo_tactile_world_model.world_model_vm", str(_script))
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    self._vm_mod = _mod
                scores = self._vm_mod.compute_dino_similarity_batch(gt_np, gen_np)
                result["dino_mean"] = float(np.mean(scores))
                result["dino_std"] = float(np.std(scores))
            except Exception as exc:
                print(f"[Validation][video_metrics] DINOv2 failed: {exc}")

        if result:
            parts = [f"{k}={v:.4f}" for k, v in result.items() if k.endswith("_mean")]
            print(f"[Validation][video_metrics] T={T}  {', '.join(parts)}")

        return result

    def _save_tactile_compare_plot(
        self,
        gt_arr: np.ndarray,
        pred_arr: np.ndarray,
        save_path: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        """Plot per-channel GT vs predicted tactile traces with strict per-frame
        alignment: pred covers frames [0, num_frames-1] of the demo (because the
        validator anchors on frame 0), so GT is truncated to the same window
        and both share the integer-frame x-axis.

        If `mean`/`std` are provided, both inputs are assumed to be in the
        normalized space (z-scored at training time); the plot is drawn in the
        denormalized physical scale and MSE is reported in BOTH spaces.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[Validation] matplotlib unavailable, skip tactile compare plot. Error: {exc}")
            return

        if pred_arr.ndim == 3:
            pred_arr = pred_arr[0]
        D = int(pred_arr.shape[-1])
        n = int(pred_arr.shape[0])
        gt_used = gt_arr[:n]
        gt_t = np.arange(gt_used.shape[0])
        pred_t = np.arange(n)

        # MSE in normalized space (training loss space).
        m = min(gt_used.shape[0], n)
        if m > 0:
            diff_n = pred_arr[:m] - gt_used[:m]
            per_ch_mse_norm = (diff_n ** 2).mean(axis=0)
            overall_mse_norm = float(per_ch_mse_norm.mean())
        else:
            per_ch_mse_norm = np.full((D,), np.nan, dtype=np.float32)
            overall_mse_norm = float("nan")

        # Optional denormalization to physical scale for plotting + a second MSE.
        if mean is not None and std is not None:
            mean = np.asarray(mean, dtype=np.float64).reshape(-1)
            std = np.asarray(std, dtype=np.float64).reshape(-1)
            gt_phys = gt_used.astype(np.float64) * std + mean
            pred_phys = pred_arr.astype(np.float64) * std + mean
            if m > 0:
                diff_p = pred_phys[:m] - gt_phys[:m]
                per_ch_mse_phys = (diff_p ** 2).mean(axis=0)
                overall_mse_phys = float(per_ch_mse_phys.mean())
            else:
                per_ch_mse_phys = np.full((D,), np.nan, dtype=np.float64)
                overall_mse_phys = float("nan")
            gt_plot = gt_phys
            pred_plot = pred_phys
            per_ch_for_title = per_ch_mse_phys
            unit_label = "physical"
        else:
            gt_plot = gt_used
            pred_plot = pred_arr
            per_ch_mse_phys = None
            overall_mse_phys = None
            per_ch_for_title = per_ch_mse_norm
            unit_label = "normalized"

        if per_ch_mse_phys is not None:
            print(
                f"[Validation][tactile] {os.path.basename(save_path)} "
                f"overall_mse_norm={overall_mse_norm:.6f} overall_mse_phys={overall_mse_phys:.4f} "
                f"per_ch_mse_norm={np.array2string(per_ch_mse_norm, precision=4)} "
                f"per_ch_mse_phys={np.array2string(per_ch_mse_phys, precision=4)}"
            )
        else:
            print(
                f"[Validation][tactile] {os.path.basename(save_path)} "
                f"overall_mse_norm={overall_mse_norm:.6f} "
                f"per_ch_mse_norm={np.array2string(per_ch_mse_norm, precision=4)}"
            )

        cols = 4
        rows = (D + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.2), squeeze=False)
        for d in range(D):
            ax = axes.flat[d]
            ax.plot(gt_t, gt_plot[:, d], label="gt", color="#1f77b4", linewidth=1.0)
            ax.plot(pred_t, pred_plot[:, d], label="pred", color="#d62728", linewidth=1.0)
            ax.set_title(f"ch{d}  mse={per_ch_for_title[d]:.4f}", fontsize=9)
            ax.tick_params(labelsize=7)
            if d == 0:
                ax.legend(fontsize=7)
        for d in range(D, rows * cols):
            axes.flat[d].axis("off")
        if overall_mse_phys is not None:
            suptitle = (
                f"tactile gt vs pred ({unit_label}) — "
                f"MSE_phys = {overall_mse_phys:.4f}, MSE_norm = {overall_mse_norm:.6f}"
            )
        else:
            suptitle = f"tactile gt vs pred ({unit_label}) — MSE_norm = {overall_mse_norm:.6f}"
        fig.suptitle(suptitle, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(save_path, dpi=80)
        plt.close(fig)

    def _caption(self, item: dict, frame_fraction: float = 0.0, frame_index: int | None = None) -> str:
        row_id = int(item.get("row_id", -1))
        demo_id = str(item.get("demo_id", "unknown"))
        camera_key = str(item.get("camera_key", "unknown"))
        prompt = str(item.get("prompt", ""))
        frame_index_text = "unknown" if frame_index is None else str(int(frame_index))
        return (
            f"row_id={row_id} demo_id={demo_id} camera_key={camera_key} "
            f"anchor_fraction={frame_fraction:.4f} anchor_frame={frame_index_text} prompt={prompt}"
        )

    def _log_to_wandb(self, saved_items: list[tuple[int, str, str, str]], global_step: int):
        if self.wandb_run is None or len(saved_items) == 0:
            return
        if not self.wandb_log_video:
            return
        try:
            import wandb
        except Exception as exc:
            print(f"[Validation] wandb is unavailable, skip video upload. Error: {exc}")
            return

        log_data = {}

        # Aggregate video metrics across all samples for scalar logging.
        all_video_metrics: list[dict] = []

        for process_index, save_path, caption, key_suffix in saved_items:
            key_base = f"val/rank_{process_index}/{key_suffix}"
            log_data[key_base] = wandb.Video(
                save_path,
                fps=self.fps,
                format="mp4",
                caption=caption,
            )
            stem = str(Path(save_path).with_suffix(""))
            compare_path = stem + "_compare.mp4"
            if os.path.exists(compare_path):
                log_data[f"{key_base}_compare"] = wandb.Video(
                    compare_path,
                    fps=self.fps,
                    format="mp4",
                    caption=f"[gt | gen] {caption}",
                )
            tactile_compare_path = stem + "_tactile_compare.png"
            if os.path.exists(tactile_compare_path):
                log_data[f"{key_base}_tactile_compare"] = wandb.Image(
                    tactile_compare_path,
                    caption=f"tactile gt vs pred — {caption}",
                )
            video_metrics_path = stem + "_video_metrics.json"
            if os.path.exists(video_metrics_path):
                try:
                    with open(video_metrics_path, "r", encoding="utf-8") as _vmf:
                        all_video_metrics.append(json.load(_vmf))
                except Exception:
                    pass

        # Log aggregated video scalar metrics (mean over all validation samples).
        if all_video_metrics:
            def _vm_nanmean(key):
                vals = [m[key] for m in all_video_metrics if key in m and not np.isnan(m[key])]
                return float(np.mean(vals)) if vals else float("nan")

            for metric in ["psnr", "ssim", "lpips", "dino"]:
                mean_key = f"{metric}_mean"
                if any(mean_key in m for m in all_video_metrics):
                    log_data[f"val/video_{metric}"] = _vm_nanmean(mean_key)

            vm_parts = [
                f"{m}={log_data[f'val/video_{m}']:.4f}"
                for m in ["psnr", "ssim", "lpips", "dino"]
                if f"val/video_{m}" in log_data
            ]
            print(
                f"[Validation][video_metrics][wandb] step={global_step}  {', '.join(vm_parts)} "
                f"(aggregated over {len(all_video_metrics)} sample(s))"
            )

        self.wandb_run.log(log_data, step=global_step)

    def _sanitize_anchor_fractions(self, fractions: list[float] | None) -> list[float]:
        if fractions is None:
            return [0.0]
        cleaned = []
        for value in fractions:
            try:
                v = float(value)
            except Exception:
                continue
            v = max(0.0, min(1.0, v))
            cleaned.append(v)
        if len(cleaned) == 0:
            return [0.0]
        return sorted(set(cleaned))

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

    def _anchor_tag(self, frame_fraction: float, frame_index: int | None) -> str:
        pct = int(round(frame_fraction * 100.0))
        frame_text = "u" if frame_index is None else str(int(frame_index))
        return f"f{frame_text}_p{pct:03d}"

    def _build_anchor_inputs(self, item: dict) -> list[tuple[float, int | None, Image.Image]]:
        video_path = item.get("video_path")
        if video_path and os.path.exists(video_path):
            total_frames = self._get_total_video_frames(video_path)
            outputs = []
            for frac in self.anchor_frame_fractions:
                target_idx = int(round(frac * max(total_frames - 1, 0)))
                frame, actual_idx = self._read_video_frame(video_path, target_idx)
                if frame is None:
                    continue
                outputs.append((float(frac), actual_idx, frame))
            if len(outputs) > 0:
                return outputs
        fallback = item["input_image"]
        return [(0.0, 0, fallback)]
