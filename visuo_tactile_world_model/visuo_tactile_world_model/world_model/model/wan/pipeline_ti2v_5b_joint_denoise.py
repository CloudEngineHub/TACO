"""
Joint denoising pipeline for Wan2.2-TI2V-5B: video + tactile/action sequence
are denoised in a single DiT forward pass, with tactile tokens concatenated
into the self-attention sequence.

See docs/joint_denoise.md and the reference implementation at
DiffSynth-Action/diffsynth/models/wan_video_tactile_joint_denoise.py.
"""

from typing import Optional, Union

import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import Literal

from ...utils.device.npu_compatible_device import get_device_type
from ...utils.gradient import gradient_checkpoint_forward
from ...utils.loader.config import ModelConfig
from .pipeline import WanVideoUnit_PromptEmbedder
from .pipeline_ti2v_5b import WanTI2V5BPipeline
from .wan_video_dit import WanModel, sinusoidal_embedding_1d
from .wan_video_tactile import WanTactileHead, WanTactileTokenizer


def model_fn_wan_video_joint_denoise(
    dit: WanModel,
    tactile_tokenizer: WanTactileTokenizer = None,
    tactile_head: WanTactileHead = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    noisy_tactile: Optional[torch.Tensor] = None,
    tactile_anchor_first: bool = False,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep_flat = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep_flat).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = dit.patchify(x)
    N_vid = x.shape[1]

    has_tactile = noisy_tactile is not None and tactile_tokenizer is not None and tactile_head is not None
    if has_tactile:
        if noisy_tactile.shape[0] != x.shape[0]:
            noisy_tactile = noisy_tactile.expand(x.shape[0], -1, -1)
        tactile_tokens = tactile_tokenizer(noisy_tactile.to(x.dtype), timestep)
        x = torch.cat([x, tactile_tokens], dim=1)

        # Extend t_mod for tactile positions if per-token modulation is in use.
        # When `tactile_anchor_first` is set, position 0 carries the clean GT
        # tactile frame (analogous to the video image-cond first frame), so its
        # modulation uses timestep=0; otherwise all tactile positions share the
        # current timestep.
        if len(t_mod.shape) == 4:
            T_tac = x.shape[1] - N_vid
            cur = timestep[:1].to(dtype=t_mod.dtype, device=t_mod.device)
            if tactile_anchor_first and T_tac > 1:
                tac_ts_flat = torch.cat(
                    [
                        torch.zeros(1, dtype=t_mod.dtype, device=t_mod.device),
                        cur.expand(T_tac - 1),
                    ]
                )
            else:
                tac_ts_flat = cur.expand(T_tac)
            t_tac = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, tac_ts_flat).unsqueeze(0))
            t_mod_tac = dit.time_projection(t_tac).unflatten(2, (6, dit.dim))
            t_mod = torch.cat([t_mod, t_mod_tac], dim=1)

    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    if has_tactile:
        T_tac_freqs = x.shape[1] - N_vid
        # Map tactile positions to the same temporal coordinate as video latent
        # frames so cross-attention can align by time. h/w axes get unit RoPE
        # (1+0j) — i.e. no rotation — instead of zero, which would null Q/K.
        if T_tac_freqs > 1:
            tac_t_pos = torch.linspace(
                0, f - 1, T_tac_freqs, device=dit.freqs[0].device
            ).round().long().clamp(0, f - 1)
        else:
            tac_t_pos = torch.zeros(T_tac_freqs, dtype=torch.long, device=dit.freqs[0].device)
        tac_t = dit.freqs[0][tac_t_pos]  # (T_tac, dim_f/2) complex
        ones_h = torch.ones(
            T_tac_freqs, dit.freqs[1].shape[-1], dtype=tac_t.dtype, device=tac_t.device
        )
        ones_w = torch.ones(
            T_tac_freqs, dit.freqs[2].shape[-1], dtype=tac_t.dtype, device=tac_t.device
        )
        tac_freqs = torch.cat([tac_t, ones_h, ones_w], dim=-1).unsqueeze(1).to(freqs.device)
        freqs = torch.cat([freqs, tac_freqs], dim=0)

    for block in dit.blocks:
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x,
            context,
            t_mod,
            freqs,
        )

    if has_tactile:
        video_tokens = x[:, :N_vid]
        tactile_tokens_out = x[:, N_vid:]
    else:
        video_tokens = x

    video_tokens = dit.head(video_tokens, t)
    video_pred = dit.unpatchify(video_tokens, (f, h, w))

    if has_tactile:
        tactile_pred = tactile_head(tactile_tokens_out)
        return video_pred, tactile_pred
    return video_pred


class WanTI2V5BJointDenoisePipeline(WanTI2V5BPipeline):
    """TI2V 5B pipeline extended with tactile joint-denoising modules."""

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.tactile_tokenizer: WanTactileTokenizer = None
        self.tactile_head: WanTactileHead = None
        self.tactile_dim: int = 12
        self.tactile_hidden_dim: int = 512
        self.tactile_num_layers: int = 2
        self.model_fn = model_fn_wan_video_joint_denoise
        self.in_iteration_models = ("dit", "tactile_tokenizer", "tactile_head")

    @classmethod
    def from_pretrained(
        cls,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list = None,
        tokenizer_config: ModelConfig = None,
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
        tactile_dim: int = 12,
        tactile_hidden_dim: int = 512,
        tactile_num_layers: int = 2,
    ):
        if model_configs is None:
            model_configs = []
        if tokenizer_config is None:
            tokenizer_config = ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            )
        pipe = WanTI2V5BPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            redirect_common_files=redirect_common_files,
            use_usp=use_usp,
            vram_limit=vram_limit,
        )
        pipe.__class__ = cls
        pipe.tactile_dim = tactile_dim
        pipe.tactile_hidden_dim = tactile_hidden_dim
        pipe.tactile_num_layers = tactile_num_layers
        pipe.model_fn = model_fn_wan_video_joint_denoise
        pipe.in_iteration_models = ("dit", "tactile_tokenizer", "tactile_head")

        if pipe.dit is not None:
            model_dim = pipe.dit.dim
            freq_dim = pipe.dit.freq_dim
            pipe.tactile_tokenizer = WanTactileTokenizer(
                tactile_dim=tactile_dim,
                model_dim=model_dim,
                freq_dim=freq_dim,
                hidden_dim=tactile_hidden_dim,
                num_layers=tactile_num_layers,
            ).to(device=device, dtype=torch_dtype)
            pipe.tactile_head = WanTactileHead(
                model_dim=model_dim,
                hidden_dim=tactile_hidden_dim,
                tactile_dim=tactile_dim,
            ).to(device=device, dtype=torch_dtype)
        else:
            pipe.tactile_tokenizer = None
            pipe.tactile_head = None

        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_image: Optional[Image.Image] = None,
        input_video: Optional[list] = None,
        denoising_strength: Optional[float] = 1.0,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        switch_DiT_boundary: Optional[float] = 0.875,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        tiled: Optional[bool] = True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
        tactile_init: Optional[torch.Tensor] = None,
        return_tactile: bool = False,
        context: Optional[torch.Tensor] = None,
    ):
        """Joint sampling: denoise video latents and tactile sequence in lockstep.

        - If `tactile_tokenizer` / `tactile_head` are not attached, this falls back
          to plain video sampling (parent behavior).
        - `tactile_init` seeds the tactile with a clean anchor at frame 0:
          pass either the full GT sequence (shape `(num_frames, D)` or
          `(1, num_frames, D)`) or only the first frame (shape `(D,)`,
          `(1, D)`, or `(1, 1, D)`). Frame 0 stays clean; the rest is noise.
          Default (None) is pure Gaussian noise with no anchor.
        - CFG is applied to the video branch only; tactile uses the positive
          prediction (no meaningful negative prompt for force signals).

        Returns:
            video                   if `return_tactile=False` or tactile modules are missing
            (video, tactile_pred)   otherwise. tactile_pred shape: (1, num_frames, tactile_dim)
        """
        if self.tactile_tokenizer is None or self.tactile_head is None:
            return WanTI2V5BPipeline.__call__(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=input_image,
                input_video=input_video,
                denoising_strength=denoising_strength,
                seed=seed,
                rand_device=rand_device,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_scale=cfg_scale,
                cfg_merge=cfg_merge,
                switch_DiT_boundary=switch_DiT_boundary,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                progress_bar_cmd=progress_bar_cmd,
                output_type=output_type,
            )

        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        if context is not None:
            inputs_posi["context"] = context.to(device=self.device, dtype=self.torch_dtype)
            inputs_nega["context"] = inputs_posi["context"]
        inputs_shared = {
            "input_image": input_image,
            "input_video": input_video,
            "denoising_strength": denoising_strength,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        units = self.units
        if context is not None:
            units = [u for u in units if not isinstance(u, WanVideoUnit_PromptEmbedder)]

        for unit in units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega,
            )

        # ---- tactile sample initialization ----
        # `tactile_init` accepts either:
        #   * full GT sequence — shape (num_frames, D), (1, num_frames, D) — the
        #     whole sequence is noised at timesteps[0] and frame 0 is overwritten
        #     with clean GT (anchor).
        #   * first-frame only ("force_0") — shape (D,), (1, D), or (1, 1, D) —
        #     frames 1..N-1 are pure noise, frame 0 is the clean anchor.
        # In both cases, frame 0 acts as the clean image-cond analogue and is
        # re-asserted after every scheduler step.
        tactile_shape = (1, num_frames, self.tactile_dim)
        tactile_anchor = None
        if tactile_init is not None:
            ti = tactile_init.to(device=self.device, dtype=self.torch_dtype)
            # Normalize to (1, T, D)
            if ti.ndim == 1:
                ti = ti.view(1, 1, -1)
            elif ti.ndim == 2:
                # (T, D) where T is either 1 or num_frames
                ti = ti.unsqueeze(0)
            elif ti.ndim != 3:
                raise ValueError(
                    f"tactile_init must have 1/2/3 dims, got shape {tuple(ti.shape)}"
                )

            if ti.shape == tactile_shape:
                # Full GT path
                tactile_anchor = ti[:, 0:1].clone()
                tactile_noise = torch.randn(tactile_shape, dtype=torch.float32).to(
                    device=self.device, dtype=self.torch_dtype
                )
                tactile = self.scheduler.add_noise(ti, tactile_noise, self.scheduler.timesteps[0])
                tactile[:, 0:1] = tactile_anchor
            elif ti.shape == (1, 1, self.tactile_dim):
                # First-frame-only ("force_0") path
                tactile_anchor = ti.clone()
                tactile = torch.randn(tactile_shape, dtype=torch.float32).to(
                    device=self.device, dtype=self.torch_dtype
                )
                tactile[:, 0:1] = tactile_anchor
            else:
                raise ValueError(
                    f"tactile_init shape {tuple(ti.shape)} not compatible; expected "
                    f"{tactile_shape} (full GT) or (1, 1, {self.tactile_dim}) "
                    f"(first-frame anchor)."
                )
        else:
            tactile = torch.randn(tactile_shape, dtype=torch.float32).to(
                device=self.device, dtype=self.torch_dtype
            )

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            if (
                timestep.item() < switch_DiT_boundary * 1000
                and self.dit2 is not None
                and models["dit"] is not self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2

            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Positive branch — produces both video and tactile predictions.
            video_pred_posi, tactile_pred_posi = self.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                noisy_tactile=tactile,
                tactile_anchor_first=tactile_anchor is not None,
                timestep=timestep,
            )

            if cfg_scale != 1.0:
                if cfg_merge:
                    video_pred_posi, video_pred_nega = video_pred_posi.chunk(2, dim=0)
                    # tactile under cfg_merge: take positive half.
                    tactile_pred_posi, _ = tactile_pred_posi.chunk(2, dim=0)
                else:
                    out_nega = self.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        noisy_tactile=tactile,
                        tactile_anchor_first=tactile_anchor is not None,
                        timestep=timestep,
                    )
                    # We only need the video branch for CFG; tactile_nega is discarded.
                    video_pred_nega = out_nega[0] if isinstance(out_nega, tuple) else out_nega
                video_pred = video_pred_nega + cfg_scale * (video_pred_posi - video_pred_nega)
            else:
                video_pred = video_pred_posi

            # Tactile: no CFG — use positive prediction directly.
            tactile_pred = tactile_pred_posi

            inputs_shared["latents"] = self.scheduler.step(
                video_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

            tactile = self.scheduler.step(
                tactile_pred,
                self.scheduler.timesteps[progress_id],
                tactile,
            )
            if tactile_anchor is not None:
                tactile[:, 0:1] = tactile_anchor

        # ---- decode ----
        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        if return_tactile:
            return video, tactile.float().cpu()
        return video
