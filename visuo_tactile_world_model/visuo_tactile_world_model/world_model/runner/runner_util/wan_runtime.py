import json
import os
from types import SimpleNamespace

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from visuo_tactile_world_model.world_model.dataset import UnifiedDataset
from visuo_tactile_world_model.world_model.dataset.operators import ImageCropAndResize, LoadAudio, LoadNumpyArray, LoadVideo, ToAbsolutePath
from visuo_tactile_world_model.world_model.model.wan.pipeline import WanVideoPipeline
from visuo_tactile_world_model.world_model.utils.loader import ModelConfig, load_state_dict

from .instantiation import flatten_config_params, import_class, resolve_local_wan_tokenizer_path


WAN_I2V_DEFAULTS = {
    "dataset_base_path": "",
    "dataset_metadata_path": None,
    "dataset_repeat": 1,
    "dataset_num_workers": 0,
    "data_file_keys": "image,video",
    "model_paths": None,
    "model_id_with_origin_paths": None,
    "extra_inputs": None,
    "fp8_models": None,
    "offload_models": None,
    "learning_rate": 1e-4,
    "num_epochs": 1,
    "max_train_steps": None,
    "batch_size": 1,
    "data_seed": 42,
    "trainable_models": None,
    "find_unused_parameters": False,
    "weight_decay": 0.01,
    "task": "sft",
    "output_path": "./logs",
    "remove_prefix_in_ckpt": "pipe.dit.",
    "save_steps": None,
    "lora_base_model": None,
    "lora_target_modules": "q,k,v,o,ffn.0,ffn.2",
    "lora_rank": 32,
    "lora_checkpoint": None,
    "preset_lora_path": None,
    "preset_lora_model": None,
    "use_gradient_checkpointing": False,
    "use_gradient_checkpointing_offload": False,
    "gradient_accumulation_steps": 1,
    "height": None,
    "width": None,
    "max_pixels": 1024 * 1024,
    "num_frames": 81,
    "fps": None,
    "video_sampling_mode": "prefix",
    "resize_mode": "stretch",
    "context_window_short_video_mode": "drop",
    "context_window_stride": 81,
    "context_window_tail_align": False,
    "context_window_wait_timeout": 7200,
    "tokenizer_path": None,
    "audio_processor_path": None,
    "max_timestep_boundary": 1.0,
    "min_timestep_boundary": 0.0,
    "initialize_model_on_cpu": False,
    "wandb_enabled": False,
    "wandb_project": "TACO",
    "wandb_entity": None,
    "wandb_run_name": None,
    "wandb_mode": "online",
    "wandb_log_every": 10,
    "wandb_log_video": True,
    "validation_every_steps": 1000,
    "validation_extra_steps": [],
    "validation_num_samples": 1,
    "validation_fps": 16,
    "validation_quality": 5,
    "validation_seed_base": 0,
    "validation_anchor_frame_fractions": [0.0],
    "validation_input_image_resize_mode": "stretch",
    "validation_infer_kwargs": {},
    "validation_video_metrics": ["psnr", "ssim"],
    "dit_checkpoint_overlays": None,
    # Joint-denoise specific
    "tactile_dim": 12,
    "tactile_loss_weight": 0.1,
    "tactile_hidden_dim": 512,
    "tactile_num_layers": 2,
    "tactile_sequence_key": "force_sequence",
    "tactile_stats_path": None,
}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_runtime_types(runtime: dict):
    int_keys = {
        "dataset_repeat",
        "dataset_num_workers",
        "num_epochs",
        "max_train_steps",
        "batch_size",
        "data_seed",
        "save_steps",
        "lora_rank",
        "gradient_accumulation_steps",
        "height",
        "width",
        "max_pixels",
        "num_frames",
        "tactile_dim",
        "tactile_hidden_dim",
        "tactile_num_layers",
        "context_window_stride",
        "context_window_wait_timeout",
        "wandb_log_every",
        "validation_every_steps",
        "validation_num_samples",
        "validation_fps",
        "validation_quality",
        "validation_seed_base",
    }
    float_keys = {
        "learning_rate",
        "weight_decay",
        "fps",
        "max_timestep_boundary",
        "min_timestep_boundary",
        "tactile_loss_weight",
    }
    bool_keys = {
        "find_unused_parameters",
        "use_gradient_checkpointing",
        "use_gradient_checkpointing_offload",
        "context_window_tail_align",
        "initialize_model_on_cpu",
        "wandb_enabled",
        "wandb_log_video",
    }

    for key in int_keys:
        if runtime.get(key) is not None:
            runtime[key] = int(runtime[key])
    for key in float_keys:
        if runtime.get(key) is not None:
            runtime[key] = float(runtime[key])
    for key in bool_keys:
        if runtime.get(key) is not None:
            runtime[key] = _as_bool(runtime[key])
    if runtime.get("validation_anchor_frame_fractions") is not None:
        fractions = runtime["validation_anchor_frame_fractions"]
        if isinstance(fractions, str):
            text = fractions.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    fractions = json.loads(text)
                except Exception:
                    fractions = []
            else:
                fractions = [x.strip() for x in text.split(",") if x.strip()]
        if not isinstance(fractions, (list, tuple)):
            fractions = [fractions]
        runtime["validation_anchor_frame_fractions"] = [float(x) for x in fractions]

    # validation_video_metrics: accept comma-separated string or list
    vm = runtime.get("validation_video_metrics", ["psnr", "ssim"])
    if isinstance(vm, str):
        vm = [x.strip() for x in vm.split(",") if x.strip()]
    runtime["validation_video_metrics"] = list(vm) if vm else ["psnr", "ssim"]

    return runtime


def build_wan_i2v_runtime_args(cfg: dict, force_task: str | None = None):
    params = flatten_config_params(cfg)
    runtime = dict(WAN_I2V_DEFAULTS)
    runtime.update(params)
    runtime = _coerce_runtime_types(runtime)
    if isinstance(runtime.get("model_paths"), list):
        runtime["model_paths"] = json.dumps(runtime["model_paths"])
    if force_task is not None:
        runtime["task"] = force_task
    return SimpleNamespace(**runtime)


def parse_vram_config(fp8=False, offload=False, cpu_offload=False, device="cpu"):
    if fp8:
        return {
            "offload_dtype": torch.float8_e4m3fn,
            "offload_device": device,
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": device,
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": device,
            "computation_dtype": torch.bfloat16,
            "computation_device": device,
        }
    if offload:
        return {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": device,
            "computation_dtype": torch.bfloat16,
            "computation_device": device,
            "clear_parameters": True,
        }
    if cpu_offload:
        # Per-layer CPU<->GPU swap. Weights live on CPU as bf16; computation()
        # casts to GPU bf16 each forward (no persistent GPU residency unless
        # vram_limit lets preparing() pin the module on GPU).
        return {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": device,
            "computation_dtype": torch.bfloat16,
            "computation_device": device,
        }
    return {}


def parse_path_or_model_id(model_id_with_origin_path, default_value=None):
    if model_id_with_origin_path is None:
        return default_value
    if os.path.exists(model_id_with_origin_path):
        return ModelConfig(path=model_id_with_origin_path)
    if ":" not in model_id_with_origin_path:
        raise ValueError(
            f"Failed to parse model config: {model_id_with_origin_path}. "
            "This is neither a valid path nor in the format of `model_id/origin_file_pattern`."
        )
    split_id = model_id_with_origin_path.rfind(":")
    model_id = model_id_with_origin_path[:split_id]
    origin_file_pattern = model_id_with_origin_path[split_id + 1 :]
    return ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern)


def parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=None, offload_models=None, cpu_offload_models=None, device="cpu"):
    if fp8_models is None:
        fp8_models = []
    elif isinstance(fp8_models, str):
        fp8_models = [item for item in fp8_models.split(",") if item]
    else:
        fp8_models = list(fp8_models)

    if offload_models is None:
        offload_models = []
    elif isinstance(offload_models, str):
        offload_models = [item for item in offload_models.split(",") if item]
    else:
        offload_models = list(offload_models)

    if cpu_offload_models is None:
        cpu_offload_models = []
    elif isinstance(cpu_offload_models, str):
        cpu_offload_models = [item for item in cpu_offload_models.split(",") if item]
    else:
        cpu_offload_models = list(cpu_offload_models)

    model_configs = []
    if model_paths is not None:
        if isinstance(model_paths, str):
            model_paths = json.loads(model_paths)
        else:
            model_paths = list(model_paths)
        for path in model_paths:
            vram_config = parse_vram_config(
                fp8=path in fp8_models,
                offload=path in offload_models,
                cpu_offload=path in cpu_offload_models,
                device=device,
            )
            model_configs.append(ModelConfig(path=path, **vram_config))

    if model_id_with_origin_paths is not None:
        if isinstance(model_id_with_origin_paths, str):
            model_id_with_origin_paths = [item for item in model_id_with_origin_paths.split(",") if item]
        else:
            model_id_with_origin_paths = list(model_id_with_origin_paths)
        for model_id_with_origin_path in model_id_with_origin_paths:
            vram_config = parse_vram_config(
                fp8=model_id_with_origin_path in fp8_models,
                offload=model_id_with_origin_path in offload_models,
                cpu_offload=model_id_with_origin_path in cpu_offload_models,
                device=device,
            )
            config = parse_path_or_model_id(model_id_with_origin_path)
            model_configs.append(
                ModelConfig(
                    model_id=config.model_id,
                    origin_file_pattern=config.origin_file_pattern,
                    **vram_config,
                )
            )
    return model_configs


def build_wan_i2v_pipeline_from_params(model_params: dict, device_override=None) -> WanVideoPipeline:
    model_paths = model_params.get("model_paths")
    model_id_with_origin_paths = model_params.get("model_id_with_origin_paths")
    fp8_models = model_params.get("fp8_models")
    offload_models = model_params.get("offload_models")
    cpu_offload_models = model_params.get("cpu_offload_models")
    tokenizer_path = model_params.get("tokenizer_path")
    audio_processor_path = model_params.get("audio_processor_path")
    dit_checkpoint_overlays = model_params.get("dit_checkpoint_overlays")
    pipeline_class_path = model_params.get(
        "pipeline_class_path",
        "visuo_tactile_world_model.world_model.model.wan.pipeline.WanVideoPipeline",
    )
    torch_dtype_value = model_params.get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, torch_dtype_value) if isinstance(torch_dtype_value, str) else torch_dtype_value
    device = device_override or model_params.get("device", "cpu")
    pipeline_cls = import_class(pipeline_class_path) if pipeline_class_path else WanVideoPipeline

    model_configs = parse_model_configs(
        model_paths=model_paths,
        model_id_with_origin_paths=model_id_with_origin_paths,
        fp8_models=fp8_models,
        offload_models=offload_models,
        cpu_offload_models=cpu_offload_models,
        device=device,
    )
    local_tokenizer_path = tokenizer_path or resolve_local_wan_tokenizer_path(model_paths)
    if local_tokenizer_path is not None:
        tokenizer_config = ModelConfig(path=local_tokenizer_path)
        print(f"[Tokenizer] Using local tokenizer path: {local_tokenizer_path}")
    else:
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        )
    audio_processor_config = parse_path_or_model_id(audio_processor_path)
    extra_pipeline_kwargs = {}
    for key in (
        "tactile_dim",
        "tactile_hidden_dim",
        "tactile_num_layers",
        "train_ckpt_path",
        "vram_limit",
    ):
        if key in model_params and model_params[key] is not None:
            extra_pipeline_kwargs[key] = model_params[key]
    import inspect as _inspect
    accepted = set(_inspect.signature(pipeline_cls.from_pretrained).parameters)
    extra_pipeline_kwargs = {k: v for k, v in extra_pipeline_kwargs.items() if k in accepted}
    pipe = pipeline_cls.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        audio_processor_config=audio_processor_config,
        **extra_pipeline_kwargs,
    )
    apply_dit_checkpoint_overlays(pipe, dit_checkpoint_overlays, model_paths=model_paths)
    return pipe


def _normalize_optional_path_list(paths):
    if paths is None:
        return []
    if isinstance(paths, str):
        return [item for item in paths.split(",") if item]
    return list(paths)


def _load_state_dict_into_module(module, state_dict):
    if is_deepspeed_zero3_enabled():
        from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model

        _load_state_dict_into_zero3_model(module, state_dict)
        return None
    return module.load_state_dict(state_dict, strict=False)


def _should_skip_dit_overlay(model_paths):
    if model_paths is None:
        return False
    if isinstance(model_paths, str):
        try:
            model_paths = json.loads(model_paths)
        except Exception:
            return False
    if not isinstance(model_paths, list) or len(model_paths) == 0:
        return False

    # Our WoW base config stores DiT shards as a nested list. When `run.py --ckpt`
    # overrides the DiT weights, model_paths[0] becomes a single checkpoint path.
    # In that case the user expects the explicit checkpoint to be used as-is.
    return isinstance(model_paths[0], str)


def apply_dit_checkpoint_overlays(pipe, overlay_paths, model_paths=None):
    overlay_paths = _normalize_optional_path_list(overlay_paths)
    if len(overlay_paths) == 0:
        return
    if getattr(pipe, "dit", None) is None:
        raise ValueError("Config requests dit_checkpoint_overlays, but the pipeline has no `dit` model.")
    if _should_skip_dit_overlay(model_paths):
        print(
            "[DiTOverlay] Skip overlay because model_paths[0] is an explicit single-checkpoint DiT load. "
            "This usually means `--ckpt` was provided, so the checkpoint should not be re-overwritten by overlay weights."
        )
        return

    for overlay_path in overlay_paths:
        print(f"[DiTOverlay] Loading overlay checkpoint: {overlay_path}")
        state_dict = load_state_dict(overlay_path, torch_dtype=pipe.torch_dtype, device="cpu")
        load_result = _load_state_dict_into_module(pipe.dit, state_dict)
        if load_result is None:
            print("[DiTOverlay] Loaded under DeepSpeed ZeRO-3.")
            continue
        missing_keys, unexpected_keys = load_result
        print(
            f"[DiTOverlay] Loaded with strict=False. "
            f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
        )


def build_wan_training_dataset(args, use_data_process_controls=False):
    if use_data_process_controls:
        if getattr(args, "video_sampling_mode", "prefix") == "uniform_full_video":
            target_fps = 24
            fix_frame_rate = False
        else:
            target_fps = args.fps if args.fps is not None else 24
            fix_frame_rate = args.fps is not None
        frame_processor = ImageCropAndResize(
            args.height,
            args.width,
            args.max_pixels,
            16,
            16,
            resize_mode=args.resize_mode,
        )
    else:
        target_fps = 24
        fix_frame_rate = False
        frame_processor = None

    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            frame_rate=target_fps,
            fix_frame_rate=fix_frame_rate,
            frame_processor=frame_processor,
            video_sampling_mode=getattr(args, "video_sampling_mode", "prefix"),
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(
                args.num_frames,
                4,
                1,
                frame_processor=ImageCropAndResize(
                    512,
                    512,
                    None,
                    16,
                    16,
                    resize_mode=args.resize_mode if use_data_process_controls else "crop",
                ),
                frame_rate=target_fps,
                fix_frame_rate=fix_frame_rate,
                video_sampling_mode=getattr(args, "video_sampling_mode", "prefix"),
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "force_sequence": ToAbsolutePath(args.dataset_base_path) >> LoadNumpyArray(num_frames=args.num_frames),
            "tactile_sequence": ToAbsolutePath(args.dataset_base_path) >> LoadNumpyArray(num_frames=args.num_frames),
        },
    )
