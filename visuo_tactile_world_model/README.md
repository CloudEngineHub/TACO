# Visuo-Tactile World Model

This directory contains the first staged TACO release: a Wan-style visuo-tactile world model with joint video/tactile denoising, tactile/force sequence loading, training/cache runners, inference support, and example configs.

## Structure

```text
visuo_tactile_world_model/
├── configs/          # Accelerate / distributed runtime configs
├── examples/         # Minimal training/cache YAML example
├── scripts/          # Data preparation and tactile utility scripts
├── visuo_tactile_world_model/world_model/ # Python package
├── run.py            # YAML config launcher
└── pyproject.toml    # Editable install metadata
```

## Install

```bash
cd visuo_tactile_world_model
pip install -e .
```

For training with ZeRO/DeepSpeed or robotics datasets:

```bash
pip install -e ".[train,robotics]"
```

## Required Downloads

```bash
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir <WAN_BASE_DIR>
hf download XuWuLingYu/Wan2.2-5B-Robot --local-dir <ROBOT_DIT_DIR>
```

| Placeholder | Expected local path | Hugging Face |
|---|---|---|
| `<TOKENIZER_DIR>` | `<WAN_BASE_DIR>/google/umt5-xxl` | https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B |
| `<ROBOT_DIT_CHECKPOINT>` | `<ROBOT_DIT_DIR>/checkpoint.safetensors` | https://huggingface.co/XuWuLingYu/Wan2.2-5B-Robot |
| `<TEXT_ENCODER_CHECKPOINT>` | `<WAN_BASE_DIR>/models_t5_umt5-xxl-enc-bf16.pth` | https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B |
| `<VAE_CHECKPOINT>` | `<WAN_BASE_DIR>/Wan2.2_VAE.pth` | https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B |

## RoPE Design Reference

The visuo-tactile joint RoPE logic is implemented in `visuo_tactile_world_model/world_model/model/wan/pipeline_ti2v_5b_joint_denoise.py`, inside `model_fn_wan_video_joint_denoise`. The relevant block builds standard video 3D RoPE frequencies for `(f, h, w)` video tokens, then appends tactile-token frequencies before the DiT self-attention blocks.

For tactile tokens, the implementation maps tactile positions onto the video latent temporal axis with `torch.linspace(0, f - 1, T_tac).round()`, reuses `dit.freqs[0]` for temporal RoPE, and uses unit complex frequencies for the spatial `h/w` axes. This gives tactile tokens temporal alignment with video tokens while applying no spatial rotation.

Related files:

- `visuo_tactile_world_model/world_model/model/wan/pipeline_ti2v_5b_joint_denoise.py`: tactile-token RoPE extension and video/tactile token concatenation.
- `visuo_tactile_world_model/world_model/model/wan/wan_video_dit.py`: base Wan 3D RoPE utilities, including `precompute_freqs_cis_3d`, `precompute_freqs_cis`, and `rope_apply`.
- `visuo_tactile_world_model/world_model/model/wan/wan_video_tactile.py`: tactile tokenizer 1D sinusoidal temporal position embedding before projection into DiT token space.

## Data Format

The training metadata is a CSV file. Paths in `video` and `force_sequence` can be relative to `dataset.params.dataset_base_path`.

| Column | Required | Description |
|---|---:|---|
| `video` | yes | Path to the RGB rollout video. |
| `force_sequence` | yes | Path to a `.npy` tactile/force array with shape `(T, 12)`. |
| `prompt` | yes | Text prompt for the rollout. |
| `demo_id` | recommended | Stable sample or episode id used in logs. |
| `camera_key` | recommended | Camera name used in validation output names. |
| `num_frames` | optional | Raw rollout length; used by context-window expansion when present. |

A minimal dataset can be organized as:

```text
<DATA_ROOT>/
├── metadata.csv
└── <task_name>/
    └── episode_000001/
        ├── video.mp4
        └── force.npy        # shape: (T, 12)
```

If your rollout folders follow the `<DATA_ROOT>/<task_name>/<episode>/` layout, you can build a metadata CSV with:

```bash
python scripts/build_worldrl_stride5_joint_metadata.py \
  --stride5-root '<DATA_ROOT>' \
  --tasks '<TASK_NAME>' \
  --output '<METADATA_CSV>'
```

## Quick Start

1. Compute tactile normalization statistics from the metadata.

```bash
python scripts/compute_force_stats.py \
  --metadata '<METADATA_CSV>' \
  --base '<DATA_ROOT>' \
  --tactile-key force_sequence \
  --output '<TACTILE_STATS_NPZ>'
```

2. Pre-cache video latents and tactile windows. This uses the same example YAML, but overrides the runner to the cache runner.

```bash
python run.py --config examples/WorldRLSingleTaskNF37/visuo_tactile_joint_train.yaml \
  --overrides \
  runner.class_path=visuo_tactile_world_model.world_model.runner.wan.wan_data_preprocess.WanCacheRunner \
  'runner.params.output_path=<LATENT_CACHE_DIR>' \
  'dataset.params.dataset_base_path=<DATA_ROOT>' \
  'dataset.params.dataset_metadata_path=<METADATA_CSV>' \
  'model.params.tokenizer_path=<TOKENIZER_DIR>' \
  'model.params.model_paths=[<ROBOT_DIT_CHECKPOINT>, <TEXT_ENCODER_CHECKPOINT>, <VAE_CHECKPOINT>]'
```

3. Train from the latent cache. Set the same placeholders in the YAML, or pass them with `--overrides`.

```bash
python run.py --config examples/WorldRLSingleTaskNF37/visuo_tactile_joint_train.yaml \
  --overrides \
  'dataset.params.dataset_base_path=<LATENT_CACHE_DIR>' \
  'validation_dataset.params.base_path=<DATA_ROOT>' \
  'validation_dataset.params.metadata_path=<METADATA_CSV>' \
  'runner.params.tactile_stats_path=<TACTILE_STATS_NPZ>' \
  'validation_dataset.params.stats_path=<TACTILE_STATS_NPZ>' \
  'model.params.tokenizer_path=<TOKENIZER_DIR>' \
  'model.params.model_paths=[<ROBOT_DIT_CHECKPOINT>, <TEXT_ENCODER_CHECKPOINT>, <VAE_CHECKPOINT>]'
```

The pre-cache step stores raw tactile values in the latent cache. Training and validation use the same `<TACTILE_STATS_NPZ>` file so tactile inputs and targets are normalized consistently.

## Acknowledgement

This module builds on the engineering foundation of [LightEWM](https://github.com/XuWuLingYu/LightEWM), especially its Wan2.2-TI2V-5B training and inference structure.
