# Visuo-Tactile World Model

This directory contains the first staged TACO release: a Wan-style visuo-tactile world model with joint video/tactile denoising, tactile/force sequence loading, training/cache runners, inference support, and example configs.

## Structure

```text
visuo_tactile_world_model/
├── configs/          # Accelerate / distributed runtime configs
├── docs/             # World-model guides and release notes
├── examples/         # Training, cache, and inference YAML examples
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

## RoPE Design Reference

The visuo-tactile joint RoPE logic is implemented in `visuo_tactile_world_model/world_model/model/wan/pipeline_ti2v_5b_joint_denoise.py`, inside `model_fn_wan_video_joint_denoise`. The relevant block builds standard video 3D RoPE frequencies for `(f, h, w)` video tokens, then appends tactile-token frequencies before the DiT self-attention blocks.

For tactile tokens, the implementation maps tactile positions onto the video latent temporal axis with `torch.linspace(0, f - 1, T_tac).round()`, reuses `dit.freqs[0]` for temporal RoPE, and uses unit complex frequencies for the spatial `h/w` axes. This gives tactile tokens temporal alignment with video tokens while applying no spatial rotation.

Related files:

- `visuo_tactile_world_model/world_model/model/wan/pipeline_ti2v_5b_joint_denoise.py`: tactile-token RoPE extension and video/tactile token concatenation.
- `visuo_tactile_world_model/world_model/model/wan/wan_video_dit.py`: base Wan 3D RoPE utilities, including `precompute_freqs_cis_3d`, `precompute_freqs_cis`, and `rope_apply`.
- `visuo_tactile_world_model/world_model/model/wan/wan_video_tactile.py`: tactile tokenizer 1D sinusoidal temporal position embedding before projection into DiT token space.

## Quick Start

```bash
python run.py examples/WorldRLSingleTaskNF37/visuo_tactile_joint_train.yaml
```

Before running, edit paths under `model.params`, `dataset.params`, and `validation_dataset.params` to point to local Wan checkpoints, metadata CSVs, latent caches, and tactile statistics.

See `docs/tactile_world_model.md` for architecture, data schema, cache workflow, and inference details.
