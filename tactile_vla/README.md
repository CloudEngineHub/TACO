# Tactile VLA

This module implements TACO's knowledge-insulated policy post-training component described in [the v2 paper](https://arxiv.org/pdf/2607.02840v2). It extends pi0.5-style flow matching with tactile force-history conditioning and CFG-RL: binary advantage labels, label dropout, and classifier-free guidance for contact-rich manipulation.

The directory was renamed from `tactile_aware_vla` to `tactile_vla`. The `openpi` Python package and existing training config names are unchanged.

The released training path uses:

- `force_history`: an 8-step, 12D left/right force-torque history.
- `advantage`: a binary CFG-RL label (`1` for positive behavior, `0` for failure behavior).
- CFG-RL: advantage-label dropout during training and positive/null-label guidance during inference.
- Knowledge insulation: the pretrained VLM prefix path is frozen for action/tactile loss, while action-side modules, `ForceEncoder`, `AdvantageEncoder`, and the null-advantage embedding remain trainable.

## Install

```bash
cd tactile_vla
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The base pi0.5-DROID checkpoint is loaded from:

```text
gs://openpi-assets/checkpoints/pi05_droid/params
```

New tactile and advantage parameters are initialized from scratch.

## Data Format

The training config expects a local LeRobot dataset named `taco_tactile_ki_advantage_example` under `HF_LEROBOT_HOME`.

Each frame should contain:

| Key | Shape | Description |
|---|---:|---|
| `image` | `(H, W, 3)` | Front RGB image. |
| `wrist_image` | `(H, W, 3)` | Wrist RGB image, or duplicated front image. |
| `state` | `(7,)` | End-effector pose plus gripper value. |
| `actions` | `(7,)` | Action target in the same convention as `state`. |
| `force_history` | `(8, 12)` | Left 6D wrench plus right 6D wrench over an 8-step history. |
| `advantage` | `(1,)` | Binary CFG-RL label (`0` or `1`). |
| `task` | string | Language instruction. |

In the paper, demonstrations, successful rollouts, and retained imagined corrections receive label `1`; failed rollouts receive `1` before failure onset and `0` from failure onset onward. The converter below assigns a constant `--advantage-value`, so mixed-success rollout data requires frame-level labels prepared according to this rule before training. Automatic failure localization and correction selection belong to the full pipeline, which is not included in this release.

## Convert HDF5 Data

For WorldRL-style HDF5 episodes:

```bash
uv run examples/taco/convert_worldrl_hdf5_to_lerobot_tactile.py \
  --data-dir /path/to/hdf5_episodes \
  --output-dir /path/to/lerobot_root \
  --repo-name taco_tactile_ki_advantage_example \
  --task "pick up the object" \
  --stride 5 \
  --force-history-len 8 \
  --advantage-value 1.0 \
  --overwrite
```

Then point LeRobot to the output root:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot_root
```

## Train

First compute normalization statistics:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_taco_tactile_ki_advantage
```

Then launch training:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_taco_tactile_ki_advantage --exp-name taco_tactile_run --overwrite
```

The config is defined in `src/openpi/training/config.py` as `pi05_taco_tactile_ki_advantage`.

## Verify KI

Training prints trainable and frozen parameter paths at startup. In the expected setup:

- `PaliGemma/img/...` and the non-action-expert `PaliGemma/llm/...` prefix paths are frozen.
- Action expert paths, `action_in_proj`, `action_out_proj`, `time_mlp_*`, `force_encoder`, `advantage_encoder`, and `null_advantage` are trainable.

You can run the focused model test with:

```bash
uv run pytest src/openpi/models/tactile_ki_advantage_test.py
```

## Inference

Serve a trained checkpoint with:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_taco_tactile_ki_advantage \
  --policy.dir=checkpoints/pi05_taco_tactile_ki_advantage/taco_tactile_run/<step>
```

The policy input must include the same image, state, prompt, force-history, and advantage fields used during training.

For positive-label inference, supply `advantage=1`. The model supports CFG-RL guidance as `u_cfg = u_null + cfg_scale * (u_positive - u_null)`. The example config uses `adv_dropout_prob=0.1` and `cfg_scale=1.0`; set the model's `cfg_scale` above `1.0` in `src/openpi/training/config.py` to enable guidance beyond the positive-label prediction. The null branch drops only the advantage condition and preserves the observation and tactile history.

## Acknowledgement

This module builds on the open-source pi0.5 implementation from Physical Intelligence and adds the tactile-aware TACO adaptation path.
