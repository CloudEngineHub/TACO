<div align="center">

# TACO: TActile World Model as a Self-COrrector for Scalable Robot Policy Post-Training

[![arXiv](https://img.shields.io/badge/arXiv-2607.02840v2-b31b1b.svg)](https://arxiv.org/pdf/2607.02840v2)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://taco-wm.github.io/)
[![Code](https://img.shields.io/badge/Code-Staged%20Release-lightgrey)](#roadmap)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](./LICENSE)

<p>
  <a href="https://liushb9.github.io/">Shengbang Liu</a><sup>1,3,*</sup>&nbsp;&nbsp;
  <a href="https://jiayueru.github.io/">Yueru Jia</a><sup>1,2,*</sup>&nbsp;&nbsp;
  <a href="https://github.com/avx34/">Yuyang Yan</a><sup>1,*</sup>&nbsp;&nbsp;
  <a href="https://liujiaming1996.github.io/">Jiaming Liu</a><sup>1,*,†</sup>&nbsp;&nbsp;
  <a href="https://github.com/XinranJoy">Xinran Zhang</a><sup>1,2,*</sup>&nbsp;&nbsp;
  <a href="https://github.com/xuanxuanzzzii">Qiuxuan Feng</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=fWDoWsQAAAAJ&hl=en">Yandong Guo</a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://arnoldshijizhou.github.io/">Shiji Zhou</a><sup>4</sup>&nbsp;&nbsp;
  <a href="https://camera.pku.edu.cn/">Boxin Shi</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=voqw10cAAAAJ&hl=en">Shanghang Zhang</a><sup>1,📧</sup>
</p>

<p>
  <sup>1</sup>State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University<br>
  <sup>2</sup>AI2 Robotics&nbsp;&nbsp;
  <sup>3</sup>Sun Yat-sen University&nbsp;&nbsp;
  <sup>4</sup>Beihang University
</p>

<sub><sup>*</sup>Equal Contribution&nbsp;&nbsp;<sup>†</sup>Project Lead&nbsp;&nbsp;<sup>📧</sup>Corresponding Author</sub>

</div>

---

<p align="center">
  <img src="./assets/overview.png" alt="TACO overview" width="100%">
</p>

TACO is a scalable **robot policy post-training** framework for contact-rich manipulation, built on a **compositional tactile world model**. It combines a **visuo-tactile generation model** with an **inverse dynamics and value model (IDVM)** to turn real rollout failures into local corrective supervision through a **Recognize-Imagine-Label** loop.

TACO filters imagined corrections for kinematic feasibility and tactile plausibility, selects them by predicted progress gain, and aggregates them with demonstrations and real rollouts. Policy updates combine **knowledge-insulated tactile adaptation** with **CFG-RL**, using binary advantage labels while keeping the pretrained VLM backbone fixed.

Across six real-world tasks, TACO improves the average task score from **0.375 to 0.825** after two post-training iterations.

This repository contains staged releases of the **visuo-tactile generation model** and **Tactile VLA**. The IDVM, full iterative correction pipeline, model checkpoints, and datasets are not included in the current release.

## 📋 Table of Contents

- [Method](#method)
- [Results](#results)
- [Roadmap](#roadmap)
- [Repository Structure](#repository-structure)
- [Visuo-Tactile Generation Model](#visuo-tactile-generation-model)
- [Tactile VLA](#tactile-vla)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Method

### Recognize-Imagine-Label

1. **Recognize:** the IDVM estimates signed task progress from synchronized RGB and tactile inputs, locating nonterminal states in failed rollouts where progress stalls or decreases.
2. **Imagine:** the visuo-tactile generation model jointly denoises future video and 12D left/right force-torque sequences conditioned on the anchor observation and task instruction.
3. **Label:** the IDVM predicts corrective actions and progress scores for the imagined segments. Candidates are checked for kinematic feasibility and tactile plausibility, then ranked by predicted progress gain.

The paper samples 32 candidates per anchor and retains up to 8 valid corrections above the progress-gain threshold. Each segment contains 49 frames, including the anchor.

<p align="center">
  <img src="./assets/pipeline.png" alt="Recognize-Imagine-Label loop with correction filtering and knowledge-insulated policy updates" width="100%">
</p>

### Compositional Tactile World Model

The **visuo-tactile generation model** builds on Wan2.2-TI2V-5B and jointly processes video and tactile tokens in DiT self-attention. The paper aligns tactile tokens with the VAE-compressed video timeline using temporal RoPE and keeps the first tactile reading clean as an anchor. The **IDVM** combines DINOv2 visual features with an MLP tactile encoder to predict 7D robot actions and signed progress scores.

<p align="center">
  <img src="./assets/model_architecture.png" alt="Visuo-tactile generation model, inverse dynamics and value model, and temporal RoPE alignment" width="100%">
</p>

### Knowledge-Insulated Post-Training

The policy uses a pi0.5-DROID backbone. The pretrained VLM is frozen, while the tactile encoder, adaptation layers, and action expert are updated. Tactile history and the CFG-RL advantage label condition the action expert through adaRMSNorm.

Demonstrations, successful real rollouts, and selected corrections receive label `1`. Failed rollouts receive label `1` before failure onset and `0` from failure onset onward. Training drops the advantage label with probability `0.1`; inference combines positive-label and null-label velocity predictions with classifier-free guidance.

## Results

Results from Table I of the [v2 paper](https://arxiv.org/pdf/2607.02840v2), averaged across Insert Flower, Wipe Whiteboard, Twist Bottle Cap, Play Xylophone, Toast Bread, and Move Hanoi Rings:

| Method | Post-training iteration | Task score (higher is better) | Completion steps (lower is better) |
|---|---:|---:|---:|
| Base Policy | 0 | 0.375 | 185.5 |
| RFT | 2 | 0.425 | 164.5 |
| DreamGen | 2 | 0.550 | 131.7 |
| TACO | 1 | 0.650 | 131.2 |
| **TACO** | **2** | **0.825** | **117.0** |

Each method is evaluated over 40 episodes per task. Task score credits partial completion; completion steps are averaged over successful episodes only.

## Roadmap

- [x] Release the visuo-tactile generation model
- [x] Release Tactile VLA with knowledge insulation, tactile conditioning, and CFG-RL support
- [ ] Release the inverse dynamics and value model (IDVM)
- [ ] Release the full iterative correction and post-training pipeline
- [ ] Release model checkpoints and datasets

## Repository Structure

```text
TACO/
├── assets/                         # Figures, videos, and project media
├── visuo_tactile_generation_model/ # Joint video/tactile generation
│   ├── configs/                    # Accelerate / distributed runtime configs
│   ├── examples/                   # Minimal YAML example
│   ├── scripts/                    # Data preparation and tactile utility scripts
│   ├── visuo_tactile_world_model/
│   │   └── world_model/            # Visuo-tactile world-model Python package
│   ├── run.py                      # YAML config launcher
│   └── pyproject.toml              # Editable install metadata
├── tactile_vla/                   # Tactile VLA with knowledge insulation and CFG-RL
│   ├── examples/taco/              # HDF5-to-LeRobot tactile data conversion
│   ├── scripts/                    # Norm stats, training, and policy serving
│   ├── src/openpi/                 # pi0.5 tactile + advantage + KI implementation
│   ├── README.md                   # Module setup and training guide
│   └── pyproject.toml              # Editable install metadata
├── README.md                       # Project overview
├── LICENSE                         # Apache-2.0 license
└── .gitignore
```

Each released module keeps its setup notes in the module README. The Python package names `visuo_tactile_world_model` and `openpi` are retained for compatibility with existing imports and configs.

## Visuo-Tactile Generation Model

The generation component lives in [`visuo_tactile_generation_model`](./visuo_tactile_generation_model). It includes joint video/tactile denoising, force-sequence loading, training/cache runners, inference support, and example configs. See its [README](./visuo_tactile_generation_model/README.md) for setup and implementation notes, including the current temporal RoPE mapping. The IDVM is a separate component of the paper's compositional world model and is not yet released here.

## Tactile VLA

The policy component lives in [`tactile_vla`](./tactile_vla). It provides pi0.5-style flow-matching training with an 8-step force history, binary advantage conditioning, label dropout, knowledge insulation, and classifier-free guidance support. See its [README](./tactile_vla/README.md) for data conversion, training, inference, and guidance configuration.

## Acknowledgement

We thank the [LightEWM project](https://github.com/XuWuLingYu/LightEWM) for its valuable codebase and engineering foundation, which informed the development of the visuo-tactile world model release.

The policy implementation builds on [OpenPI](https://github.com/Physical-Intelligence/openpi) from Physical Intelligence.

## Citation

```bibtex
@article{liu2026taco,
  title={TACO: TActile World Model as a Self-COrrector for Scalable Robot Policy Post-Training},
  author={Liu, Shengbang and Jia, Yueru and Yan, Yuyang and Liu, Jiaming and Zhang, Xinran and Feng, Qiuxuan and Guo, Yandong and Zhou, Shiji and Shi, Boxin and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2607.02840},
  year={2026},
  url={https://arxiv.org/abs/2607.02840v2}
}
```
