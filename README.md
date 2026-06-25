<div align="center">

# TACO: TActile World Model as a Self-COrrector for Scalable VLA Post-Training

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Coming%20Soon-blue)]()
[![Code](https://img.shields.io/badge/Code-Staged%20Release-lightgrey)](#-roadmap)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](./LICENSE)

<p>
  Shengbang Liu<sup>1,3,*</sup>,
  Yueru Jia<sup>1,2,*</sup>,
  Yuyang Yan<sup>1,*</sup>,
  Jiaming Liu<sup>1,*,†</sup>,
  Xinran Zhang<sup>1,2</sup>,
  Qiuxuan Feng<sup>1</sup>,
  Yandong Guo<sup>2</sup>,
  Shiji Zhou<sup>4</sup>,
  Boxin Shi<sup>1</sup>,
  Shanghang Zhang<sup>1</sup>
</p>

<p>
  <sup>1</sup>Peking University&nbsp;&nbsp;
  <sup>2</sup>AI2 Robotics&nbsp;&nbsp;
  <sup>3</sup>Sun Yat-sen University&nbsp;&nbsp;
  <sup>4</sup>Beihang University
</p>

<sub><sup>*</sup>Equal Contribution&nbsp;&nbsp;<sup>†</sup>Project Lead</sub>

</div>

---

TACO is a tactile-aware world-model-driven framework for scalable VLA post-training in contact-rich robot manipulation. Given real-world rollouts, TACO follows a **Recognize-Imagine-Label** loop: it recognizes failure-adjacent contact states, imagines local visuo-tactile correction segments, and labels corrective actions for policy post-training.

The resulting corrective supervision is used for **advantage-conditioned post-training** with **knowledge-insulated tactile adaptation**, allowing the policy to learn contact recovery behaviors without degrading pretrained visual-language priors.

This repository is currently a lightweight public landing repo. Core code, model checkpoints, datasets, and detailed training recipes will be released progressively according to the roadmap below. The paper PDF is not included in this repository; arXiv and project-page links will be filled in once available.

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Method Overview](#-method-overview)
- [Experimental Setup](#-experimental-setup)
- [Roadmap](#-roadmap)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Acknowledgments](#-acknowledgments)
- [Citation](#-citation)

## ✨ Highlights

- **Tactile-aware world model:** jointly denoises future video and 12-D force-torque trajectories for contact-consistent imagination.
- **Recognize-Imagine-Label loop:** converts real-world failures into local visuo-tactile corrections without repeated human intervention.
- **Unified progress-action model:** estimates task progress, localizes failure-adjacent states, and labels imagined segments with executable 7-DoF corrective actions.
- **Knowledge-insulated tactile adaptation:** blocks tactile-action gradients from updating the pretrained VLM backbone, preserving visual-language priors during tactile post-training.
- **Advantage-conditioned VLA post-training:** separates recovery-oriented corrections from failed segments using binary advantage labels.

## 🧠 Method Overview

TACO uses real robot rollouts as the starting point for iterative post-training:

1. **Recognize:** a unified progress-action model predicts dense task progress and selects failure-adjacent contact states where progress stalls or decreases.
2. **Imagine:** a visuo-tactile generation model imagines local correction segments by jointly generating future RGB observations and left/right 6-DoF force-torque signals.
3. **Label:** the progress-action model converts imagined visuo-tactile corrections into executable corrective actions and progress labels.
4. **Post-train:** the VLA action expert is updated with force and advantage conditioning while the pretrained VLM backbone is protected through knowledge insulation.

See [Method Overview](./docs/method_overview.md) for a more detailed summary.

## 🤖 Experimental Setup

TACO is evaluated on a single-arm **Franka Research 3** robot with a front-view **Intel RealSense D455** camera and two fingertip-mounted **Xense tactile sensors**. Each timestep includes RGB observations, 12-D left/right force-torque readings, and proprioceptive state.

The real-world contact-rich tasks are:

| Task | Contact challenge |
| --- | --- |
| Insert Flower | Align and insert a deformable stem into a narrow vase opening |
| Wipe Whiteboard | Apply sufficient contact force to erase the target mark |
| Twist Bottle Cap | Maintain stable grasp and effective twisting torque |
| Play Xylophone | Strike specified keys accurately with a mallet |
| Toast Bread | Pick and insert two bread slices without misalignment |
| Move Hanoi Rings | Grasp, align, and seat rings onto target pegs |

In the paper experiments, TACO improves average success rate by **44% absolute** over the base policy after two post-training iterations.

## 🗓 Roadmap

### 1. Open-source visuo-tactile world model

- [ ] Release model architecture and training code
- [ ] Release data preprocessing scripts for paired RGB and force-torque trajectories
- [ ] Release inference and rollout visualization examples
- [ ] Release pretrained checkpoints

### 2. Open-source tactile-aware VLA

- [ ] Release tactile-aware policy architecture
- [ ] Release fine-tuning configs for visuo-tactile robot datasets
- [ ] Release policy serving and robot-side inference examples
- [ ] Release evaluation scripts and benchmark task configs

### 3. Open-source the full TACO framework

- [ ] Release end-to-end training pipeline
- [ ] Release unified data, model, and evaluation interfaces
- [ ] Release full experiment configs and reproducibility scripts
- [ ] Release documentation for extending TACO to new robots and tactile sensors

## 📁 Repository Structure

```text
TACO/
├── assets/                 # Figures, videos, and project media
├── docs/                   # Guides and release notes
├── examples/               # Minimal runnable examples
├── taco/                   # Core package; released progressively
├── README.md               # Project overview
├── LICENSE                 # Apache-2.0 license
└── .gitignore
```

> 📚 Documentation and examples will be expanded as each module is released.

## 🛠 Installation

The full environment will be documented together with the first code release.

For now, clone the repository:

```bash
git clone https://github.com/liushb9/TACO.git
cd TACO
```

Future releases will provide a reproducible Python environment and installation command, for example:

```bash
pip install -e .
```

## 🚀 Quick Start

Runnable examples will be added with each staged release:

- Visuo-tactile world model training and inference
- Tactile-aware VLA fine-tuning
- End-to-end TACO framework experiments

## 📚 Documentation

- [Method Overview](./docs/method_overview.md)
- [Experiments](./docs/experiments.md)
- [Release Plan](./docs/release_plan.md)
- [Release Notes](./docs/release_notes.md)
- [Examples](./examples)

## 🙏 Acknowledgments

TACO builds on ideas and infrastructure from open-source robotics, VLA, and embodied world-modeling projects, including vision-language-action policies, video generation models, tactile robot learning, and robot world models. We thank the broader research community for making these foundations available.

## 📄 Citation

If you find TACO useful for your research, please consider citing our work. The official BibTeX entry will be updated after the arXiv release.

```bibtex
@misc{taco2026,
  title = {TACO: TActile World Model as a Self-COrrector for Scalable VLA Post-Training},
  author = {Liu, Shengbang and Jia, Yueru and Yan, Yuyang and Liu, Jiaming and Zhang, Xinran and Feng, Qiuxuan and Guo, Yandong and Zhou, Shiji and Shi, Boxin and Zhang, Shanghang},
  year = {2026},
  url = {https://github.com/liushb9/TACO}
}
```
