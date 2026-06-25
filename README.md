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
  <sup>1</sup>State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University&nbsp;&nbsp;
  <sup>2</sup>AI2 Robotics&nbsp;&nbsp;
  <sup>3</sup>Sun Yat-sen University&nbsp;&nbsp;
  <sup>4</sup>Beihang University
</p>

<sub><sup>*</sup>Equal Contribution&nbsp;&nbsp;<sup>†</sup>Project Lead</sub>

</div>

---

<p align="center">
  <img src="./assets/overview.png" alt="TACO overview" width="100%">
</p>

TACO is a tactile-aware world-model-driven framework for scalable VLA post-training in contact-rich robot manipulation. Given real-world rollouts, TACO follows a **Recognize-Imagine-Label** loop: it recognizes failure-adjacent contact states, imagines local visuo-tactile correction segments, and labels corrective actions for policy post-training.

The resulting corrective supervision is used with **knowledge-insulated tactile adaptation**, allowing the policy to learn contact recovery behaviors without degrading pretrained visual-language priors.

This repository is currently a lightweight public landing repo. Core code, model checkpoints, datasets, and detailed training recipes will be released progressively according to the roadmap below.

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Method Overview](#-method-overview)
- [Roadmap](#-roadmap)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Citation](#-citation)

## ✨ Highlights

- **Tactile-aware world model:** jointly denoises future video and 12-D force-torque trajectories for contact-consistent imagination.
- **Recognize-Imagine-Label loop:** converts real-world failures into local visuo-tactile corrections without repeated human intervention.
- **Unified progress-action model:** estimates task progress, localizes failure-adjacent states, and labels imagined segments with executable 7-DoF corrective actions.
- **Knowledge-insulated tactile adaptation:** blocks tactile-action gradients from updating the pretrained VLM backbone, preserving visual-language priors during tactile post-training.

## 🧠 Method Overview

TACO uses real robot rollouts as the starting point for iterative post-training:

1. **Recognize:** a unified progress-action model predicts dense task progress and selects failure-adjacent contact states where progress stalls or decreases.
2. **Imagine:** a visuo-tactile generation model imagines local correction segments by jointly generating future RGB observations and left/right 6-DoF force-torque signals.
3. **Label:** the progress-action model converts imagined visuo-tactile corrections into executable corrective actions and progress labels.
4. **Post-train:** the VLA action expert is updated with tactile correction data while the pretrained VLM backbone is protected through knowledge insulation.

See [Method Overview](./docs/method_overview.md) for a more detailed summary.

<p align="center">
  <img src="./assets/pipeline.png" alt="TACO iterative post-training pipeline" width="100%">
</p>

<p align="center">
  <img src="./assets/model_architecture.png" alt="TACO tactile-aware world model architecture" width="100%">
</p>

## 🗓 Roadmap

- [ ] Open-source visuo-tactile world model
- [ ] Open-source tactile-aware VLA
- [ ] Open-source the full TACO framework

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

Coming soon.

## 📚 Documentation

- [Method Overview](./docs/method_overview.md)
- [Release Notes](./docs/release_notes.md)
- [Examples](./examples)

## 📄 Citation

Coming soon.
