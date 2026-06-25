<div align="center">

# TACO: Tactile-Aware Control with Visuo-Tactile World Models

[![Project](https://img.shields.io/badge/Project-TACO-blue)](#)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-lightgrey)](#-roadmap)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](./LICENSE)

<p>
  A staged open-source framework for visuo-tactile world modeling and tactile-aware VLA policies.
</p>

</div>

---

TACO is a research codebase for learning robot manipulation policies that reason over both visual and tactile feedback. The project will be released in stages, starting from the visuo-tactile world model, then tactile-aware VLA training and inference, and finally the complete TACO framework.

This repository is currently a lightweight public landing repo. Core code, model checkpoints, datasets, and detailed training recipes will be added according to the roadmap below.

## 📋 Table of Contents

- [Roadmap](#-roadmap)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Acknowledgments](#-acknowledgments)
- [Citation](#-citation)

## 🗓 Roadmap

### 1. Open-source visuo-tactile world model

- [ ] Release model architecture and training code
- [ ] Release data preprocessing scripts for paired visual-tactile trajectories
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

- [Release Notes](./docs/release_notes.md)
- [Examples](./examples)

## 🙏 Acknowledgments

TACO builds on ideas and infrastructure from open-source robotics, VLA, and embodied world-modeling projects. We thank the broader research community for making these foundations available.

## 📄 Citation

If you find TACO useful for your research, please consider citing our work. Citation information will be added after the paper/preprint release.

```bibtex
@misc{taco2026,
  title = {TACO: Tactile-Aware Control with Visuo-Tactile World Models},
  author = {TACO Contributors},
  year = {2026},
  url = {https://github.com/liushb9/TACO}
}
```
