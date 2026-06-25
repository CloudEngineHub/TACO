<div align="center">

# TACO: TActile World Model as a Self-COrrector for Scalable VLA Post-Training

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Coming%20Soon-blue)]()
[![Code](https://img.shields.io/badge/Code-Staged%20Release-lightgrey)](#-roadmap)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](./LICENSE)

<p>
  <a href="https://liushb9.github.io/">Shengbang Liu</a><sup>1,3,*</sup>&nbsp;&nbsp;
  <a href="https://jiayueru.github.io/">Yueru Jia</a><sup>1,2,*</sup>&nbsp;&nbsp;
  <a href="https://github.com/avx34/">Yuyang Yan</a><sup>1,*</sup>&nbsp;&nbsp;
  <a href="https://liujiaming1996.github.io/">Jiaming Liu</a><sup>1,*,†</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=16aBS6kAAAAJ&hl=zh-CN&oi=sra">Xinran Zhang</a><sup>1,2</sup>&nbsp;&nbsp;
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

TACO is a tactile-aware world-model-driven framework for scalable VLA post-training in contact-rich robot manipulation. Given real-world rollouts, TACO follows a **Recognize-Imagine-Label** loop: it recognizes failure-adjacent contact states, imagines local visuo-tactile correction segments, and labels corrective actions for policy post-training.

The resulting corrective supervision is used with **knowledge-insulated tactile adaptation**, allowing the policy to learn contact recovery behaviors without degrading pretrained visual-language priors.

This repository is currently a lightweight public landing repo. Core code, model checkpoints, datasets, and detailed training recipes will be released progressively according to the roadmap below.

## 📋 Table of Contents

- [Roadmap](#-roadmap)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Citation](#-citation)

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

- [Release Notes](./docs/release_notes.md)
- [Examples](./examples)

## 📄 Citation

Coming soon.
