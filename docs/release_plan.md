# Release Plan

This repository will be released in stages. The current version is a public landing repository with documentation placeholders.

## Stage 1: Visuo-Tactile World Model

- Model architecture for joint video-force denoising.
- Temporal RoPE alignment and first-frame force anchoring.
- Unified progress-action model for progress estimation and corrective action labeling.
- Data preprocessing scripts for RGB, force-torque, action, and progress annotations.
- Inference and visualization scripts for imagined correction segments.
- Pretrained checkpoints.

## Stage 2: Tactile-Aware VLA

- Tactile and advantage conditioning modules for the VLA action expert.
- Knowledge-insulated tactile adaptation recipe.
- Fine-tuning configs for contact-rich tasks.
- Policy serving and robot-side inference examples.
- Evaluation scripts and task configs.

## Stage 3: Full TACO Framework

- End-to-end Recognize-Imagine-Label pipeline.
- Iterative real-to-imagine-to-real post-training workflow.
- Unified data, model, and evaluation interfaces.
- Reproducibility scripts for main experiments and ablations.
- Documentation for adapting TACO to new robots, cameras, and tactile sensors.

