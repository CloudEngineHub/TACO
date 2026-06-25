# Method Overview

TACO is a tactile-aware world-model-driven framework for scalable VLA post-training in contact-rich manipulation. It targets localized contact failures where the policy understands the task semantics but fails to recover from slippage, insufficient pressure, misalignment, or abnormal torque.

## Tactile-Aware World Model

The tactile-aware world model contains two components:

- **Visuo-tactile generation model:** imagines local correction segments from failure-adjacent states by jointly denoising future RGB video tokens and 12-D left/right 6-DoF force-torque sequences.
- **Unified progress-action model:** predicts task progress and 7-DoF corrective actions from visual and tactile observations.

The generation model aligns video and force tokens with temporal RoPE and keeps the first-frame force signal as a clean contact-state anchor. This lets imagined rollouts model contact dynamics instead of treating tactile feedback as an auxiliary condition only.

## Recognize-Imagine-Label Loop

TACO turns real-world failures into corrective supervision through an iterative loop:

1. **Recognize failure-adjacent states:** deploy the current policy, estimate dense progress on each rollout, and select anchors where progress stalls or decreases.
2. **Imagine visuo-tactile corrections:** generate local future RGB and force-torque segments from the selected anchors.
3. **Label corrective actions:** use the progress-action model to label imagined segments with executable action sequences and progress values.
4. **Post-train the VLA:** train on demonstrations, real rollouts, and imagined corrections, then deploy the improved policy for another iteration.

## Knowledge-Insulated Tactile Adaptation

Naively fine-tuning a full VLA on tactile-heavy correction data can erode pretrained visual-language priors. TACO avoids this by applying stop-gradient to the pretrained VLM prefix representation and routing tactile learning into the action expert.

During post-training:

- image, language, and state tokens are encoded by the VLM backbone;
- force history and advantage are injected into the action expert through adaRMSNorm conditioning;
- only the action expert, tactile encoder, advantage encoder, and adaptation layers are optimized.

## Advantage-Conditioned Post-Training

TACO assigns binary advantage labels to distinguish recovery-oriented corrections from failed rollout segments:

- `advantage = 1`: expert demonstrations, successful pre-failure segments, and imagined corrective segments;
- `advantage = 0`: failed segments after recognized failure onset.

At inference time, the policy is conditioned on positive advantage to encourage high-progress tactile recovery behavior.

