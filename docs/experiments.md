# Experiments

This document summarizes the experimental setup reported in the paper. Code, configs, and evaluation scripts will be released in later stages.

## Robot Setup

- **Robot:** Franka Research 3 single-arm manipulator
- **Camera:** Intel RealSense D455 front-view RGB camera
- **Tactile sensors:** two fingertip-mounted Xense tactile sensors
- **Observation:** RGB image, left/right 6-DoF force-torque readings, and proprioceptive state
- **Action:** 7-DoF end-effector action

## Tasks

TACO is evaluated on six real-world contact-rich manipulation tasks:

| Task | Success criterion |
| --- | --- |
| Insert Flower | Insert the flower into the vase without dropping it or knocking over the vase |
| Wipe Whiteboard | Grasp the eraser and completely erase the target mark |
| Twist Bottle Cap | Grasp, twist open, and lift the cap away from the bottle |
| Play Xylophone | Strike the 1st, 3rd, 5th, and 8th keys in order |
| Toast Bread | Pick up two bread slices and insert both into the toaster |
| Move Hanoi Rings | Move the top two rings to the target pegs without dropping or disturbing them |

Each task uses 50 teleoperated demonstration trajectories for the base policy warm start. Each method is evaluated over 40 independent episodes per task with randomized tabletop object positions.

## Main Result

Average success rate and completion steps across six tasks:

| Method | Iteration | Average success rate | Average completion steps |
| --- | --- | --- | --- |
| Base Policy | 0 | 0.38 | 185.5 |
| Filtered BC | 1 | 0.41 | 148.8 |
| TACO without KI | 1 | 0.49 | 154.8 |
| TACO | 1 | 0.66 | 141.8 |
| Filtered BC | 2 | 0.43 | 155.5 |
| TACO without KI | 2 | 0.50 | 146.5 |
| TACO | 2 | 0.82 | 127.7 |

After two post-training iterations, TACO improves average success rate by 44% absolute over the base policy and 32% over the variant without knowledge-insulated tactile adaptation.

## Generalization

The paper also evaluates fast adaptation under three out-of-domain shifts:

- unseen backgrounds;
- unseen objects;
- unseen object positions.

TACO improves robustness after one adaptation iteration by generating imagined corrections around OOD states, without requiring additional expert demonstrations in the target scenarios.

