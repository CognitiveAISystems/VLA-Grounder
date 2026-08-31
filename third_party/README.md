# Third-party source

This directory contains the compatibility code used by the frozen VLA policies.

- `SimplerEnv` contains the project-specific simulator wrapper and π0/OpenVLA policy interfaces used by training and evaluation.
- `agent`, `experiments`, and `utils` are adapted from the [INT-ACT](https://github.com/ai4ce/INT-ACT) pipeline and contain the configuration, policy wrapper, and utilities required by the π0 integration. Local path handling for dataset statistics is marked in `experiments/env_adapters/simpler.py`.
- `lerobot` contains the Apache-2.0-licensed LeRobot policy implementation used by π0.
- `openvla` contains the OpenVLA model implementation used by the evaluation pipeline.

The benchmark repositories and their assets are kept as pinned submodules under `external/`. See the root README for installation instructions and upstream acknowledgements. Third-party source remains subject to its upstream terms. The INT-ACT-derived directories are not covered by the repository's Apache-2.0 license; see `NOTICE` in this directory.
