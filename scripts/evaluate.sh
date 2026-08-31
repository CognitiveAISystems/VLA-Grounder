#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_DATA_DIR="${VLA_DATA_DIR:-${repository}/data}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONPATH="${repository}/src:${repository}/third_party:${repository}/third_party/SimplerEnv:${repository}/external/BlindVLA/ManiSkill:${repository}/third_party/lerobot:${repository}/third_party/openvla${PYTHONPATH:+:${PYTHONPATH}}"

config="${1:?Usage: $0 <config.yaml> [--seed SEED] [--scene-index INDEX]}"
export VLA_PYTHON="${VLA_PYTHON:-${repository}/envs/vla-grounder/bin/python}"
export GROUNDER_PYTHON="${GROUNDER_PYTHON:-${repository}/envs/grpo/bin/python}"

shift
"${VLA_PYTHON}" -m vla_grounder.evaluation_suite --config "${config}" "$@"
