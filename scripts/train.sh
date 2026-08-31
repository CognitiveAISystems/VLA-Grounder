#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_DATA_DIR="${VLA_DATA_DIR:-${repository}/data}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONPATH="${repository}/src:${repository}/third_party:${repository}/third_party/SimplerEnv:${repository}/external/BlindVLA/ManiSkill:${repository}/third_party/lerobot:${repository}/third_party/openvla${PYTHONPATH:+:${PYTHONPATH}}"

config="${1:?Usage: $0 <config.yaml>}"
environment_python="${repository}/envs/vla-grounder/bin/python"
training_python="${TRAINING_PYTHON:-${repository}/envs/grpo/bin/python}"

environment_device="$("${environment_python}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["environment"]["device"])' "${config}")"
model_device="$("${environment_python}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["model"]["device"])' "${config}")"

CUDA_VISIBLE_DEVICES="${environment_device}" "${environment_python}" -m vla_grounder.environment_server --config "${config}" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES="${model_device}" "${training_python}" -m vla_grounder.training --config "${config}"
