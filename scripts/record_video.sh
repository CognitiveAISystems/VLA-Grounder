#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_DATA_DIR="${VLA_DATA_DIR:-${repository}/data}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONPATH="${repository}/src:${repository}/third_party:${repository}/third_party/SimplerEnv:${repository}/external/BlindVLA/ManiSkill:${repository}/third_party/lerobot:${repository}/third_party/openvla${PYTHONPATH:+:${PYTHONPATH}}"

source "${repository}/envs/vla-grounder/bin/activate"

config=""
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    if [[ "${arguments[index]}" == "--config" ]]; then
        config="${arguments[index + 1]}"
        break
    fi
done
if [[ -z "${config}" ]]; then
    echo "--config is required" >&2
    exit 2
fi
environment_device="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["environment"]["device"])' "${config}")"
CUDA_VISIBLE_DEVICES="${environment_device}" python -m vla_grounder.video "$@"
