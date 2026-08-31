#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker run --detach \
    --name vla-grounder \
    --gpus all \
    --network host \
    --shm-size 16g \
    --volume "${repository}:/workspace" \
    --volume "${HF_HOME:-${repository}/data/huggingface}:/root/.cache/huggingface:ro" \
    --env PYTHONPATH=/workspace/src:/workspace/third_party:/workspace/third_party/SimplerEnv:/workspace/external/BlindVLA/ManiSkill:/workspace/third_party/lerobot:/workspace/third_party/openvla \
    vla-grounder:cuda12.8
