#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-all}"

if [[ "${mode}" != "all" && "${mode}" != "--eval-only" ]]; then
    echo "Usage: $0 [--eval-only]" >&2
    exit 2
fi

create_environment() {
    local path="$1"
    if python -m venv "${path}"; then
        return
    fi
    if ! command -v virtualenv >/dev/null 2>&1; then
        echo "Python venv support or virtualenv is required" >&2
        exit 1
    fi
    virtualenv --python="$(command -v python)" "${path}"
}

cd "${repository}"
git submodule update --init external/BlindVLA external/RL4VLA

create_environment envs/vla-grounder
envs/vla-grounder/bin/python -m pip install --upgrade pip
envs/vla-grounder/bin/python -m pip install \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
envs/vla-grounder/bin/python -m pip install --requirement requirements.lock
envs/vla-grounder/bin/python -m pip install --no-deps --editable external/BlindVLA/ManiSkill
envs/vla-grounder/bin/python -m pip install --no-deps --editable third_party/SimplerEnv
envs/vla-grounder/bin/python -m pip install --no-deps --editable third_party/openvla
envs/vla-grounder/bin/python -m pip install --no-deps --editable .

if [[ "${mode}" == "--eval-only" ]]; then
    exit 0
fi

create_environment envs/grpo
envs/grpo/bin/python -m pip install --upgrade pip
envs/grpo/bin/python -m pip install \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
envs/grpo/bin/python -m pip install --requirement requirements-training.lock
envs/grpo/bin/python -m pip install --no-deps --editable .
