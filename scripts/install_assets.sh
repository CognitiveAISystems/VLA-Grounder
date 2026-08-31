#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository}"
git submodule update --init external/BlindVLA external/RL4VLA

asset_root="external/BlindVLA/ManiSkill/mani_skill/assets/carrot"
required=(
    more_arrows
    more_carrot
    more_color
    more_laundry
    more_plate
    more_public_info
    more_shape
    more_traffic
    more_weather
)

for directory in "${required[@]}"; do
    if [[ ! -d "${asset_root}/${directory}" ]]; then
        echo "Missing benchmark assets: ${asset_root}/${directory}" >&2
        exit 1
    fi
done

echo "Benchmark assets are available from the BlindVLA and RL4VLA submodules."
