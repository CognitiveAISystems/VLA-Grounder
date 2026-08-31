#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repository}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${repository}/envs/grpo/bin/python" -m vla_grounder.merge_lora "$@"
