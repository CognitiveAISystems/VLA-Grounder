#!/usr/bin/env bash
set -euo pipefail

docker build --file docker/Dockerfile --tag vla-grounder:cuda12.8 .
