from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from .config import load_config


def run(
    config_path: str,
    seed: int | None = None,
    scene_index: int | None = None,
) -> None:
    config = load_config(config_path)
    repository = Path(__file__).resolve().parents[2]
    vla_python = os.environ.get(
        "VLA_PYTHON", str(repository / "envs" / "vla-grounder" / "bin" / "python")
    )
    grounder_python = os.environ.get(
        "GROUNDER_PYTHON", str(repository / "envs" / "grpo" / "bin" / "python")
    )
    evaluation_python = grounder_python if config.grounder.enabled else vla_python
    seeds = [seed] if seed is not None else config.seeds
    scene_indices = (
        [scene_index]
        if scene_index is not None
        else range(len(config.environment.scenes))
    )
    for benchmark_seed in seeds:
        for current_scene_index in scene_indices:
            policy_seed = benchmark_seed
            common = os.environ.copy()
            common.update(
                {
                    "VLA_POLICY_SEED": str(policy_seed),
                }
            )
            server_environment = common | {
                "CUDA_VISIBLE_DEVICES": str(config.environment.device),
                "VLA_SCENE_INDEX": str(current_scene_index),
            }
            evaluation_environment = common | {
                "CUDA_VISIBLE_DEVICES": str(
                    config.grounder.device
                    if config.grounder.enabled
                    else config.environment.device
                )
            }
            server = subprocess.Popen(
                [
                    vla_python,
                    "-m",
                    "vla_grounder.environment_server",
                    "--config",
                    config_path,
                ],
                env=server_environment,
            )
            try:
                subprocess.run(
                    [
                        evaluation_python,
                        "-m",
                        "vla_grounder.evaluation",
                        "--config",
                        config_path,
                        "--seed",
                        str(benchmark_seed),
                        "--scene-index",
                        str(current_scene_index),
                    ],
                    env=evaluation_environment,
                    check=True,
                )
            finally:
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.terminate()
                    server.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation suite")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scene-index", type=int)
    args = parser.parse_args()
    run(args.config, args.seed, args.scene_index)


if __name__ == "__main__":
    main()
