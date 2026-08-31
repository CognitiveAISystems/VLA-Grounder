from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import load_config
from .grounder import Grounder
from .protocol import EnvironmentClient


def evaluate(config_path: str, seed: int | None = None, scene_index: int | None = None) -> None:
    config = load_config(config_path)
    if config.grounder.enabled:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.grounder.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = EnvironmentClient(config.network.host, config.network.port)
    grounder = None
    if config.grounder.enabled:
        grounder = Grounder(
            config.grounder.model,
            config.grounder.checkpoint,
            config.grounder.attention,
            config.grounder.prompt,
            config.grounder.thinking,
        )

    try:
        seeds = [seed] if seed is not None else config.seeds
        scene_indices = [scene_index] if scene_index is not None else range(len(config.environment.scenes))
        for benchmark_seed in seeds:
            torch.manual_seed(benchmark_seed)
            torch.cuda.manual_seed_all(benchmark_seed)
            np.random.seed(benchmark_seed)
            random.seed(benchmark_seed)
            print(f"[Client] RNG seeded with {benchmark_seed}", flush=True)
            episode_rng = np.random.RandomState(benchmark_seed)
            for current_scene_index in scene_indices:
                scene = config.environment.scenes[current_scene_index]
                episode_ids = torch.from_numpy(
                    episode_rng.randint(0, 1_000_000_000, size=config.environment.num_envs)
                )
                images, instructions = client.request("reset", (current_scene_index, episode_ids))
                commands = instructions
                if grounder is not None:
                    commands = grounder.generate(
                        images,
                        instructions,
                        config.grounder.batch_size,
                        config.grounder.max_new_tokens,
                        config.grounder.temperature,
                        config.grounder.top_p,
                    )
                    grounder.release()
                result = client.request(
                    "evaluate",
                    {
                        "scene_index": current_scene_index,
                        "episode_ids": episode_ids,
                        "instructions": instructions,
                        "commands": commands,
                    },
                )
                metric_names = {
                    "reward": "reward",
                    "success": "success",
                    "consecutive_grasp": "consecutive_grasp",
                    "source_grasped": "is_src_obj_grasped",
                }
                per_episode = {
                    index: {
                        "episode_id": int(episode_ids[index]),
                        **{
                            output_name: float(result[input_name][index])
                            for input_name, output_name in metric_names.items()
                        },
                    }
                    for index in range(config.environment.num_envs)
                }
                instructions_log = {
                    index: f"{instructions[index]}=|=|={result['commands'][index]}"
                    for index in range(config.environment.num_envs)
                }
                stats = {
                    output_name: float(np.mean(result[input_name]))
                    for input_name, output_name in metric_names.items()
                }
                use_grounder = int(config.grounder.enabled)
                record = {
                    "args": {
                        "ckpt_path": config.grounder.get("checkpoint"),
                        "model": config.grounder.get("model"),
                        "num_envs": config.environment.num_envs,
                        "obj_set": config.environment.split,
                        "output_dir": str(output_dir),
                        "port": config.network.port,
                        "scenes": scene,
                        "seed": benchmark_seed,
                        "use_vlm": use_grounder,
                        "vla": config.environment.backend,
                    },
                    "scenes": scene,
                    "episode_ids": episode_ids.tolist(),
                    "instruction": instructions_log,
                    "per_episode": per_episode,
                    "stats": stats,
                }
                path = output_dir / f"{scene}_{benchmark_seed}_{use_grounder}.yaml"
                with path.open("w", encoding="utf-8") as stream:
                    yaml.dump(record, stream, default_flow_style=False)
    finally:
        try:
            client.request("stop")
        except Exception:
            pass
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLA Grounder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scene-index", type=int)
    args = parser.parse_args()
    evaluate(args.config, args.seed, args.scene_index)


if __name__ == "__main__":
    main()
