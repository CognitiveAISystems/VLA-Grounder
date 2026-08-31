from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import zmq

from .config import load_config
from .protocol import extract_evaluation_command, extract_training_command
from .vla import FrozenVLAEnvironment


def _tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def serve(config_path: str) -> None:
    config = load_config(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.environment.device)
    selected_scene = os.environ.get("VLA_SCENE_INDEX")
    configured_scene_index = int(selected_scene) if selected_scene is not None else None
    scenes = config.environment.scenes
    if configured_scene_index is not None:
        scenes = [scenes[configured_scene_index]]

    def environment_scene_index(scene_index: int) -> int:
        if configured_scene_index is None:
            return scene_index
        if scene_index != configured_scene_index:
            raise ValueError(f"Server is configured for scene index {configured_scene_index}")
        return 0

    seed = int(os.environ.get("VLA_POLICY_SEED", "0"))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    environment = FrozenVLAEnvironment(
        backend=config.environment.backend,
        mode=config.environment.mode,
        scenes=scenes,
        num_envs=config.environment.num_envs,
        model_path=config.environment.model,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_log = output_dir / "rollouts.jsonl"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{config.network.host}:{config.network.port}")

    try:
        while True:
            request = socket.recv_pyobj()
            try:
                command = request["command"]
                data = request.get("data")
                if command == "ping":
                    response = "pong"
                elif command == "reset":
                    if len(data) == 3:
                        scene_index, episode_ids, split = data
                    else:
                        scene_index, episode_ids = data
                        split = config.environment.split
                    observation, instruction = environment.reset(
                        environment_scene_index(scene_index),
                        split,
                        episode_ids,
                    )
                    response = (observation.cpu().numpy(), instruction)
                elif command == "rollout":
                    generated = list(data["commands"])
                    if config.model.thinking:
                        generated = [extract_training_command(text) for text in generated]
                    repeats = config.environment.rollouts_per_command
                    baselines = config.environment.baseline_rollouts
                    scene_indices = data["scene_indices"]
                    if len(set(scene_indices)) != 1:
                        raise AssertionError("All scene indices in a GRPO group must be identical")
                    splits = data["splits"]
                    if len(set(splits)) != 1:
                        raise AssertionError("All splits in a GRPO group must be identical")
                    commands = np.repeat(generated, repeats).tolist()
                    commands.extend([data["instructions"][0]] * baselines)
                    group_episode_ids = data["episode_ids"]
                    if not torch.all(group_episode_ids == group_episode_ids[0]):
                        raise AssertionError("All episode IDs in a GRPO group must be identical")
                    episode_ids = group_episode_ids.repeat_interleave(repeats)
                    episode_ids = torch.cat(
                        [episode_ids, episode_ids[:1].repeat(baselines)]
                    )
                    if len(episode_ids) != config.environment.num_envs:
                        raise AssertionError(
                            "num_envs must equal group_size * rollouts_per_command + baseline_rollouts"
                        )
                    environment.reset(
                        environment_scene_index(scene_indices[0]),
                        splits[0],
                        episode_ids,
                    )
                    result, _ = environment.rollout(commands)
                    baseline = result.reward[-baselines:].mean()
                    rewards = result.reward[:-baselines].reshape(-1, repeats).mean(dim=1)
                    rewards = rewards - baseline
                    record = {
                        "scene": config.environment.scenes[scene_indices[0]],
                        "episode_id": int(group_episode_ids[0]),
                        "split": splits[0],
                        "instructions": data["instructions"],
                        "commands": generated,
                        "rewards": _tensor_list(rewards),
                        "baseline": float(baseline),
                    }
                    with reward_log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                    response = rewards.cpu().numpy()
                elif command == "evaluate":
                    commands = [extract_evaluation_command(text) for text in data["commands"]]
                    environment.reset(
                        environment_scene_index(data["scene_index"]),
                        config.environment.split,
                        data["episode_ids"],
                    )
                    result, _ = environment.rollout(commands)
                    response = {
                        "reward": _tensor_list(result.reward),
                        "success": _tensor_list(result.success),
                        "consecutive_grasp": _tensor_list(result.consecutive_grasp),
                        "source_grasped": _tensor_list(result.source_grasped),
                        "commands": commands,
                    }
                elif command == "stop":
                    socket.send_pyobj({"status": "ok", "data": None})
                    break
                else:
                    raise ValueError(f"Unknown command: {command}")
                socket.send_pyobj({"status": "ok", "data": response})
            except Exception as error:
                socket.send_pyobj({"status": "error", "error": str(error)})
    finally:
        environment.close()
        socket.close()
        context.term()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve frozen VLA rollouts")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    serve(args.config)


if __name__ == "__main__":
    main()
