from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from .config import load_config
from .vla import FrozenVLAEnvironment


def grid_frame(batch: torch.Tensor) -> np.ndarray:
    images = batch.float()
    if images.max() > 1:
        images = images / 255
    if images.shape[-1] in (3, 4):
        images = images.permute(0, 3, 1, 2)
    images = F.interpolate(images[:, :3], scale_factor=0.5, mode="bilinear", align_corners=False)
    grid = make_grid(images, nrow=int(math.sqrt(len(images))), padding=2)
    return (grid.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def record(config_path: str, scene: str, prompts_path: str, output: str, seed: int, fps: int) -> None:
    config = load_config(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.environment.device)
    prompts = [line.strip() for line in Path(prompts_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("The prompt file is empty")
    if len(prompts) > config.environment.num_envs:
        raise ValueError("Prompt count exceeds environment count")
    prompts.extend([prompts[-1]] * (config.environment.num_envs - len(prompts)))
    scene_index = list(config.environment.scenes).index(scene)
    generator = torch.Generator().manual_seed(seed)
    episode_ids = torch.randint(1_000_000_000, (config.environment.num_envs,), generator=generator)
    environment = FrozenVLAEnvironment(
        config.environment.backend,
        config.environment.mode,
        config.environment.scenes,
        config.environment.num_envs,
        config.environment.model,
    )
    try:
        environment.reset(scene_index, config.environment.split, episode_ids)
        _, frames = environment.rollout(prompts, capture_frames=True)
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, [grid_frame(frame) for frame in frames], fps=fps)
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record raw grid video for selected commands")
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    record(args.config, args.scene, args.prompts, args.output, args.seed, args.fps)


if __name__ == "__main__":
    main()
