from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class RolloutResult:
    reward: torch.Tensor
    success: torch.Tensor
    consecutive_grasp: torch.Tensor
    source_grasped: torch.Tensor


class FrozenVLAEnvironment:
    def __init__(
        self,
        backend: str,
        mode: str,
        scenes: Sequence[str],
        num_envs: int,
        model_path: str,
    ):
        from simpler_env.env.simpler_wrapper import SimlerWrapper, SimlerWrapperOpenVLA
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy
        from simpler_env.policies.pi0.pi0_model import PI0Policy
        from simpler_env.train_ms3_ppo import Args

        wrapper_type, policy_type = {
            "pi0": (SimlerWrapper, PI0Policy),
            "openvla": (SimlerWrapperOpenVLA, OpenVLAPolicy),
        }[backend]
        self.backend = backend
        if mode not in {"single", "efficient", "parallel"}:
            raise ValueError(f"Unknown environment mode: {mode}")
        if mode == "single" and len(scenes) != 1:
            raise ValueError("single mode requires exactly one scene")
        self.mode = mode
        self.scenes = list(scenes)
        self.num_envs = num_envs
        self.model_path = model_path
        self.args_type = Args
        self.wrapper_type = wrapper_type
        self.policy_type = policy_type
        self._policy = None
        self._normalization = None
        self._environment = None
        self._environments = []
        self._scene_index = None
        self.last_observation = None
        self.last_info = None
        self._initialize()

    def _initialize(self) -> None:
        self._args = self.args_type(
            num_envs=self.num_envs,
            buffer_inferbatch=self.num_envs,
            buffer_minibatch=self.num_envs,
            env_id=None,
            vla_path=self.model_path,
        )
        self._policy = self.policy_type(self._args, 0)
        self._policy.prep_rollout()
        if self.backend == "openvla":
            self._normalization = self._policy.vla.get_action_stats(self._args.vla_unnorm_key)
        if self.mode == "single":
            self._environment = self._make_environment(0)
            self._scene_index = 0
        elif self.mode == "parallel":
            self._environments = [self._make_environment(index) for index in range(len(self.scenes))]

    def _make_environment(self, scene_index: int):
        self._args.env_id = self.scenes[scene_index]
        return self.wrapper_type(self._args, self._normalization)

    def _select_environment(self, scene_index: int) -> None:
        if not 0 <= scene_index < len(self.scenes):
            raise IndexError(f"Scene index {scene_index} is out of range")
        if self.mode == "efficient":
            self._args.seed += self.num_envs
            if self._environment is None or len(self.scenes) > 1:
                self._close_environment()
                self._environment = self._make_environment(scene_index)
        elif self.mode == "parallel":
            self._environment = self._environments[scene_index]
        self._scene_index = scene_index

    def reset(self, scene_index: int, split: str, episode_ids: torch.Tensor):
        self._select_environment(scene_index)
        observation, instruction, info = self._environment.reset(
            obj_set=split, episode_id=episode_ids
        )
        if self.backend == "pi0":
            self._policy.wrapper.reset()
        self.last_observation = observation
        self.last_info = info
        return observation, instruction

    @torch.inference_mode()
    def rollout(self, commands: Sequence[str], capture_frames: bool = False):
        observation = self.last_observation
        info = self.last_info
        frames = []
        if self.backend == "pi0":
            self._policy.wrapper.reset()
            for _ in range(20):
                batch = {
                    "task_description": list(commands),
                    "image": observation,
                    "pi_0": info["pi_0"],
                }
                _, actions, _ = self._policy.get_action(batch)
                for action_index in range(4):
                    observation, _, _, info = self._environment.step(actions[:, action_index, :])
                    if capture_frames:
                        frames.append(observation.cpu())
        else:
            for _ in range(80):
                batch = {"task_description": list(commands), "image": observation}
                _, actions, _ = self._policy.get_action(batch, deterministic=True)
                observation, _, _, info = self._environment.step(actions)
                if capture_frames:
                    frames.append(observation.cpu())
        episode = info["episode"]
        tensors = {
            key: torch.as_tensor(value, dtype=torch.float32, device=observation.device)
            for key, value in episode.items()
        }
        success = tensors["success"]
        consecutive = tensors["consecutive_grasp"]
        grasped = tensors["is_src_obj_grasped"]
        reward = torch.clamp(success + 0.1 * grasped + 0.2 * consecutive, max=1.0)
        return RolloutResult(reward, success, consecutive, grasped), frames

    def close(self) -> None:
        if self.mode == "parallel":
            for environment in self._environments:
                self._close_wrapper(environment)
            self._environments = []
            self._environment = None
        else:
            self._close_environment()
        self._policy = None
        self._normalization = None

    @staticmethod
    def _close_wrapper(environment) -> None:
        inner = getattr(environment, "env", None)
        if inner is not None:
            inner.close()

    def _close_environment(self) -> None:
        if self._environment is not None:
            self._close_wrapper(self._environment)
        self._environment = None
        self._scene_index = None
