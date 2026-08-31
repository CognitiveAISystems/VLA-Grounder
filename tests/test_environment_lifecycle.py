from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch

from vla_grounder.vla import FrozenVLAEnvironment


class InnerEnvironment:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class Wrapper:
    instances = []

    def __init__(self, args, normalization):
        self.scene = args.env_id
        self.seed = args.seed
        self.env = InnerEnvironment()
        self.resets = []
        self.instances.append(self)

    def reset(self, obj_set, episode_id):
        self.resets.append((obj_set, episode_id.clone()))
        count = len(episode_id)
        observation = torch.zeros(count, 2, 2, 3, dtype=torch.uint8)
        instruction = [self.scene] * count
        info = {"pi_0": {}, "episode": {}}
        return observation, instruction, info


class PolicyWrapper:
    def reset(self):
        return None


class Policy:
    def __init__(self, args, device_id):
        self.wrapper = PolicyWrapper()

    def prep_rollout(self):
        return None


@dataclass
class Args:
    num_envs: int
    buffer_inferbatch: int
    buffer_minibatch: int
    env_id: str | None
    vla_path: str
    seed: int = 0


def runtime_modules():
    modules = {}
    wrappers = types.ModuleType("simpler_env.env.simpler_wrapper")
    wrappers.SimlerWrapper = Wrapper
    wrappers.SimlerWrapperOpenVLA = Wrapper
    modules[wrappers.__name__] = wrappers
    pi0 = types.ModuleType("simpler_env.policies.pi0.pi0_model")
    pi0.PI0Policy = Policy
    modules[pi0.__name__] = pi0
    openvla = types.ModuleType("simpler_env.policies.openvla.openvla_train")
    openvla.OpenVLAPolicy = Policy
    modules[openvla.__name__] = openvla
    arguments = types.ModuleType("simpler_env.train_ms3_ppo")
    arguments.Args = Args
    modules[arguments.__name__] = arguments
    return modules


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        Wrapper.instances = []
        self.modules = patch.dict(sys.modules, runtime_modules())
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_single_retains_one_environment(self):
        environment = FrozenVLAEnvironment("pi0", "single", ["scene"], 2, "model")
        environment.reset(0, "test", torch.tensor([4, 5]))
        environment.reset(0, "test", torch.tensor([6, 7]))
        self.assertEqual(len(Wrapper.instances), 1)
        self.assertEqual(Wrapper.instances[0].seed, 0)
        environment.close()
        self.assertTrue(Wrapper.instances[0].env.closed)

    def test_efficient_recreates_selected_environment(self):
        environment = FrozenVLAEnvironment("pi0", "efficient", ["a", "b"], 2, "model")
        environment.reset(0, "test", torch.tensor([4, 5]))
        first = Wrapper.instances[0]
        environment.reset(1, "test", torch.tensor([4, 5]))
        self.assertTrue(first.env.closed)
        self.assertEqual([item.scene for item in Wrapper.instances], ["a", "b"])
        self.assertEqual([item.seed for item in Wrapper.instances], [2, 4])
        environment.close()

    def test_efficient_reuses_single_environment(self):
        environment = FrozenVLAEnvironment("pi0", "efficient", ["scene"], 2, "model")
        environment.reset(0, "test", torch.tensor([4, 5]))
        environment.reset(0, "test", torch.tensor([4, 5]))
        self.assertEqual(len(Wrapper.instances), 1)
        self.assertEqual(len(Wrapper.instances[0].resets), 2)
        self.assertFalse(Wrapper.instances[0].env.closed)
        environment.close()
        self.assertTrue(Wrapper.instances[0].env.closed)

    def test_parallel_retains_every_environment(self):
        environment = FrozenVLAEnvironment("pi0", "parallel", ["a", "b"], 2, "model")
        environment.reset(0, "test", torch.tensor([4, 5]))
        environment.reset(1, "test", torch.tensor([4, 5]))
        self.assertEqual(len(Wrapper.instances), 2)
        self.assertFalse(any(item.env.closed for item in Wrapper.instances))
        environment.close()
        self.assertTrue(all(item.env.closed for item in Wrapper.instances))


if __name__ == "__main__":
    unittest.main()
