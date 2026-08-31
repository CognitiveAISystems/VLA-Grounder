from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from .checkpoints import SaveEveryCheckpointCallback
from .config import load_config
from .prompts import build_messages
from .protocol import EnvironmentClient


class RolloutDataset(Dataset):
    def __init__(self, client: EnvironmentClient, config):
        self.samples = []
        for cycle in range(config.environment.steps_per_epoch):
            split = "train" if cycle % 2 == 0 else "test"
            for scene_index, _ in enumerate(config.environment.scenes):
                episode_ids = torch.randint(1_000_000_000, (config.environment.num_envs,))
                images, instructions = client.request(
                    "reset", (scene_index, episode_ids, split)
                )
                self.samples.extend(
                    {
                        "prompt": build_messages(
                            instruction,
                            Image.fromarray(image.astype(np.uint8)),
                            config.model.prompt,
                            config.model.thinking,
                        ),
                        "episode_id": episode_id,
                        "scene_index": scene_index,
                        "split": split,
                        "instruction": instruction,
                    }
                    for image, instruction, episode_id in zip(images, instructions, episode_ids)
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def reward_function(client: EnvironmentClient, scene_name: str, scene_index: int):
    def reward(completions, prompts, episode_id, instruction, split, **kwargs):
        indices = kwargs["scene_index"]
        if indices[0] != scene_index:
            return [None] * len(completions)
        texts = [item[0]["content"] if isinstance(item, list) else item for item in completions]
        rewards = client.request(
            "rollout",
            {
                "scene_indices": indices,
                "episode_ids": torch.stack(episode_id),
                "instructions": instruction,
                "splits": split,
                "commands": texts,
            },
        )
        return rewards.tolist()

    reward.__name__ = f"{scene_name}_id_{scene_index}"
    return reward


def format_reward(completions, **kwargs):
    pattern = r".*?</think>\n\n<answer>.*?</answer>$"
    contents = [
        completion[0]["content"]
        if isinstance(completion, list) and isinstance(completion[0], dict)
        else completion
        for completion in completions
    ]
    return [1.0 if re.fullmatch(pattern, content, re.DOTALL) else 0.0 for content in contents]


def linear_module_names(model, excluded: list[str]) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
        and not any(part in name for part in excluded)
    ]


def train(config_path: str) -> None:
    config = load_config(config_path)
    expected_envs = (
        config.training.group_size * config.environment.rollouts_per_command
        + config.environment.baseline_rollouts
    )
    if config.environment.num_envs != expected_envs:
        raise ValueError(
            "environment.num_envs must equal training.group_size * "
            "environment.rollouts_per_command + environment.baseline_rollouts"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.model.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    client = EnvironmentClient(config.network.host, config.network.port)
    if client.request("ping") != "pong":
        raise RuntimeError("Environment server did not respond")

    model = AutoModelForImageTextToText.from_pretrained(
        config.model.name,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.model.attention,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = linear_module_names(model, list(config.model.lora_exclude))
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=targets,
            bias="none",
        ),
    )
    processor = AutoProcessor.from_pretrained(
        config.model.name,
        trust_remote_code=True,
        truncation_side="left",
        padding_side="left",
    )
    processor.image_processor.do_resize = False
    dataset = RolloutDataset(client, config)
    args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=float(config.training.learning_rate),
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.training.gradient_accumulation,
        num_generations=config.training.group_size,
        max_completion_length=config.training.max_completion_length,
        max_steps=config.training.max_steps,
        logging_steps=1,
        save_strategy="no",
        max_grad_norm=1.0,
        bf16=config.training.bf16,
        beta=config.training.kl_coefficient,
        temperature=config.training.temperature,
        top_p=config.training.top_p,
        report_to="tensorboard",
    )
    callback = SaveEveryCheckpointCallback(config.callback.save_every)
    rewards = [
        reward_function(client, name, index)
        for index, name in enumerate(config.environment.scenes)
    ]
    if config.model.thinking:
        rewards.append(format_reward)
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        reward_funcs=rewards,
        processing_class=processor,
        callbacks=[callback],
    )
    callback.set_trainer(trainer)
    try:
        trainer.train()
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VLA Grounder with GRPO")
    parser.add_argument("--config", required=True)
    train(parser.parse_args().config)


if __name__ == "__main__":
    main()
