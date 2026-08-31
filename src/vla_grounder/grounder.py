from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .prompts import build_messages


class Grounder:
    def __init__(
        self,
        model_name: str,
        checkpoint: str,
        attention: str = "flash_attention_2",
        prompt: str = "guided",
        thinking: bool = False,
    ):
        self._model_name = model_name
        self._checkpoint = checkpoint
        self._attention = attention
        checkpoint_path = Path(checkpoint)
        if (checkpoint_path / "adapter_config.json").is_file():
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=attention,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(model, checkpoint).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation=attention,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
        self.processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            truncation_side="left",
            padding_side="left",
        )
        self.prompt = prompt
        self.thinking = thinking

    def release(self) -> None:
        """Release the grounder before loading the frozen VLA policy."""
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def _reload(self) -> None:
        checkpoint_path = Path(self._checkpoint)
        if (checkpoint_path / "adapter_config.json").is_file():
            model = AutoModelForImageTextToText.from_pretrained(
                self._model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=self._attention,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(model, self._checkpoint).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self._checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation=self._attention,
                device_map="auto",
                trust_remote_code=True,
            ).eval()

    @torch.inference_mode()
    def generate(
        self,
        images: np.ndarray,
        instructions: list[str],
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 20,
    ) -> list[str]:
        if self.model is None:
            self._reload()
        commands = []
        pil_images = [Image.fromarray(image.astype(np.uint8)) for image in images]
        for start in range(0, len(images), batch_size):
            batch_images = pil_images[start : start + batch_size]
            batch_instructions = instructions[start : start + batch_size]
            messages = [
                self.processor.apply_chat_template(
                    build_messages(instruction, image, self.prompt, self.thinking),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for image, instruction in zip(batch_images, batch_instructions)
            ]
            inputs = self.processor(
                text=messages,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                top_k=top_k,
            )
            trimmed = [output[len(tokens) :] for output, tokens in zip(outputs, inputs.input_ids)]
            commands.extend(self.processor.batch_decode(trimmed, skip_special_tokens=True))
        return commands
