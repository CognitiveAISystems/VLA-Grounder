from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def merge_lora(
    adapter: str,
    output: str,
    base_model: str | None = None,
    attention: str = "sdpa",
    max_shard_size: str = "5GB",
) -> None:
    adapter_config = PeftConfig.from_pretrained(adapter)
    model_name = base_model or adapter_config.base_model_name_or_path
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention,
        device_map="auto",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    try:
        processor = AutoProcessor.from_pretrained(adapter, trust_remote_code=True)
    except (OSError, ValueError):
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.save_pretrained(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a VLA Grounder LoRA adapter")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()
    merge_lora(
        adapter=args.adapter,
        output=args.output,
        base_model=args.base_model,
        attention=args.attention,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
