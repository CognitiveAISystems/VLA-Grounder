from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as error:
            raise AttributeError(key) from error
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return Config(data)
