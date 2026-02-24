from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

from .base import RLConfig

_CONFIGS: dict[str, Callable[[], RLConfig]] = {}
_CONFIGS_DIR = Path(__file__).parent
_EXCLUDE = {"__init__.py", "base.py"}


def get_config(name: str) -> RLConfig:
    if name not in _CONFIGS:
        available = ", ".join(sorted(_CONFIGS.keys()))
        raise ValueError(f"Unknown config: {name}. Available configs: {available}")
    return _CONFIGS[name]()


def list_configs() -> list[str]:
    return sorted(_CONFIGS.keys())


for file_path in _CONFIGS_DIR.glob("*.py"):
    if file_path.name in _EXCLUDE:
        continue
    module_name = file_path.stem
    config_name = module_name.replace("_", "-")
    module = importlib.import_module(f".{module_name}", package="configs")
    if hasattr(module, "get_config"):
        _CONFIGS[config_name] = module.get_config
