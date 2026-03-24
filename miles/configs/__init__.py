"""Config registry for Miles + Harbor training."""

import importlib
from pathlib import Path
from typing import Callable

from .base import RLConfig

_CONFIGS: dict[str, Callable[[bool], RLConfig]] = {}


def get_config(name: str, sync: bool = False) -> RLConfig:
    if name not in _CONFIGS:
        available = ", ".join(sorted(_CONFIGS.keys()))
        raise ValueError(f"Unknown config: {name}. Available configs: {available}")
    return _CONFIGS[name](sync)


def list_configs() -> list[str]:
    return sorted(_CONFIGS.keys())


_configs_dir = Path(__file__).parent
_exclude = {"__init__.py", "base.py"}

for _file in _configs_dir.glob("*.py"):
    if _file.name in _exclude:
        continue

    _module_name = _file.stem
    _config_name = _module_name.replace("_", "-")

    try:
        _module = importlib.import_module(f".{_module_name}", package="configs")
        if hasattr(_module, "get_config"):
            _CONFIGS[_config_name] = _module.get_config
    except Exception as exc:
        print(f"Warning: Failed to load config {_config_name}: {exc}")
