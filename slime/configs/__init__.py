"""Config registry for SLIME GRPO training."""

import importlib
from pathlib import Path
from typing import Callable
from .base import RLConfig

# Registry of available configs
_CONFIGS: dict[str, Callable[[], RLConfig]] = {}


def get_config(name: str) -> RLConfig:
    """Get a config by name."""
    if name not in _CONFIGS:
        available = ", ".join(sorted(_CONFIGS.keys()))
        raise ValueError(f"Unknown config: {name}. Available configs: {available}")
    return _CONFIGS[name]()


def list_configs() -> list[str]:
    """List all available config names."""
    return sorted(_CONFIGS.keys())


# Auto-discover and register configs
_configs_dir = Path(__file__).parent
_exclude = {"__init__.py", "base.py"}

for _file in _configs_dir.glob("*.py"):
    if _file.name in _exclude:
        continue
    
    _module_name = _file.stem  # e.g., "qwen_4b" or "qwen_4b_3T1R"
    _config_name = _module_name.replace("_", "-")  # e.g., "qwen-4b" or "qwen-4b-3T1R"
    
    _module = importlib.import_module(f".{_module_name}", package="configs")
    
    if hasattr(_module, "get_config"):
        _CONFIGS[_config_name] = _module.get_config
