"""Config registry for SLIME GRPO training."""

from typing import Callable
from .base import TrainingConfig

# Registry of available configs
_CONFIGS: dict[str, Callable[[], TrainingConfig]] = {}


def get_config(name: str) -> TrainingConfig:
    """Get a config by name."""
    if name not in _CONFIGS:
        available = ", ".join(sorted(_CONFIGS.keys()))
        raise ValueError(f"Unknown config: {name}. Available configs: {available}")
    return _CONFIGS[name]()


def list_configs() -> list[str]:
    """List all available config names."""
    return sorted(_CONFIGS.keys())


# Import and register configs
from . import qwen_0_5b, qwen_4b

_CONFIGS["qwen-0.5b-sync"] = qwen_0_5b.get_config_sync
_CONFIGS["qwen-0.5b-async"] = qwen_0_5b.get_config_async
_CONFIGS["qwen-4b-sync"] = qwen_4b.get_config_sync
_CONFIGS["qwen-4b-async"] = qwen_4b.get_config_async
