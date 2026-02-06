"""Config registry for TOML thin wrappers."""

from __future__ import annotations

from pathlib import Path

from .toml_loader import discover_toml_configs, resolve_toml_config
from .types import SlimExperimentConfig

_CONFIGS_DIR = Path(__file__).parent
_RAW_TOML_CONFIGS = discover_toml_configs(_CONFIGS_DIR)


def get_config(name: str) -> SlimExperimentConfig:
    """Get a resolved config by name."""
    if name not in _RAW_TOML_CONFIGS:
        available = ", ".join(sorted(list_configs()))
        raise ValueError(f"Unknown config: {name}. Available configs: {available}")
    return resolve_toml_config(name, _RAW_TOML_CONFIGS)


def list_configs() -> list[str]:
    """List public config names."""
    return sorted(
        name
        for name, raw in _RAW_TOML_CONFIGS.items()
        if not raw.is_hidden
    )
