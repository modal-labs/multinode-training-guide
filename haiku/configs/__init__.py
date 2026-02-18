"""Config registry for SLIME GRPO training."""

import importlib
from pathlib import Path
from typing import Callable
from .base import RLConfig

# Registry of available configs
_CONFIGS: dict[str, Callable[[str], RLConfig]] = {}


def get_config(name: str, run_name: str = "") -> RLConfig:
    """Get a config by name."""
    if name not in _CONFIGS:
        available = ", ".join(sorted(_CONFIGS.keys()))
        raise ValueError(f"Unknown config: {name}. Available configs: {available}")
    return _CONFIGS[name](run_name=run_name)


def list_configs() -> list[str]:
    """List all available config names."""
    return sorted(_CONFIGS.keys())


# Auto-discover and register configs
_configs_dir = Path(__file__).parent
_test_configs_dir = _configs_dir.parent / "test-configs"
_exclude = {"__init__.py", "base.py"}


def _load_config_from_file(file_path: Path) -> Callable[[], RLConfig] | None:
    """Load a config file that uses 'from .base import ...' syntax."""
    import importlib.util
    import sys

    module_name = f"configs._test_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "configs"  # So 'from .base import' works
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "get_config"):
        return module.get_config
    return None


# Register from main configs directory
for _file in _configs_dir.glob("*.py"):
    if _file.name in _exclude:
        continue

    _module_name = _file.stem
    _config_name = _module_name.replace("_", "-")

    try:
        _module = importlib.import_module(f".{_module_name}", package="configs")
        if hasattr(_module, "get_config"):
            _CONFIGS[_config_name] = _module.get_config
    except Exception as e:
        print(f"Warning: Failed to load config {_config_name}: {e}")

# Register from test-configs directory (if exists)
if _test_configs_dir.exists():
    for _file in _test_configs_dir.glob("*.py"):
        if _file.name in _exclude:
            continue

        _config_name = _file.stem.replace("_", "-")

        # Don't override main configs
        if _config_name in _CONFIGS:
            continue

        try:
            _getter = _load_config_from_file(_file)
            if _getter:
                _CONFIGS[_config_name] = _getter
        except Exception as e:
            print(f"Warning: Failed to load test config {_config_name}: {e}")
