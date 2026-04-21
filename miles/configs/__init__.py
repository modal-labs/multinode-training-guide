import importlib
from pathlib import Path

_CONFIGS_DIR = Path(__file__).parent
_SKIP = {"base", "__init__", "model_configuration"}


def get_module(name: str):
    try:
        return importlib.import_module(f"configs.{name}")
    except ModuleNotFoundError:
        available = sorted(
            f.stem for f in _CONFIGS_DIR.glob("*.py") if f.stem not in _SKIP
        )
        raise ValueError(f"Unknown config {name!r}. Available: {available}")
