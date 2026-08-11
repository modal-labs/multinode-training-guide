import importlib
from pathlib import Path

_CONFIGS_DIR = Path(__file__).parent
_SKIP = {"base", "__init__"}


def get_module(name: str):
    try:
        return importlib.import_module(f"configs.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"configs.{name}":
            raise
        available = sorted(
            f.relative_to(_CONFIGS_DIR).with_suffix("").as_posix().replace("/", ".")
            for f in _CONFIGS_DIR.rglob("*.py")
            if f.stem not in _SKIP
        )
        raise ValueError(f"Unknown config {name!r}. Available: {available}") from exc
