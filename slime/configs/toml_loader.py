"""TOML-first config loader for thin SLIME experiment wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from .types import SlimExperimentConfig


@dataclass(frozen=True)
class RawTomlConfig:
    name: str
    path: Path
    data: dict[str, Any]
    is_hidden: bool = False


def _config_name_from_path(path: Path) -> tuple[str, bool]:
    stem = path.stem
    is_hidden = stem.startswith("_")
    if is_hidden:
        stem = stem[1:]
    return stem.replace("_", "-"), is_hidden


def discover_toml_configs(configs_dir: Path) -> dict[str, RawTomlConfig]:
    """Discover TOML configs from the canonical configs directory."""
    discovered: dict[str, RawTomlConfig] = {}

    if not configs_dir.exists():
        return discovered

    for path in sorted(configs_dir.glob("*.toml")):
        name, is_hidden = _config_name_from_path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        discovered[name] = RawTomlConfig(name=name, path=path, data=data, is_hidden=is_hidden)

    return discovered


def resolve_toml_config(name: str, raw_configs: dict[str, RawTomlConfig]) -> SlimExperimentConfig:
    """Resolve a config by name with inheritance and normalization."""
    if name not in raw_configs:
        raise KeyError(name)

    visited: set[str] = set()
    stack: list[str] = []

    def _resolve(cfg_name: str) -> tuple[dict[str, Any], Path]:
        if cfg_name in visited:
            cycle = " -> ".join([*stack, cfg_name])
            raise ValueError(f"Config inheritance cycle detected: {cycle}")
        if cfg_name not in raw_configs:
            raise ValueError(f"Config '{stack[-1]}' extends unknown config '{cfg_name}'")

        visited.add(cfg_name)
        stack.append(cfg_name)
        raw = raw_configs[cfg_name]
        child = dict(raw.data)

        merged: dict[str, Any] = {}
        base_path = raw.path

        parent_name = child.pop("extends", None)
        if parent_name:
            if not isinstance(parent_name, str):
                raise ValueError(f"{raw.path}: 'extends' must be a string")
            parent_cfg, parent_base = _resolve(parent_name)
            merged = dict(parent_cfg)
            base_path = parent_base

        # args files are append-merged; children extend parent presets.
        parent_args_files = list(merged.get("args_files", []))
        child_args_files = child.get("args_files", [])
        if child_args_files is None:
            child_args_files = []
        if not isinstance(child_args_files, list):
            raise ValueError(f"{raw.path}: 'args_files' must be a list")
        child_args_files_resolved: list[str] = []
        for item in child_args_files:
            if not isinstance(item, str):
                raise ValueError(f"{raw.path}: 'args_files' entries must be strings")
            item_path = Path(item)
            if not item_path.is_absolute():
                item_path = (raw.path.parent / item_path).resolve()
            child_args_files_resolved.append(str(item_path))
        merged["args_files"] = [*parent_args_files, *child_args_files_resolved]

        # args are append-merged.
        parent_args = list(merged.get("args", []))
        child_args = child.get("args", [])
        if child_args is None:
            child_args = []
        if not isinstance(child_args, list):
            raise ValueError(f"{raw.path}: 'args' must be a list")
        merged["args"] = [*parent_args, *child_args]

        # model_args_env merges by key.
        parent_model_env = dict(merged.get("model_args_env", {}))
        child_model_env = child.get("model_args_env", {})
        if child_model_env is None:
            child_model_env = {}
        if not isinstance(child_model_env, dict):
            raise ValueError(f"{raw.path}: 'model_args_env' must be a table")
        parent_model_env.update({str(k): str(v) for k, v in child_model_env.items()})
        merged["model_args_env"] = parent_model_env

        for key, value in child.items():
            if key in {"args_files", "args", "model_args_env"}:
                continue
            merged[key] = value

        # Keep the leaf path for relative file resolution.
        merged["__leaf_path__"] = raw.path

        stack.pop()
        visited.remove(cfg_name)
        return merged, base_path

    data, _ = _resolve(name)
    leaf_path = Path(data.pop("__leaf_path__"))

    model_id = data.get("model_id")
    model_args_script = data.get("model_args_script")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"{leaf_path}: 'model_id' is required")
    if not isinstance(model_args_script, str) or not model_args_script:
        raise ValueError(f"{leaf_path}: 'model_args_script' is required")

    sync = bool(data.get("sync", False))
    train_script = data.get("train_script")
    if train_script is None:
        train_script = "slime/train.py" if sync else "slime/train_async.py"
    if not isinstance(train_script, str) or not train_script:
        raise ValueError(f"{leaf_path}: 'train_script' must be a non-empty string")

    args_files_value = data.get("args_files", [])
    args_files: list[Path] = []
    for item in args_files_value:
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            path = (leaf_path.parent / path).resolve()
        args_files.append(path.resolve())

    args_value = data.get("args", [])
    args: list[str] = []
    for item in args_value:
        args.append(str(item))

    app_name = str(data.get("app_name", "slime-grpo"))
    gpu = str(data.get("gpu", "H100:8"))
    n_nodes = int(data.get("n_nodes", 4))
    wandb_project = str(data.get("wandb_project", "slime-grpo"))
    wandb_run_name_prefix = str(data.get("wandb_run_name_prefix", ""))
    model_args_env = {str(k): str(v) for k, v in dict(data.get("model_args_env", {})).items()}

    return SlimExperimentConfig(
        name=name,
        source_path=leaf_path,
        model_id=model_id,
        model_args_script=model_args_script,
        model_args_env=model_args_env,
        args_files=tuple(args_files),
        args=tuple(args),
        app_name=app_name,
        n_nodes=n_nodes,
        gpu=gpu,
        wandb_project=wandb_project,
        wandb_run_name_prefix=wandb_run_name_prefix,
        train_script=train_script,
        sync=sync,
    )
