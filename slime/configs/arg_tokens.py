"""Token utilities for opaque SLIME argument passthrough."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import shlex
import subprocess


def parse_args_file(path: Path) -> list[str]:
    """Parse a .args file using shell-like tokenization with comment support."""
    text = path.read_text(encoding="utf-8")
    return shlex.split(text, comments=True, posix=True)


def parse_raw_arg_string(text: str | None) -> list[str]:
    if not text:
        return []
    return shlex.split(text, comments=False, posix=True)


def load_args_files(paths: list[Path] | tuple[Path, ...]) -> list[str]:
    tokens: list[str] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Args file not found: {path}")
        tokens.extend(parse_args_file(path))
    return tokens


def discover_model_args_root() -> Path:
    """Locate SLIME scripts/models root with optional environment override."""
    override = os.environ.get("SLIME_MODEL_ARGS_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if root.exists():
            return root
        raise FileNotFoundError(f"SLIME_MODEL_ARGS_ROOT does not exist: {root}")

    # Prefer SLIME's own helper so we follow the installed package layout.
    try:
        from slime.utils.external_utils import command_utils as command_utils

        root = (Path(command_utils.repo_base_dir) / "scripts" / "models").resolve()
        if root.exists():
            return root
    except Exception:
        pass

    # Fallback to path discovery via package location.
    try:
        import slime

        package_root = Path(inspect.getfile(slime)).resolve().parents[1]
        root = (package_root / "scripts" / "models").resolve()
        if root.exists():
            return root
    except Exception:
        pass

    # Last fallback for this repository layout.
    repo_candidate = (Path(__file__).resolve().parents[2] / "dev" / "slime" / "scripts" / "models").resolve()
    if repo_candidate.exists():
        return repo_candidate

    raise FileNotFoundError(
        "Unable to discover SLIME scripts/models root. "
        "Set SLIME_MODEL_ARGS_ROOT to override discovery."
    )


def source_model_args_tokens(model_args_script: str, model_args_env: dict[str, str] | None = None) -> list[str]:
    """Source MODEL_ARGS from an upstream SLIME shell script."""
    root = discover_model_args_root()
    script_path = (root / model_args_script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(
            f"MODEL_ARGS script not found: {script_path}. "
            f"Resolved root: {root}"
        )

    env_parts = []
    for key, value in (model_args_env or {}).items():
        if not key:
            continue
        env_parts.append(f"export {shlex.quote(str(key))}={shlex.quote(str(value))}")
    env_prefix = "; ".join(env_parts)
    if env_prefix:
        env_prefix = f"{env_prefix}; "

    command = (
        "set -euo pipefail; "
        f"{env_prefix}"
        f"source {shlex.quote(str(script_path))}; "
        'printf "%s\\0" "${MODEL_ARGS[@]}"'
    )
    proc = subprocess.run(
        ["bash", "-lc", command],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Failed to source MODEL_ARGS from {script_path}: {stderr or f'exit {proc.returncode}'}"
        )

    raw = proc.stdout.split(b"\0")
    if raw and raw[-1] == b"":
        raw = raw[:-1]
    tokens = [item.decode("utf-8", errors="strict") for item in raw]
    if not tokens:
        raise RuntimeError(f"MODEL_ARGS is empty after sourcing script: {script_path}")
    return tokens
