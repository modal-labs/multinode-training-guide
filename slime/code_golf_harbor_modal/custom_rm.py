from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import modal

from code_utils import code_size_bytes, extract_python_code

_DEFAULT_MODAL_APP_NAME = "slime-harbor-rm"
_DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"
_DEFAULT_TASK_ROOT = "/data/mbpp_harbor/tasks"

_app_cache: modal.App | None = None
_sandbox_image = modal.Image.from_registry(_DEFAULT_SANDBOX_IMAGE).entrypoint([])


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


async def _get_modal_app() -> modal.App:
    global _app_cache
    if _app_cache is None:
        app_name = os.environ.get("HARBOR_RM_MODAL_APP", _DEFAULT_MODAL_APP_NAME)
        _app_cache = await modal.App.lookup.aio(app_name, create_if_missing=True)
    return _app_cache


async def _read_remote_file(sandbox: modal.Sandbox, remote_path: str) -> str:
    async with await sandbox.open.aio(remote_path, "rb") as handle:
        chunks: list[bytes] = []
        while True:
            chunk = await handle.read.aio(8192)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8")


async def _write_remote_file(sandbox: modal.Sandbox, remote_path: str, content: str) -> None:
    parent = str(Path(remote_path).parent)
    await sandbox.exec.aio("bash", "-lc", f"mkdir -p {parent}")
    async with await sandbox.open.aio(remote_path, "wb") as handle:
        await handle.write.aio(content.encode("utf-8"))


def _extract_label(sample: Any) -> dict[str, Any]:
    if not getattr(sample, "label", None):
        return {}
    try:
        return json.loads(sample.label)
    except json.JSONDecodeError:
        return {}


def _harbor_task_abs_path(label_payload: dict[str, Any]) -> str:
    root = Path(os.environ.get("HARBOR_TASKS_ROOT", _DEFAULT_TASK_ROOT))
    rel = label_payload.get("harbor_task_rel")
    if rel:
        return str((root.parent / rel).resolve())
    task_id = label_payload.get("task_id")
    if task_id is None:
        raise ValueError("Missing task_id and harbor_task_rel in sample label.")
    return str((root / f"mbpp_{int(task_id):04d}").resolve())


def _compose_reward(
    pass_rate: float,
    candidate_size: int,
    reference_size: int,
    length_weight: float,
) -> float:
    if pass_rate <= 0:
        return 0.0
    ratio = float(reference_size) / float(max(candidate_size, 1))
    capped_ratio = min(2.0, ratio)
    size_bonus = max(0.0, capped_ratio - 1.0)
    return pass_rate * (1.0 + length_weight * size_bonus)


async def _score_sample(sample: Any, semaphore: asyncio.Semaphore) -> float:
    timeout_sec = _get_env_int("HARBOR_RM_TIMEOUT_SEC", 120)
    length_weight = _get_env_float("HARBOR_LENGTH_BONUS_WEIGHT", 0.2)

    label_payload = _extract_label(sample)
    task_path = _harbor_task_abs_path(label_payload)
    reference_size = int(label_payload.get("reference_bytes", 1))
    candidate_code = extract_python_code(getattr(sample, "response", ""))
    candidate_size = code_size_bytes(candidate_code)

    app = await _get_modal_app()
    volume_name = os.environ.get("HARBOR_DATA_VOLUME_NAME", "").strip()
    volumes = {"/data": modal.Volume.from_name(volume_name)} if volume_name else None

    async with semaphore:
        sandbox = await modal.Sandbox.create.aio(
            app=app,
            image=_sandbox_image,
            timeout=timeout_sec,
            cpu=1,
            memory=2048,
            volumes=volumes,
        )
        try:
            await _write_remote_file(sandbox, "/workspace/solution.py", candidate_code)

            test_script = f"{task_path}/tests/test.sh"
            process = await sandbox.exec.aio(
                "bash",
                "-lc",
                f"mkdir -p /logs/verifier && bash {test_script}",
                timeout=timeout_sec,
            )
            await process.stdout.read.aio()
            await process.stderr.read.aio()
            await process.wait.aio()

            reward_json = await _read_remote_file(sandbox, "/logs/verifier/reward.json")
            result = json.loads(reward_json)
            pass_rate = float(result.get("pass_rate", result.get("reward", 0.0)))
            reward = _compose_reward(pass_rate, candidate_size, reference_size, length_weight)

            if not isinstance(sample.metadata, dict):
                sample.metadata = {}
            sample.metadata["harbor_rm"] = {
                "pass_rate": pass_rate,
                "reference_bytes": reference_size,
                "candidate_bytes": candidate_size,
                "reward": reward,
            }
            return reward
        except Exception as exc:
            if not isinstance(sample.metadata, dict):
                sample.metadata = {}
            sample.metadata["harbor_rm_error"] = repr(exc)
            return 0.0
        finally:
            try:
                await sandbox.terminate.aio()
            except Exception:
                pass


async def custom_rm(args: Any, sample: Any, **kwargs: Any) -> float:
    max_concurrency = _get_env_int("HARBOR_RM_MAX_CONCURRENCY", 64)
    semaphore = asyncio.Semaphore(max_concurrency)
    return await _score_sample(sample, semaphore)
