from __future__ import annotations

import json
import os
import shlex
import time
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


def _get_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


async def _get_modal_app() -> modal.App:
    global _app_cache
    if _app_cache is None:
        app_name = os.environ.get("HARBOR_RM_MODAL_APP", _DEFAULT_MODAL_APP_NAME)
        _app_cache = await modal.App.lookup.aio(app_name, create_if_missing=True)
    return _app_cache


async def _read_remote_file(sandbox: modal.Sandbox, remote_path: str) -> str:
    process = await sandbox.exec.aio("bash", "-lc", f"cat {shlex.quote(remote_path)}")
    stdout = await process.stdout.read.aio()
    stderr = await process.stderr.read.aio()
    return_code = await process.wait.aio()
    if return_code != 0:
        raise RuntimeError(f"cat {remote_path}: {stderr}")
    return stdout


async def _write_remote_file(sandbox: modal.Sandbox, remote_path: str, content: str) -> None:
    parent = str(Path(remote_path).parent)
    delimiter = "__SLIME_CONTENT_EOF__"
    while delimiter in content:
        delimiter += "_X"
    cmd = (
        f"mkdir -p {shlex.quote(parent)} && "
        f"cat > {shlex.quote(remote_path)} <<'{delimiter}'\n"
        f"{content}\n"
        f"{delimiter}\n"
    )
    process = await sandbox.exec.aio("bash", "-lc", cmd)
    stderr = await process.stderr.read.aio()
    return_code = await process.wait.aio()
    if return_code != 0:
        raise RuntimeError(f"write {remote_path}: {stderr}")


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


async def _score_sample(sample: Any) -> float:
    profile_enabled = _get_env_bool("HARBOR_RM_PROFILE", default=False)
    log_samples = _get_env_bool("HARBOR_RM_LOG_SAMPLES", default=False)
    t0 = time.perf_counter()
    timings_s: dict[str, float] = {}

    def _mark(name: str) -> None:
        if not profile_enabled:
            return
        now = time.perf_counter()
        timings_s[name] = now

    timeout_sec = _get_env_int("HARBOR_RM_TIMEOUT_SEC", 120)
    length_weight = _get_env_float("HARBOR_LENGTH_BONUS_WEIGHT", 0.2)

    label_payload = _extract_label(sample)
    task_path = _harbor_task_abs_path(label_payload)
    reference_size = int(label_payload.get("reference_bytes", 1))
    candidate_code = extract_python_code(getattr(sample, "response", ""))
    candidate_size = code_size_bytes(candidate_code)
    _mark("prepared_sample")

    app = await _get_modal_app()
    volume_name = os.environ.get("HARBOR_DATA_VOLUME_NAME", "").strip()
    volumes = {"/data": modal.Volume.from_name(volume_name)} if volume_name else None
    _mark("prepared_runtime")

    sandbox = await modal.Sandbox.create.aio(
        app=app,
        image=_sandbox_image,
        timeout=timeout_sec,
        cpu=1,
        memory=2048,
        volumes=volumes,
    )
    _mark("sandbox_created")
    try:
        await _write_remote_file(sandbox, "/workspace/solution.py", candidate_code)
        _mark("solution_uploaded")

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
        _mark("test_finished")

        reward_json = await _read_remote_file(sandbox, "/logs/verifier/reward.json")
        result = json.loads(reward_json)
        pass_rate = float(result.get("pass_rate", result.get("reward", 0.0)))
        reward = _compose_reward(pass_rate, candidate_size, reference_size, length_weight)
        _mark("reward_parsed")

        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["harbor_rm"] = {
            "pass_rate": pass_rate,
            "reference_bytes": reference_size,
            "candidate_bytes": candidate_size,
            "reward": reward,
        }
        if log_samples:
            print(
                "harbor_rm_sample: "
                f"task_id={label_payload.get('task_id')} "
                f"pass_rate={pass_rate:.6f} "
                f"reward={reward:.6f} "
                f"candidate_bytes={candidate_size} "
                f"reference_bytes={reference_size}"
            )
        if profile_enabled:
            timestamps = {**timings_s, "done": time.perf_counter()}
            timing_ms = {
                "prepare": round((timestamps.get("prepared_sample", t0) - t0) * 1000, 2),
                "runtime_setup": round(
                    (timestamps.get("prepared_runtime", timestamps.get("prepared_sample", t0))
                     - timestamps.get("prepared_sample", t0))
                    * 1000,
                    2,
                ),
                "sandbox_create": round(
                    (timestamps.get("sandbox_created", timestamps.get("prepared_runtime", t0))
                     - timestamps.get("prepared_runtime", t0))
                    * 1000,
                    2,
                ),
                "upload_solution": round(
                    (timestamps.get("solution_uploaded", timestamps.get("sandbox_created", t0))
                     - timestamps.get("sandbox_created", t0))
                    * 1000,
                    2,
                ),
                "test_exec": round(
                    (timestamps.get("test_finished", timestamps.get("solution_uploaded", t0))
                     - timestamps.get("solution_uploaded", t0))
                    * 1000,
                    2,
                ),
                "read_reward": round(
                    (timestamps.get("reward_parsed", timestamps.get("test_finished", t0))
                     - timestamps.get("test_finished", t0))
                    * 1000,
                    2,
                ),
                "total": round((timestamps["done"] - t0) * 1000, 2),
            }
            sample.metadata["harbor_rm_timing_ms"] = timing_ms
            print(f"harbor_rm_timing_ms: {timing_ms}")
        return reward
    except Exception as exc:
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        sample.metadata["harbor_rm_error"] = repr(exc)
        print(f"harbor_rm_error: {exc!r}")
        if log_samples:
            print(
                "harbor_rm_sample_error: "
                f"task_id={label_payload.get('task_id')} "
                f"error={exc!r} "
                f"candidate_bytes={candidate_size} "
                f"reference_bytes={reference_size}"
            )
        if profile_enabled:
            timing_ms = {
                "total_until_error": round((time.perf_counter() - t0) * 1000, 2)
            }
            sample.metadata["harbor_rm_timing_ms"] = timing_ms
            print(f"harbor_rm_timing_ms_error: {timing_ms}")
        return 0.0
    finally:
        try:
            await sandbox.terminate.aio()
        except Exception:
            pass


async def custom_rm(args: Any, sample: Any, **kwargs: Any) -> float:
    return await _score_sample(sample)
