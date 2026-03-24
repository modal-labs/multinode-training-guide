"""Miles custom agent hook that runs Harbor trials inline."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    task_path = metadata.get("harbor_task_path")
    if not task_path:
        return {"reward": 0.0, "exit_status": "missing-task-path", "agent_metrics": {}}

    task_mode = metadata.get("harbor_task_mode", "hello")
    trial_name = metadata.get("harbor_task_name", Path(task_path).name)
    model_name = os.getenv("AGENT_MODEL_NAME", "model")

    started = time.time()
    config = TrialConfig(
        task=TaskConfig(path=Path(task_path)),
        trials_dir=Path("/tmp/harbor-trials"),
        trial_name=f"{trial_name}-{int(started * 1000)}-{uuid.uuid4().hex[:8]}",
        agent=AgentConfig(
            import_path="harbor_agent:SimpleHarborAgent",
            model_name=model_name,
            kwargs={
                "base_url": f"{base_url}/v1",
                "request_kwargs": request_kwargs,
                "task_mode": task_mode,
            },
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType.MODAL,
            force_build=False,
            delete=True,
            kwargs={
                "sandbox_timeout_secs": 60 * 30,
                "sandbox_idle_timeout_secs": 60 * 5,
            },
        ),
    )

    trial = Trial(config)
    result = await trial.run()

    reward = 0.0
    if result.verifier_result and result.verifier_result.rewards:
        reward = float(result.verifier_result.rewards.get("reward", 0.0))

    agent_metrics = {
        "total_time": time.time() - started,
        "model_latency": (
            result.agent_result.metadata.get("model_latency", 0.0)
            if result.agent_result and result.agent_result.metadata
            else 0.0
        ),
        "verifier_latency": (
            (
                result.verifier.finished_at - result.verifier.started_at
            ).total_seconds()
            if result.verifier and result.verifier.started_at and result.verifier.finished_at
            else 0.0
        ),
    }

    return {
        "reward": reward,
        "exit_status": "ok" if result.exception_info is None else result.exception_info.exception_type,
        "eval_report": result.verifier_result.rewards if result.verifier_result else {},
        "agent_metrics": agent_metrics,
    }
