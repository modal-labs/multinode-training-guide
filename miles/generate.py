"""Reward and rollout wrappers for Harbor-backed Miles runs."""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial
from transformers import AutoTokenizer

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_REWARD_CONCURRENCY = int(os.environ.get("HARBOR_REWARD_CONCURRENCY", "2"))
_REWARD_SEMAPHORE = asyncio.Semaphore(_REWARD_CONCURRENCY)
_TOKENIZER = None
_TOKENIZER_ID = None
_SAFE_STUB = "import sys\n\ndef main():\n    sys.stdout.write('')\n\nif __name__ == '__main__':\n    main()\n"


def _get_tokenizer(args):
    global _TOKENIZER
    global _TOKENIZER_ID
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if _TOKENIZER is None or _TOKENIZER_ID != hf_checkpoint:
        _TOKENIZER = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
        _TOKENIZER_ID = hf_checkpoint
    return _TOKENIZER


def _extract_python_candidate(text: str) -> str:
    if not text:
        return ""

    candidate = text.strip()
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[-1].strip()
    elif "<think>" in candidate:
        return ""

    if "```python" in candidate:
        parts = candidate.split("```python")
        candidate = parts[-1].split("```", 1)[0].strip()
    elif "```" in candidate:
        parts = candidate.split("```")
        if len(parts) >= 3:
            candidate = parts[-2].strip()

    for marker in ("<|assistant|>", "<|user|>", "<|system|>", "<sop>", "[gMASK]"):
        if marker in candidate:
            candidate = candidate.rsplit(marker, 1)[-1].strip()

    return candidate.strip()


def _looks_like_python(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    python_signals = (
        "def ",
        "class ",
        "import ",
        "from ",
        "for ",
        "while ",
        "if ",
        "print(",
        "sys.stdin",
        "sys.stdout",
        "input(",
    )
    return any(signal in stripped for signal in python_signals)


def _stabilize_sample(args, sample: Sample) -> None:
    tokenizer = _get_tokenizer(args)
    prompt_tokens = sample.tokens[:-sample.response_length] if sample.response_length > 0 else list(sample.tokens)

    candidate = _extract_python_candidate(sample.response)
    needs_stub = (
        sample.status == Sample.Status.TRUNCATED
        or not _looks_like_python(candidate)
        or len(candidate) > 2048
    )
    if needs_stub:
        candidate = _SAFE_STUB

    response_tokens = tokenizer.encode(candidate, add_special_tokens=False)
    if not response_tokens:
        candidate = _SAFE_STUB
        response_tokens = tokenizer.encode(candidate, add_special_tokens=False)

    sample.response = candidate
    sample.response_length = len(response_tokens)
    sample.tokens = prompt_tokens + response_tokens
    sample.loss_mask = [1] * sample.response_length
    if sample.rollout_log_probs is not None:
        sample.rollout_log_probs = [0.0] * sample.response_length


async def _score_sample_with_harbor(args, sample: Sample) -> float:
    _stabilize_sample(args, sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_path = metadata.get("harbor_task_path")
    if not task_path:
        sample.metadata["reward"] = 0.0
        sample.metadata["exit_status"] = "missing-task-path"
        return 0.0

    task_mode = metadata.get("harbor_task_mode", "hello")
    task_name = metadata.get("harbor_task_name", Path(task_path).name)
    started = time.time()

    config = TrialConfig(
        task=TaskConfig(path=Path(task_path)),
        trials_dir=Path("/tmp/harbor-trials"),
        trial_name=f"{task_name}-{int(started * 1000)}-{uuid.uuid4().hex[:8]}",
        agent=AgentConfig(
            import_path="harbor_agent:SimpleHarborAgent",
            model_name=os.getenv("AGENT_MODEL_NAME", "model"),
            kwargs={
                "base_url": "http://127.0.0.1:1/v1",
                "task_mode": task_mode,
                "submitted_response": sample.response,
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
    eval_report = {}
    if result.verifier_result and result.verifier_result.rewards:
        eval_report = dict(result.verifier_result.rewards)
        reward = float(eval_report.get("reward", 0.0))

    verifier_latency = 0.0
    if result.verifier and result.verifier.started_at and result.verifier.finished_at:
        verifier_latency = (
            result.verifier.finished_at - result.verifier.started_at
        ).total_seconds()

    sample.metadata.update(
        {
            "reward": reward,
            "eval_report": eval_report,
            "exit_status": "ok" if result.exception_info is None else result.exception_info.exception_type,
            "agent_metrics": {
                "total_time": time.time() - started,
                "model_latency": (
                    result.agent_result.metadata.get("model_latency", 0.0)
                    if result.agent_result and result.agent_result.metadata
                    else 0.0
                ),
                "verifier_latency": verifier_latency,
            },
        }
    )
    return reward


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(samples, list):
        rewards = []
        for sample in samples:
            async with _REWARD_SEMAPHORE:
                rewards.append(await _score_sample_with_harbor(args, sample))
        return rewards
    async with _REWARD_SEMAPHORE:
        return await _score_sample_with_harbor(args, samples)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    metrics = {}
    all_metrics = [
        s.metadata.get("agent_metrics", {})
        for s in samples
        if getattr(s, "metadata", None) and s.metadata.get("agent_metrics")
    ]
    if not all_metrics:
        return metrics

    for key in ["total_time", "model_latency", "verifier_latency"]:
        values = [m.get(key, 0.0) for m in all_metrics if key in m]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    rewards = [float(s.metadata.get("reward", 0.0)) for s in samples]
    if rewards:
        metrics["agent/reward_mean"] = sum(rewards) / len(rewards)
        metrics["agent/reward_max"] = max(rewards)

    return metrics


class RolloutFn(InferenceRolloutFn):
    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)

        flat_samples = []
        for group in output.samples:
            if isinstance(group, list):
                flat_samples.extend(group)
            else:
                flat_samples.append(group)

        metrics = aggregate_agent_metrics(flat_samples)
        if metrics:
            output.metrics = output.metrics or {}
            output.metrics.update(metrics)
            logger.info("Harbor rollout metrics for rollout %s: %s", input.rollout_id, metrics)

        return output
