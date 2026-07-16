"""Eval-only run of the Qwen3.6-35B-A3B SWE agentic-RL setup — no training.

Reuses ``w_qwen3_6_swe_colocate_1n`` (already ``async_mode = False``).
``num_rollout = 0`` routes through train.py's stock eval-only branch (eval once,
exit) — for baselining the base model or sweeping a checkpoint.

Datasets: one HF repo each (configs/datasets.py). Edit ``_EVAL`` to pick datasets
+ per-dataset subsample size (``None`` = full held-out eval.jsonl).

    EXPERIMENT_CONFIG=w_qwen3_6_swe_eval uv run --no-dev modal run slime/modal_train.py::download_data
    EXPERIMENT_CONFIG=w_qwen3_6_swe_eval uv run --no-dev modal run -d slime/modal_train.py::train
"""

from configs.base import CHECKPOINTS_PATH, run_tag
from configs.datasets import eval_datasets, pull, subsample
from configs.w_qwen3_6_swe_colocate_1n import _Slime, modal  # noqa: F401

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-eval")

# (dataset key, subsample n | None). None evals the full held-out eval.jsonl.
_EVAL = [
    ("swebench_verified", None),
    ("swebenchpro", None),
    ("swebench_multilingual", None),
    ("terminal_bench_2_1", None),
    # ("swegym_lite", 100), ("usaco", 50),  # in-distribution / transfer
]


class _SlimeEval(_Slime):
    # async_mode=False already, so num_rollout=0 routes through train.py's
    # eval-only branch (eval once, exit).
    num_rollout = 0
    eval_interval = 1  # any non-None value arms the eval-only branch

    # slime derives train_iters from num_rollout, so eval-only computes
    # lr_decay_steps == 0 and trips Megatron's `assert lr_decay_steps > 0`. Pin
    # a dummy 1-iter schedule; no optimizer step runs.
    lr_decay_iters = 1

    # Point `load` at a Megatron dir to eval a trained checkpoint; else the base
    # hf_checkpoint weights are evaluated.
    # load = f"{CHECKPOINTS_PATH}/<run>/..."

    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": eval_datasets(_EVAL),
    }

    wandb_group = _RUN_TAG
    # Override the base's dump path so rollouts land under THIS run's tag (the
    # eval W&B name), not the inherited train-config tag.
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    def download_data(self) -> None:
        """Pull each eval dataset's repo into /data/<key>/, then subsample where asked."""
        for key, _ in _EVAL:
            pull(key)
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)


slime = _SlimeEval()
