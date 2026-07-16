"""Eval-only run of the Qwen3.6-35B-A3B SWE agentic-RL setup — no training.

Evaluates a list of datasets, one HF repo each (configs/datasets.py). Edit ``_EVAL``
to pick datasets + per-dataset subsample size (``None`` = full held-out eval.jsonl).
"""

from configs.base import CHECKPOINTS_PATH, run_tag
from configs.datasets import eval_datasets, pull, subsample
from configs.w_qwen3_6_swe_colocate_2n import _Slime, modal  # noqa: F401

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-eval")

# (dataset key, subsample n | None). None evals the full held-out eval.jsonl; an
# int subsamples it in download_data. Comment a line out to eval fewer datasets.
_EVAL = [
    ("swebench_verified", None),
    ("swebenchpro", None),
    ("swebench_multilingual", None),
    ("terminal_bench_2_1", None),
    ("swegym_lite", None),   # in-distribution held-out (30)
    # ("usaco", 50),
]


class _SlimeEval(_Slime):
    num_rollout = 0
    eval_interval = 1  # any non-None value arms the eval-only branch
    sglang_server_concurrency = 32
    lr_decay_iters = 1

    # Fast rollout-engine recipe from rollout_profile/colo2n_tp2.py: 8×TP2 engines
    # (dp-attention off) across the 16 GPUs + 64k ctx. mem_fraction stays 0.6 — TP2
    # replicates the model 8× (each engine spans 2 GPUs), so per-GPU KV headroom is
    # tight next to the resident Megatron weights; raising it OOMs (drop ctx to 32k if so).
    rollout_num_gpus = 16
    rollout_num_gpus_per_engine = 2
    sglang_enable_dp_attention = False
    sglang_dp_size = None
    sglang_enable_dp_lm_head = None
    sglang_ep_size = None
    sglang_mem_fraction_static = 0.6
    rollout_max_context_len = 65536

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
