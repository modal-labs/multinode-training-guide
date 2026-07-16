"""Qwen3.6-35B-A3B USACO (harbor) agentic RL — colocated, single node (1× H200:8).

Delta on ``w_qwen3_6_swe_colocate_1n``: same model/parallelism/optimizer/topology,
only the task changes. Harbor rows boot the task's Dockerfile sandbox (all USACO
tasks share one ``python:3.13-slim`` image -> one cached build), mini-swe-agent
writes ``solution.py``, in-sandbox ``tests/test.sh`` grades. Trains on usaco;
evals the in-distribution held-out usaco slice (50) + openthoughts_tblite transfer.

Sanity check before burning GPU-hours (expect reward=1.0)::

    uv run --with modal python -m agentic_rl.environment.harbor \
        <local-out>/usaco.jsonl --task-root <local-out> --limit 3
"""

from configs.base import CHECKPOINTS_PATH, run_tag
from configs.datasets import eval_datasets, pull, subsample, train_path
from configs.w_qwen3_6_swe_colocate_1n import _Slime as _SweSlime, modal  # noqa: F401

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-usaco-colocate-1n")

# Train on usaco; eval the in-distribution held-out usaco slice (50) + transfer.
_TRAIN = "usaco"
_EVAL = [("usaco", 50), ("openthoughts_tblite", None)]


class _Slime(_SweSlime):
    # ── Task data: harbor-converted USACO instead of harbor SWE-Gym-Lite ─────
    prompt_data = train_path(_TRAIN)

    # ── Eval ─────────────────────────────────────────────────────────────────
    # Each pass blocks the train loop on the shared engines — keep small (full
    # sweeps: w_qwen3_6_swe_eval).
    eval_interval = 20
    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": eval_datasets(_EVAL),
    }

    environment = {
        **_SweSlime.environment,
        # Full cores so a correct-but-slow USACO solution isn't graded TLE.
        "SLIME_AGENT_SANDBOX_CPU": "2",
    }
    # Shorter episode/grade budgets than SWE (override the SWE base's limits).
    custom_config_path = {
        **_SweSlime.custom_config_path,
        "agentic_episode_timeout": 900,
        "agentic_eval_timeout": 900,
    }

    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG

    def download_data(self) -> None:
        """Pull usaco (train + held-out eval) + openthoughts_tblite onto /data."""
        for key in {_TRAIN, *(k for k, _ in _EVAL)}:
            pull(key)
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)


slime = _Slime()
