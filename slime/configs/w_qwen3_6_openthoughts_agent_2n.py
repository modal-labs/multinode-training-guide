"""Qwen3.6-35B-A3B on OpenThoughts-Agent-v1-RL (harbor) — colocated, 2× H200:8."""

from configs.base import CHECKPOINTS_PATH, run_tag
from configs.datasets import eval_datasets, pull, subsample, train_path
from configs.w_qwen3_6_swe_colocate_1n import _Slime as _SweSlime, modal  # noqa: F401

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-openthoughts-agent-colocate-1n")

# Train on openthoughts_agent; held-out transfer eval: usaco (50) + openthoughts_tblite.
_TRAIN = "openthoughts_agent"
_EVAL = [("usaco", 50), ("openthoughts_tblite", None)]


class _Slime(_SweSlime):
    # ── Task data: OpenThoughts-Agent (harbor) instead of SWE-Gym-Lite ───────
    prompt_data = train_path(_TRAIN)

    # ── Colocate / sync ──────────────────────────────────────────────────────
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    rollout_num_gpus_per_engine = 8
    sglang_mem_fraction_static = 0.85

    # ── Eval ─────────────────────────────────────────────────────────────────
    # Held-out transfer. Each pass blocks the train loop on the shared engines —
    # keep small (full sweeps: w_qwen3_6_swe_eval).
    eval_interval = 20
    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": eval_datasets(_EVAL),
    }

    environment = {
        **_SweSlime.environment,
        "SLIME_AGENT_SANDBOX_CPU": "2",
    }
    # Short episodes: longer than the default to cover cold Dockerfile builds.
    custom_config_path = {
        **_SweSlime.custom_config_path,
        "agentic_episode_timeout": 1200,
        "agentic_eval_timeout": 600,
    }

    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG

    def download_data(self) -> None:
        """Pull openthoughts_agent (train) + the transfer-eval datasets onto /data."""
        for key in {_TRAIN, *(k for k, _ in _EVAL)}:
            pull(key)
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)


slime = _Slime()
