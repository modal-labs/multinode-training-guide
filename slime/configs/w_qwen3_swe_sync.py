"""Qwen3-30B-A3B SWE agentic RL — sync, non-colocated / disaggregated (2× H200:8).

Same rollout/model/data/algorithm and GPU layout as ``w_qwen3_swe_async`` (8
train + 8 separate rollout GPUs), but driven by ``train.py`` (sync): rollout to
completion, then train, then push fresh weights — fully on-policy, step wall =
rollout + train (the async parent instead serves continuously and trains on
stale rollouts, ~= max(rollout, train)). Engine and Megatron never share GPUs,
so no offload dance and the engine keeps the full mem_fraction_static=0.85 pool.
Pick this for strictly on-policy data; ``w_qwen3_swe_colocate_*`` to cut cost.
Checkpoint, data, agent env and algorithm inherited unchanged.
"""

from configs import w_qwen3_swe_async as _base
from configs.base import CHECKPOINTS_PATH, run_tag

# Same image/overlay as the parent.
modal = _base.modal

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3-30b-a3b-swe-gym-lite-sync")


class _Slime(_base._Slime):
    # ── Sync / non-colocate (disaggregated) ───────────────────────────────────
    async_mode = False
    colocate = False
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    rollout_num_gpus = 8  # → total_nodes() == 2 (8 actor + 8 rollout)
    update_weights_interval = 1

    # ── WandB ──────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _Slime()
