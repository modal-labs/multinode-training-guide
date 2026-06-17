"""Qwen3-30B-A3B SWE agentic RL — Option B: colocated, two nodes (2× H200:8).

Same rollout as ``w_qwen3_swe_async`` but BOTH nodes time-share training and
inference (sync): rollout on 2 TP8 engines (sgl-router load-balanced, twice the
async KV pool), then engines offload and Megatron trains across all 16 GPUs
(TP4 x CP2 -> DP=2; the TP4/PP1 checkpoint loads unchanged). Rollout is
latency-bound, so the extra engine GPUs mostly buy KV headroom, not speed — pick
this only if train time dominates or KV is the binding constraint, else prefer
``w_qwen3_swe_colocate_1n`` (cost) or the async parent (step wall).
Checkpoint, data, agent env and algorithm inherited unchanged.
"""

from configs import w_qwen3_swe_async as _base
from configs.base import CHECKPOINTS_PATH, run_tag

# Same image/overlay as the parent.
modal = _base.modal

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3-30b-a3b-swe-gym-lite-colocate-2n")


class _Slime(_base._Slime):
    # ── Colocate / sync ──────────────────────────────────────────────────────
    async_mode = False  # train.py: rollout and train alternate on the same GPUs
    colocate = True
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    rollout_num_gpus = None  # colocate derives rollout GPUs (16 -> 2 TP8 engines)
    update_weights_interval = 1  # sync: fresh weights every step

    # ── Engine sizing under colocation ───────────────────────────────────────
    # 2 engines double total capacity at the same per-engine pool.
    sglang_mem_fraction_static = 0.85

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _Slime()
