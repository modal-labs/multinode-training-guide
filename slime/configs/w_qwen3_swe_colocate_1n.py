"""Qwen3-30B-A3B SWE agentic RL — Option A: colocated, single node (1× H200:8).

Same rollout as ``w_qwen3_swe_async`` but training and the engine time-share ONE
node: each step runs rollout, then the engine offloads and Megatron trains. Sync
on-policy. Vs the 2-node async parent: ~half the GPU cost and fully on-policy,
but step wall = rollout + train (serial) and a smaller KV pool
(mem_fraction_static 0.7 -> 0.55) makes prefix-cache health more important.
Checkpoint, data, agent env and algorithm inherited unchanged.
"""

from configs import w_qwen3_swe_async as _base
from configs.base import CHECKPOINTS_PATH, run_tag

# Same image/overlay as the parent.
modal = _base.modal

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3-30b-a3b-swe-gym-lite-colocate-1n")


class _Slime(_base._Slime):
    # ── Colocate / sync ──────────────────────────────────────────────────────
    async_mode = False  # train.py: rollout and train alternate on the same GPUs
    colocate = True
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    rollout_num_gpus = None  # colocate derives rollout GPUs from the actor allotment
    update_weights_interval = 1  # sync: fresh weights every step

    # ── Engine sizing under colocation ───────────────────────────────────────
    # Megatron residuals share the 141GB, so the static pool shrinks. If startup
    # OOMs during cuda-graph capture, drop toward 0.5 before touching the graph list.
    sglang_mem_fraction_static = 0.7

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _Slime()
