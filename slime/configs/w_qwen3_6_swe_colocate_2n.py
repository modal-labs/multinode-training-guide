"""Qwen3.6-35B-A3B SWE agentic RL — colocated, two nodes (2× H200:8).

Two-node twin of ``w_qwen3_6_swe_colocate_1n`` — same model, checkpoint, data,
agent env and algorithm; only the node count and the few knobs that scale with
it change. BOTH nodes time-share inference and training (sync, on-policy):
rollout runs on 2 TP8 engines (sgl-router load-balanced, twice the KV pool),
then the engines offload and Megatron trains across all 16 GPUs.
"""

from configs import w_qwen3_6_swe_colocate_1n as _base
from configs.base import CHECKPOINTS_PATH, run_tag

# Same image/overlay as the 1n config.
modal = _base.modal

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-gym-lite-colocate-2n")


class _Slime(_base._Slime):
    # ── Colocate / sync ──────────────────────────────────────────────────────
    # Only delta vs 1n: a second node. DP doubles (2 -> 4); TP/CP/EP/PP inherited.
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    # 16 GPUs / 8-per-engine -> 2 TP8 engines (sgl-router load-balanced).
    rollout_num_gpus_per_engine = 4

    sglang_mem_fraction_static = 0.65

    # Test
    sglang_speculative_algorithm = "EAGLE"
    num_rollout = 500
    sglang_enable_dp_attention = True
    sglang_dp_size = 4
    sglang_ep_size = 4
    sglang_enable_dp_lm_head = True

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _Slime()
