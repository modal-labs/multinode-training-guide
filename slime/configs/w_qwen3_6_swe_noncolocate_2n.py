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
    async_mode = True
    update_weights_interval = 2
    colocate = False
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8

    # ── Rollout sizing ────────────────────────────────────────────────────────
    rollout_max_response_len = 8192
    rollout_temperature = 1.0
    n_samples_per_prompt = 8
    num_steps_per_rollout = 1
    global_batch_size = 256  # rollout_batch_size * n_samples_per_prompt // steps
    micro_batch_size = 1
    rollout_max_context_len = 32768 * 2
    sglang_reasoning_parser = "qwen3"  # strip <think> blocks
    sglang_tool_call_parser = "qwen3_coder"
    rollout_num_gpus_per_engine = 8
    rollout_num_gpus = 8

    # ── Engine sizing under colocation ────────────────────────────────────────
    sglang_mem_fraction_static = 0.7
    sglang_speculative_algorithm = "EAGLE"

    sglang_enable_dp_attention = True
    sglang_dp_size = 8
    sglang_ep_size = 8
    sglang_enable_dp_lm_head = True
    sglang_disable_custom_all_reduce = False


    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _Slime()
