"""Qwen3.6-35B-A3B DAPO-Math GRPO — colocated, two nodes (2× H200:8).

Two-node twin of ``w_qwen3_6_swe_colocate_1n`` for model/infrastructure, but
with the DAPO-Math-17k data/reward/eval setup from ``qwen3_dapo``. BOTH nodes
time-share inference and training (sync, on-policy):
rollout runs on 2 TP8 engines (sgl-router load-balanced, twice the KV pool),
then the engines offload and Megatron trains across all 16 GPUs.
"""

from configs import w_qwen3_6_swe_colocate_1n as _base
from configs.base import CHECKPOINTS_PATH, DATA_PATH, run_tag

# Same image/overlay as the 1n config.
modal = _base.modal

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-dapo-math-colocate-2n")


class _Slime(_base._Slime):
    # ── Colocate / sync ──────────────────────────────────────────────────────
    # Only delta vs 1n: a second node. DP doubles (2 -> 4); TP/CP/EP/PP inherited.
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    # 16 GPUs / 8-per-engine -> 2 TP8 engines (sgl-router load-balanced).
    rollout_num_gpus_per_engine = 8

    sglang_mem_fraction_static = 0.65

    # ── Data ─────────────────────────────────────────────────────────────────
    custom_generate_function_path = None
    metadata_key = None
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "deepscaler"
    balance_data = True

    # ── Eval ─────────────────────────────────────────────────────────────────
    eval_interval = 20
    skip_eval_before_train = False
    eval_config = None
    eval_prompt_data = ["aime", f"{DATA_PATH}/aime-2024/aime-2024.jsonl"]
    n_samples_per_eval_prompt = 16
    eval_max_response_len = 32768
    eval_top_p = 1.0
    sglang_reasoning_parser = None
    sglang_tool_call_parser = None

    # Test
    sglang_speculative_algorithm = None
    num_rollout = 2

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/dapo_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    def download_data(self) -> None:
        """Download DAPO-Math-17k and AIME-2024 from HuggingFace to the data volume."""
        import os

        from huggingface_hub import snapshot_download

        os.makedirs(f"{DATA_PATH}/dapo-math-17k", exist_ok=True)
        os.makedirs(f"{DATA_PATH}/aime-2024", exist_ok=True)

        snapshot_download(
            repo_id="zhuzilin/dapo-math-17k",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/dapo-math-17k",
        )
        snapshot_download(
            repo_id="zhuzilin/aime-2024",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/aime-2024",
        )


slime = _Slime()
