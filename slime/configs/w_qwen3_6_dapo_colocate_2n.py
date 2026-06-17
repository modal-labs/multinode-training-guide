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

_BOXED_ANSWER_INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: \\boxed{$Answer} where $Answer is the answer "
    "to the problem.\n\n"
)


def _add_boxed_answer_instruction(prompt):
    if isinstance(prompt, str):
        if prompt.startswith(_BOXED_ANSWER_INSTRUCTION):
            return prompt
        return f"{_BOXED_ANSWER_INSTRUCTION}{prompt}"

    if isinstance(prompt, list):
        updated = []
        added = False
        for message in prompt:
            if (
                not added
                and isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                message = dict(message)
                content = message["content"]
                if not content.startswith(_BOXED_ANSWER_INSTRUCTION):
                    message["content"] = f"{_BOXED_ANSWER_INSTRUCTION}{content}"
                added = True
            updated.append(message)
        if added:
            return updated

    raise TypeError(f"unsupported AIME prompt format: {type(prompt).__name__}")


class _Slime(_base._Slime):
    # ── Colocate / sync ──────────────────────────────────────────────────────
    # Only delta vs 1n: a second node. DP doubles (2 -> 4); TP/CP/EP/PP inherited.
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    rollout_num_gpus_per_engine = 4


    tensor_model_parallel_size = 2
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 2
    expert_model_parallel_size = 8

    rollout_batch_size = 16
    rollout_max_response_len = 16384
    global_batch_size = 128

    rollout_num_gpus_per_engine = 4
    sglang_mem_fraction_static = 0.65
    sglang_ep_size = 4
    sglang_enable_dp_attention = True
    sglang_dp_size = 4
    sglang_enable_dp_lm_head = True
    sglang_disable_custom_all_reduce = False

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
    eval_interval = 5
    skip_eval_before_train = True
    eval_config = None
    eval_prompt_data = ["aime", f"{DATA_PATH}/aime-2024/aime-2024-boxed.jsonl"]
    n_samples_per_eval_prompt = 16
    eval_max_response_len = 32768
    eval_top_p = 1.0
    sglang_reasoning_parser = None
    sglang_tool_call_parser = None

    # Test
    sglang_speculative_algorithm = None
    num_rollout = 2
    sglang_max_running_requests = 512
    max_tokens_per_gpu = 8192  

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    def download_data(self) -> None:
        """Download DAPO-Math-17k and AIME-2024 from HuggingFace to the data volume."""
        import json
        import os
        from pathlib import Path

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

        src = Path(f"{DATA_PATH}/aime-2024/aime-2024.jsonl")
        dst = Path(f"{DATA_PATH}/aime-2024/aime-2024-boxed.jsonl")
        rows = []
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["prompt"] = _add_boxed_answer_instruction(row["prompt"])
            rows.append(row)
        dst.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"wrote boxed AIME eval prompts -> {dst}")


slime = _Slime()
