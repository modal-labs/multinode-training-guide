"""Qwen3-30B-A3B LoRA SFT — 2 nodes, train-only.

Run:
    EXPERIMENT_CONFIG=qwen3_30b_a3b_lora_sft modal run -d miles/modal_train.py::train

Supervised fine-tuning with Miles' SFT rollout, mirroring upstream
``scripts/run_qwen3_sft.py``: ``sft_rollout`` tokenizes chat messages with a
per-token loss mask, ``--debug-train-only`` skips SGLang entirely, and both
nodes run Megatron training. Bridge mode loads the HF checkpoint directly, so
no torch_dist conversion is needed.

Knobs to tune:
  - ``lora_rank`` / ``lora_alpha``: 32/32 default.
  - ``num_rollout``: optimizer steps (one global batch per rollout).
  - ``max_tokens_per_gpu``: raise for longer conversations.
"""

from configs.base import (
    ModalConfig,
    MilesConfig,
    DATA_PATH,
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
)

SFT_DATA_DIR = f"{DATA_PATH}/ultrachat-sft"
SFT_DATA_FILE = f"{SFT_DATA_DIR}/train.jsonl"
SFT_EXAMPLES = 2048
SFT_MAX_CHARS = 8000  # ~2k tokens; keeps every sample under max_tokens_per_gpu

modal = ModalConfig(
    docker_image="radixark/miles:dev-202609151226",
    gpu="H100",
    memory=(1024, int(2 * 1024 * 1024)),
    image_run_commands=[
        # Remove HF cache for modal volume mount.
        f"rm -rf {HF_CACHE_PATH} 2>/dev/null || true",
        # Remove pip nvidia-cudnn — TE loads system cuDNN via absolute paths and
        # the pip version has H200 symbol mismatches.
        "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
    ],
    # Ensure system libraries (cuDNN, NCCL) take precedence over pip versions.
    image_env={"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"},
)


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/qwen3-30B-A3B.py"
    # Upstream SFT runs through train_async.py, which requires colocate=False.
    async_mode = True

    environment = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    hf_checkpoint = "Qwen/Qwen3-30B-A3B"
    save = f"{CHECKPOINTS_PATH}/Qwen3-30B-A3B-lora-sft"
    save_interval = 20
    megatron_to_hf_mode = "bridge"

    # LoRA config (attention + dense/expert MLP linears)
    lora_rank = 32
    lora_alpha = 32
    lora_dropout = 0.0
    target_modules = "linear_qkv,linear_proj,linear_fc1,linear_fc2"

    # Infrastructure: train-only, so all 16 GPUs are Megatron actors.
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    colocate = False
    debug_train_only = True

    # SFT
    rollout_function_path = "miles.rollout.sft_rollout.generate_rollout"
    loss_type = "sft_loss"
    loss_mask_type = "qwen3"
    calculate_per_token_loss = True
    disable_compute_advantages_and_returns = True

    # Data (sft_rollout applies the chat template itself; no apply_chat_template)
    prompt_data = SFT_DATA_FILE
    input_key = "messages"
    rollout_shuffle = True
    num_rollout = 20
    rollout_batch_size = 32
    global_batch_size = 32

    # Parallelism: TP4 + SP, EP8 → DP4 over 16 GPUs
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 4096

    # Optimizer
    optimizer = "adam"
    lr = 1e-4
    lr_decay_style = "constant"
    weight_decay = 0.0
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    # Megatron
    train_backend = "megatron"
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"

    # WandB (optional; requires the wandb-secret Modal secret when enabled)
    use_wandb = False
    wandb_project = "miles-lora-sft"
    wandb_group = "qwen3-30b-a3b-lora-sft"

    def download_data(self) -> None:
        import json
        import os

        from datasets import load_dataset

        rows = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        os.makedirs(SFT_DATA_DIR, exist_ok=True)
        written = 0
        with open(SFT_DATA_FILE, "w") as f:
            for row in rows:
                messages = row["messages"]
                if sum(len(m["content"]) for m in messages) > SFT_MAX_CHARS:
                    continue
                f.write(json.dumps({"messages": messages}) + "\n")
                written += 1
                if written == SFT_EXAMPLES:
                    break
        print(f"Wrote {written} SFT conversations to {SFT_DATA_FILE}")


miles = _Miles()
