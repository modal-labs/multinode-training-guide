"""Qwen3-4B LoRA GRPO — single node, colocated.

Run:
    EXPERIMENT_CONFIG=qwen3_4b_lora modal run -d miles/modal_train.py::train

Mirrors ``examples/lora/run-qwen3-4b-megatron-lora-result.sh``.

Knobs to tune:
  - ``lora_rank`` / ``lora_alpha``: 64/32 default.
  - ``lr``: 2e-5 default; PEFT tolerates ~10x full-param.
  - ``rollout_max_response_len`` / ``rollout_batch_size``.
"""

from configs.base import ModalConfig, MilesConfig, DATA_PATH, CHECKPOINTS_PATH, HF_CACHE_PATH

modal = ModalConfig(
    docker_image="radixark/miles:dev-202605291323",
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
    miles_model_script = "scripts/models/qwen3-4B.sh"

    environment = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    hf_checkpoint = "Qwen/Qwen3-4B"
    save = f"{CHECKPOINTS_PATH}/Qwen3-4B-lora-ckpt"
    save_interval = 50
    megatron_to_hf_mode = "bridge"

    # LoRA config
    lora_rank = 64
    lora_alpha = 32
    lora_dropout = 0.0
    target_modules = "all-linear"

    # Infrastructure
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = True
    calculate_per_token_loss = True
    use_miles_router = True

    # Data
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"

    # Rollout
    num_rollout = 100
    rollout_batch_size = 8
    n_samples_per_prompt = 8
    rollout_max_response_len = 4096
    rollout_temperature = 1
    global_batch_size = 64

    # Eval (effectively disabled — large interval + skip pre-train eval)
    eval_interval = 100000
    skip_eval_before_train = True
    eval_prompt_data = ["aime24", f"{DATA_PATH}/aime-2024/aime-2024.jsonl"]
    n_samples_per_eval_prompt = 16
    eval_max_response_len = 16384
    eval_top_p = 1

    # GRPO
    advantage_estimator = "grpo"
    kl_loss_coef = 0.00
    kl_loss_type = "low_var_kl"
    kl_coef = 0.00
    entropy_coef = 0.00
    eps_clip = 0.2
    eps_clip_high = 0.28

    # Optimizer
    optimizer = "adam"
    lr = 2e-5
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    # Megatron
    train_backend = "megatron"
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True

    # SGLang
    rollout_num_gpus_per_engine = 1
    sglang_decode_log_interval = 1000
    sglang_mem_fraction_static = 0.4
    sglang_chunked_prefill_size = 4096

    # WandB
    use_wandb = True
    wandb_project = "miles-lora-test"
    wandb_group = "qwen3-4b-megatron-lora-dapo-lr2e-5"
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
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


miles = _Miles()
