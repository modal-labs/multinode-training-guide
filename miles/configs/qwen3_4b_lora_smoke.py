"""Qwen3-4B LoRA GRPO smoke test — single node, 4x H200, colocated.

Reduced config for Modal verification: num_rollout=2, no WandB, no eval.
Corresponds to nan_wonderland/miles/examples/lora/run-qwen3-4b-megatron-lora-result.sh
"""

from configs.base import ModalConfig, MilesConfig, DATA_PATH

modal = ModalConfig(gpu="H200")


class _Miles(MilesConfig):
    # ── Model (architecture sourced from scripts/models/qwen3-4B.sh) ─────────
    miles_model_script = "scripts/models/qwen3-4B.sh"

    # ── Checkpoint & LoRA ─────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3-4B"
    save = "/checkpoints/Qwen3-4B-lora-ckpt"
    save_interval = 50
    lora_rank = 64
    lora_alpha = 32
    lora_dropout = 0.0
    target_modules = "all-linear"
    megatron_to_hf_mode = "bridge"

    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 1
    actor_num_gpus_per_node = 4
    colocate = True
    calculate_per_token_loss = True
    use_miles_router = True

    # ── Data ──────────────────────────────────────────────────────────────────
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"

    # ── Rollout (reduced for smoke test) ─────────────────────────────────────
    num_rollout = 2
    rollout_batch_size = 8
    n_samples_per_prompt = 8
    rollout_max_response_len = 4096
    rollout_temperature = 1
    global_batch_size = 64

    # ── Algorithm (GRPO) ──────────────────────────────────────────────────────
    advantage_estimator = "grpo"
    kl_loss_coef = 0.00
    kl_loss_type = "low_var_kl"
    kl_coef = 0.00
    entropy_coef = 0.00
    eps_clip = 0.2
    eps_clip_high = 0.28

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = "adam"
    lr = 2e-5
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    # ── Megatron ──────────────────────────────────────────────────────────────
    train_backend = "megatron"
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True

    # ── SGLang ────────────────────────────────────────────────────────────────
    rollout_num_gpus_per_engine = 1
    sglang_decode_log_interval = 1000
    sglang_mem_fraction_static = 0.4
    sglang_chunked_prefill_size = 4096

    # ── Environment (per-config, not base defaults) ───────────────────────────
    def __init__(self):
        super().__init__()
        self.environment.update({
            "NCCL_ALGO": "Ring",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        })

    def prepare_data(self) -> None:
        """Download dapo-math-17k and aime-2024 datasets."""
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="BytedTsinghua/dapo-math-17k",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/dapo-math-17k",
        )
        snapshot_download(
            repo_id="AI-MO/aime-2024",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/aime-2024",
        )


miles = _Miles()
