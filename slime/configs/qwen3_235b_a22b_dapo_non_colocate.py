"""Qwen3-235B-A22B GSPO on DAPO-Math-17k — 8-node non-colocated actor + rollout.

Aligned to the upstream slime run-qwen3-235B-A22B.sh launch settings, adapted
to this guide repo's Python config style.
"""

from configs.base import ModalConfig, SlimeConfig, DATA_PATH, CHECKPOINTS_PATH

modal = ModalConfig(gpu="H200")


class _Slime(SlimeConfig):
    slime_model_script = "scripts/models/qwen3-235B-A22B.sh"
    make_vocab_size_divisible_by = 32

    environment = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "MODEL_ARGS_ROTARY_BASE": "5000000",
    }

    # ── Model ─────────────────────────────────────────────────────────────────
    # Upstream GRPO uses an FP8 rollout checkpoint by default.
    hf_checkpoint = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3-235B-A22B-Instruct-2507_torch_dist_tp4pp4"

    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 4
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 32
    update_weight_buffer_size = 4 * 1024**3

    # ── Data ──────────────────────────────────────────────────────────────────
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "deepscaler"
    balance_data = True

    # ── Rollout ───────────────────────────────────────────────────────────────
    num_rollout = 3000
    rollout_batch_size = 8
    rollout_max_response_len = 8192
    rollout_temperature = 1.0
    n_samples_per_prompt = 8
    global_batch_size = 64
    rollout_num_gpus_per_engine = 32
    sglang_mem_fraction_static = 0.7
    sglang_cuda_graph_bs = [1, 2, 4, 8] + list(range(16, 257, 8))
    sglang_enable_dp_attention = True
    sglang_dp_size = 4
    sglang_ep_size = 32
    sglang_enable_dp_lm_head = True
    sglang_moe_a2a_backend = "deepep"
    sglang_deepep_mode = "auto"

    # ── Eval ──────────────────────────────────────────────────────────────────
    # Upstream leaves eval_interval disabled by default for this run.
    eval_prompt_data = ["aime", f"{DATA_PATH}/aime-2024/aime-2024.jsonl"]
    n_samples_per_eval_prompt = 16
    eval_max_response_len = 16384
    eval_top_p = 1.0

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 4
    context_parallel_size = 2
    expert_model_parallel_size = 16
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 22
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 16384
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"

    # ── Algorithm ─────────────────────────────────────────────────────────────
    advantage_estimator = "gspo"
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0
    eps_clip = 4e-4

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = "slime-grpo"
    wandb_group = "qwen3-235b-a22b-dapo-math-8n"
    disable_wandb_random_suffix = True

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
