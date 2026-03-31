"""GLM-4.7-Flash (30B-A3B MoE) GRPO on DAPO-Math-17k — single node (1×8 H100).

TP=1, PP=1, EP=8 → TP×EP=8, fits exactly on one node.
Checkpoint: convert with nproc=8 auto (TP=1) → GLM-4.7-Flash_torch_dist
See glm47_flash_dapo_multinode for the 2-node variant (run-glm4.7-30B-A3B.sh).
"""

from configs.base import ModalConfig, SlimeConfig, DATA_PATH, CHECKPOINTS_PATH

modal = ModalConfig(
    gpu="H200",
    image_run_commands=[
        "uv pip install --system 'transformers>=5.0'",
    ],
    # local_slime="/home/ec2-user/nan_wonderland/slime",
)


class _Slime(SlimeConfig):
    slime_model_script = "scripts/models/glm4.7-30B-A3B.sh"

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "zai-org/GLM-4.7-Flash"
    ref_load = f"{CHECKPOINTS_PATH}/GLM-4.7-Flash_torch_dist"

    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = True

    # ── Data ──────────────────────────────────────────────────────────────────
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "deepscaler"

    # ── Rollout ───────────────────────────────────────────────────────────────
    num_rollout = 3000
    rollout_batch_size = 128
    rollout_max_response_len = 32768
    rollout_temperature = 1.0
    n_samples_per_prompt = 8
    global_batch_size = 1024
    rollout_num_gpus_per_engine = 8
    sglang_mem_fraction_static = 0.7
    sglang_enable_dp_attention = True
    sglang_dp_size = 8
    sglang_enable_dp_lm_head = True
    sglang_moe_dense_tp_size = 1
    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 2
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 3
    sglang_max_running_requests = 512

    # ── Eval ──────────────────────────────────────────────────────────────────
    eval_interval = 20
    eval_prompt_data = ["aime24", f"{DATA_PATH}/aime-2024/aime-2024.jsonl"]
    n_samples_per_eval_prompt = 2
    eval_max_response_len = 32768
    eval_temperature = 1.0
    eval_top_p = 0.95

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 1
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 32768
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    moe_token_dispatcher_type = "flex"
    moe_enable_deepep = True

    # ── MTP ───────────────────────────────────────────────────────────────────
    mtp_num_layers = 1
    enable_mtp_training = True
    mtp_loss_scaling_factor = 0.2

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

    # ── Algorithm ─────────────────────────────────────────────────────────────
    advantage_estimator = "grpo"
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = "slime-grpo"
    wandb_group = "glm4.7-flash-dapo-math-1n"
    disable_wandb_random_suffix = True

    def prepare_data(self) -> None:
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
