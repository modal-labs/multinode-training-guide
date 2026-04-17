"""GLM-4.7-355B-A32B GSPO on DAPO-Math-17k — 8-node colocated run.

Aligned to upstream run-glm4.7-355B-A32B.sh, adapted to this repo's config style.
Checkpoint conversion uses 4 nodes x 8 GPUs for tp=8, pp=4.
"""

from configs.base import ModalConfig, SlimeConfig, DATA_PATH, CHECKPOINTS_PATH

modal = ModalConfig(gpu="H200", memory=(1024, int(2 * 1024 * 1024)))  # 2 TiB in MiB


class _Slime(SlimeConfig):
    slime_model_script = "scripts/models/glm4.5-355B-A32B.sh"

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "zai-org/GLM-4.7"
    ref_load = f"{CHECKPOINTS_PATH}/GLM-4.7-355B-A32B_torch_dist"

    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 8
    actor_num_gpus_per_node = 8
    colocate = True

    # ── Data ──────────────────────────────────────────────────────────────────
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "deepscaler"
    # over_sampling_batch_size = 256
    # dynamic_sampling_filter_path = (
    #     "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    # )
    num_steps_per_rollout = 4
    balance_data = True
    rollout_stop_token_ids = [151329, 151336, 151338]
    skip_eval_before_train = True

    # ── Rollout ───────────────────────────────────────────────────────────────
    num_rollout = 3000
    rollout_batch_size = 64
    rollout_max_response_len = 8192
    rollout_temperature = 1.0
    n_samples_per_prompt = 8
    rollout_num_gpus_per_engine = 32
    sglang_mem_fraction_static = 0.7
    sglang_enable_dp_attention = True
    sglang_dp_size = 4
    sglang_ep_size = 32
    sglang_enable_dp_lm_head = True
    sglang_moe_dense_tp_size = 1
    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 3
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 4
    use_fault_tolerance = True

    # ── Eval ──────────────────────────────────────────────────────────────────
    eval_interval = 20
    eval_prompt_data = ["aime", f"{DATA_PATH}/aime-2024/aime-2024.jsonl"]
    n_samples_per_eval_prompt = 8
    eval_max_response_len = 8192
    eval_top_p = 1.0

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 8
    sequence_parallel = True
    pipeline_model_parallel_size = 4
    context_parallel_size = 2
    expert_model_parallel_size = 16
    expert_tensor_parallel_size = 1
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
    # moe_token_dispatcher_type = "flex"
    # moe_enable_deepep = True

    # ── Algorithm ─────────────────────────────────────────────────────────────
    advantage_estimator = "gspo"
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0
    eps_clip = 1e-4
    eps_clip_high = 2e-4
    use_tis = True

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
    wandb_group = "glm4.7-355b-a32b-dapo-math-8n"
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
