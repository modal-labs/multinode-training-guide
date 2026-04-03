"""Qwen3-VL-30B-A3B-Instruct GRPO on Geo3K using the standard Slime VLM path."""

from configs.base import ModalConfig, SlimeConfig, DATA_PATH

modal = ModalConfig(gpu="H200")


class _Slime(SlimeConfig):
    slime_model_script = "scripts/models/qwen3-30B-A3B.sh"
    environment = {
        **SlimeConfig.environment,
        "MODEL_ARGS_ROTARY_BASE": "5000000",
    }

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    load = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    megatron_to_hf_mode = "bridge"

    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = True

    # ── Data ──────────────────────────────────────────────────────────────────
    prompt_data = f"{DATA_PATH}/geo3k_imgurl/train.parquet"
    eval_prompt_data = ["geo3k_imgurl", f"{DATA_PATH}/geo3k_imgurl/test.parquet"]
    input_key = "problem"
    label_key = "answer"
    multimodal_keys = '{"image": "images"}'
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"

    # ── Rollout ───────────────────────────────────────────────────────────────
    num_rollout = 3000
    rollout_batch_size = 64
    n_samples_per_prompt = 8
    rollout_max_response_len = 4096
    rollout_temperature = 0.8
    rollout_num_gpus_per_engine = 8
    sglang_mem_fraction_static = 0.7
    sglang_cuda_graph_bs = [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
    ]
    sglang_ep_size = 8
    sglang_max_running_requests = 512
    global_batch_size = 512

    # ── Eval ──────────────────────────────────────────────────────────────────
    eval_interval = 20
    n_samples_per_eval_prompt = 1
    eval_max_response_len = 4096

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 2
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    qkv_format = "bshd"
    micro_batch_size = 1

    # ── Algorithm ─────────────────────────────────────────────────────────────
    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = "slime-grpo"
    wandb_group = "qwen3-vl-30b-a3b-instruct-megatron"
    disable_wandb_random_suffix = True

    def prepare_data(self) -> None:
        """Download the Geo3K VLM parquet dataset to the data volume."""
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="chenhegu/geo3k_imgurl",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/geo3k_imgurl",
        )


slime = _Slime()
