"""Kimi-K2.5 full-param smoke test — 8x H200, colocated, freeze most layers.

Run: EXPERIMENT_CONFIG=kimi_k25_fullparam_smoke modal run -d miles/modal_train.py::train
"""

from configs.base import ModalConfig, MilesConfig, DATA_PATH, CHECKPOINTS_PATH

_MILES_PR896_SHA = "6ff48d526f35278a6161f57957fc8cf6ab18c525"
_SGLANG_SYNC_SHA = "6d79c609954585c5e40d5f2b24dc5eb30d1fe41a"
_MEGATRON_BRIDGE_SHA = "d2ee05178d382414bec006fb94dc415483ec6cda"

modal = ModalConfig(
    gpu="H200",
    memory=(1024, int(2 * 1024 * 1024)),
    local_miles="/home/ec2-user/nan_wonderland/miles",
    patch_files=[
        "patches/sglang_lora_bias_22402.patch",
        "patches/megatron_bridge_kimi_vl.patch",
    ],
    image_run_commands=[
        # Remove pip nvidia-cudnn: TE loads via absolute paths, pip version has H200 symbol mismatch
        "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
        # Merge upstream SGLang commits (up to PR #22381)
        "cd /sgl-workspace/sglang && git config user.email 'miles@local' && git config user.name 'miles'",
        "cd /sgl-workspace/sglang && git fetch --unshallow origin && git fetch origin main",
        f"cd /sgl-workspace/sglang && git merge {_SGLANG_SYNC_SHA} --no-edit -X theirs",
        "cd /sgl-workspace/sglang && git apply /tmp/sglang_lora_bias_22402.patch",
        # K2.5 SGLang patches
        "python /root/miles/patches/sglang_fused_qkv_weight_fix.py",
        "python /root/miles/patches/sglang_pynccl_nonfatal.patch.py",
        # Install Megatron-Bridge 0.4.0 (has official KimiK2Bridge)
        f"uv pip install --system --no-deps --no-build-isolation git+https://github.com/radixark/Megatron-Bridge.git@{_MEGATRON_BRIDGE_SHA}",
        # Add KimiK25VL bridge from fzyzcjy/Megatron-Bridge PR #7 (handles language_model prefix + VL model)
        "cd $(python -c 'import megatron.bridge; import os; p=megatron.bridge.__file__; print(os.path.dirname(os.path.dirname(os.path.dirname(p))))') && patch -p1 --no-backup-if-mismatch < /tmp/megatron_bridge_kimi_vl.patch",
        # Bridges registered at runtime via megatron_utils/__init__.py
    ],
)


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/kimi-k2.5.sh"

    hf_checkpoint = f"{CHECKPOINTS_PATH}/Kimi-K2.5-bf16"
    save = f"{CHECKPOINTS_PATH}/Kimi-K2.5-fullparam-ckpt"
    save_interval = 1
    megatron_to_hf_mode = "bridge"

    only_train_params_name_list = ["layers\\.60\\."]

    actor_num_nodes = 8
    actor_num_gpus_per_node = 8
    colocate = True
    calculate_per_token_loss = True
    use_miles_router = True
    skip_eval_before_train = True

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"

    num_rollout = 2
    rollout_batch_size = 8
    n_samples_per_prompt = 4
    rollout_max_response_len = 4096
    rollout_temperature = 1
    global_batch_size = 32

    advantage_estimator = "grpo"
    kl_loss_coef = 0.00
    kl_loss_type = "low_var_kl"
    kl_coef = 0.00
    entropy_coef = 0.00
    eps_clip = 0.2
    eps_clip_high = 0.28

    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    train_backend = "megatron"
    tensor_model_parallel_size = 8
    sequence_parallel = True
    pipeline_model_parallel_size = 4
    context_parallel_size = 1
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 4  # (61 - 4) / 3 = 19 layers per middle stage

    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1

    use_dynamic_batch_size = True
    max_tokens_per_gpu = 4096

    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"

    moe_token_dispatcher_type = "alltoall"

    rollout_num_gpus_per_engine = 8
    sglang_mem_fraction_static = 0.7
    sglang_enable_dp_attention = True
    sglang_dp_size = 4
    sglang_moe_dense_tp_size = 1
    sglang_enable_dp_lm_head = True
    sglang_ep_size = 8
    sglang_disable_custom_all_reduce = True
    sglang_disable_cuda_graph = True
    sglang_watchdog_timeout = 1800
    sglang_server_concurrency = 512

    use_wandb = True
    wandb_project = "miles-kimi-k25"
    wandb_group = "kimi-k25-fullparam-smoke"
    disable_wandb_random_suffix = True

    def __init__(self):
        super().__init__()
        # Disable INT4 fake QAT for smoke test — saves memory on dequant buffers.
        # Re-enable for production training with INT4 weights.
        # self.environment.update({
        #     "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
        #     "OPEN_TRAINING_INT4_GROUP_SIZE": "32",
        # })

    def prepare_data(self) -> None:
        import json
        import os
        from huggingface_hub import snapshot_download

        os.makedirs(f"{DATA_PATH}/dapo-math-17k", exist_ok=True)
        snapshot_download(
            repo_id="zhuzilin/dapo-math-17k",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/dapo-math-17k",
        )

        # K2.5 config: keep quantization_config (compressed-tensors needed for memory)
        # The weight update path must handle compressed-tensors param names.


miles = _Miles()
