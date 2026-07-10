"""GLM-5.2 (full 744B-A40B) LoRA GRPO on dapo-math — long-context DSA variant.

untested -- same recipe as glm5_2_744b_a40b_lora.py but on dapo-math task instead of gsm8k and increase 
rollout respone length to 4096 tokens and context window to 8192 tokens

    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo uv run modal run miles/modal_train.py::download_model
    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo uv run modal run miles/modal_train.py::download_data
    EXPERIMENT_CONFIG=glm5_2_744b_a40b_lora_dapo uv run modal run miles/modal_train.py::train

"""

from configs.base import ModalConfig, MilesConfig, DATA_PATH, CHECKPOINTS_PATH, HF_CACHE_PATH

modal = ModalConfig(
    docker_image="radixark/miles:dev-202607090055",
    gpu="H200",
    memory=(1024, int(2 * 1024 * 1024)),
    image_run_commands=[
        f"rm -rf {HF_CACHE_PATH} 2>/dev/null || true",
        "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
        "pip install --no-cache-dir hf_xet",
    ],
    image_env={
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
        "HF_XET_HIGH_PERFORMANCE": "1",
    },
)


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/glm5.2-744B-A40B_lora.sh"

    environment = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "SGLANG_NSA_FORCE_MLA": "1",
    }

    hf_checkpoint = "zai-org/GLM-5.2"
    megatron_to_hf_mode = "bridge"
    dsa_attention_backend = "tilelang"
    qkv_format = "thd"
    micro_batch_size = 1
    save = f"{CHECKPOINTS_PATH}/GLM-5.2-lora-dapo-ckpt"
    save_interval = 1

    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = True
    use_miles_router = True
    calculate_per_token_loss = True
    tensor_model_parallel_size = 8
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1

    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.0
    target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"
    experts_shared_outer_loras = True
    lora_base_cpu_backup = True
    no_gradient_accumulation_fusion = True

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"

    num_rollout = 1
    rollout_batch_size = 4
    n_samples_per_prompt = 4
    rollout_max_response_len = 4096
    seq_length = 8192
    rollout_max_context_len = 8192
    rollout_temperature = 1.0
    global_batch_size = 16
    use_rollout_routing_replay = True

    advantage_estimator = "grpo"
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0
    eps_clip = 0.2
    eps_clip_high = 0.28

    optimizer = "adam"
    lr = 1e-5
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"

    rollout_num_gpus_per_engine = 8
    sglang_mem_fraction_static = 0.5
    sglang_enable_dp_attention = True
    sglang_ep_size = 8
    sglang_dp_size = 8
    sglang_moe_dense_tp_size = 1
    sglang_enable_dp_lm_head = True
    sglang_attention_backend = "nsa"
    sglang_nsa_decode_backend = "flashmla_kv"
    sglang_nsa_prefill_backend = "flashmla_sparse"
    sglang_kv_cache_dtype = "fp8_e4m3"
    sglang_page_size = 64
    sglang_cuda_graph_max_bs = 256
    sglang_max_running_requests = 512
    sglang_chunked_prefill_size = 16384
    sglang_watchdog_timeout = 3600
    sglang_moe_runner_backend = "triton"
    sglang_disable_shared_experts_fusion = True
    sglang_max_lora_rank = 16
    sglang_lora_backend = "triton"

    use_wandb = True
    wandb_project = "miles-run_glm5_2_744b_a40b_lora"
    wandb_group = "glm5.2-744B-lora-dapo"
    disable_wandb_random_suffix = True

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(self.hf_checkpoint, max_workers=32)

    def download_data(self) -> None:
        import os

        from huggingface_hub import snapshot_download

        os.makedirs(f"{DATA_PATH}/dapo-math-17k", exist_ok=True)
        snapshot_download(
            repo_id="zhuzilin/dapo-math-17k",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/dapo-math-17k",
        )


miles = _Miles()
