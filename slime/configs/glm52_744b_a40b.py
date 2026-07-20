"""GLM-5.2-744B-A40B GRPO on DAPO-Math-17k — 32-node H100 colocated run.

Aligned to upstream run-glm5.2-744B-A40B.sh, adapted for Modal.
The upstream script uses PD disaggregation with InfiniBand. In this environment
the available PD transfer paths are not usable: Mooncake TCP fails CUDA-buffer
copies, NIXL/LIBFABRIC needs unavailable EFA memory-registration privileges on
AWS, and NIXL/UCX only exposes loopback to the plugin. Use standard colocated
SGLang rollout engines instead of cross-node KV transfer, and leave cloud/EFA
placement open so Modal can find a 32-node H100 allocation.
"""

from configs.base import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    ModalConfig,
    SlimeConfig,
)

modal = ModalConfig(
    docker_image="slimerl/slime:nightly-dev-20260629a",
    gpu="H100",
    conversion_gpu="H200",
    cloud=None,
    efa_enabled=False,
    image_run_commands=[
        # Remove baked HF cache paths before Modal mounts the shared cache volume.
        f"rm -rf {HF_CACHE_PATH}",
        "test -f /root/slime/scripts/models/glm5.2-744B-A40B.sh",
    ],
    memory=(1024, int(2 * 1024 * 1024)),
)


class _Slime(SlimeConfig):
    slime_model_script = "scripts/models/glm5.2-744B-A40B.sh"

    environment = {
        "PYTHONPATH": "/root/slime:/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTHONUNBUFFERED": "1",
        "NCCL_DEBUG": "WARN",
        "GLOO_SOCKET_IFNAME": "overlay0",
        "TP_SOCKET_IFNAME": "overlay0",
        "NCCL_SOCKET_IFNAME": "=overlay0",
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "NCCL_OOB_NET_IFNAME": "overlay0",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_NET_GDR_LEVEL": "2",
        "NCCL_IB_QPS_PER_CONNECTION": "2",
        "NCCL_IB_TC": "160",
        "NCCL_IB_TIMEOUT": "22",
        "NCCL_PXN_DISABLE": "0",
        "NCCL_MIN_CTAS": "4",
        "NVTE_FWD_LAYERNORM_SM_MARGIN": "8",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "8",
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "MC_IB_PCI_RELAXED_ORDERING": "1",
        "MLP_SKIP_SORT_RDMA": "true",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "true",
        "SGLANG_QUANT_ALLOW_DOWNCASTING": "1",
        "NVSHMEM_DISABLE_NCCL": "1",
        "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": "=overlay0",
        "NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY": "AF_INET",
        "NVSHMEM_REMOTE_TRANSPORT": "libfabric",
        "NVSHMEM_LIBFABRIC_PROVIDER": "efa",
        "NVSHMEM_LIBFABRIC_SUPPORT": "1",
        "NVSHMEM_DISABLE_IB": "1",
        "NVSHMEM_DISABLE_IBRC": "1",
        "NVSHMEM_DISABLE_IBGDA": "1",
        "NVSHMEM_DISABLE_P2P": "1",
        "NVSHMEM_DISABLE_CUDA_VMM": "1",
        "NVSHMEM_USE_GDRCOPY": "0",
        "MODAL_DCP_BUFFERED_TORCH_SAVE": "1",
        "MODAL_DCP_THREAD_COUNT": "1",
        "PATCH_MEGATRON_ALLGATHER_CP_ROPE": "1",
    }

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "zai-org/GLM-5.2-FP8"
    load = hf_checkpoint
    ref_load = None
    save = None
    save_interval = None
    megatron_to_hf_mode = "bridge"

    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 32
    actor_num_gpus_per_node = 8
    colocate = True
    no_check_for_nan_in_loss_and_grad = True
    update_weight_buffer_size = 512 * 1024**2

    # ── Data ──────────────────────────────────────────────────────────────────
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "deepscaler"

    # ── Rollout ───────────────────────────────────────────────────────────────
    num_rollout = 3000
    rollout_batch_size = 8
    n_samples_per_prompt = 8
    rollout_max_response_len = 1024
    rollout_temperature = 1.0
    global_batch_size = 64

    # Standard colocated rollout: 16 engines × 16 GPUs, avoiding PD KV transfer.
    rollout_num_gpus_per_engine = 16
    # Leave room for the co-located Megatron-to-HF weight sync TP all-gathers.
    # Large individual tensors are sent zero-copy by the Modal runtime patch.
    sglang_mem_fraction_static = 0.86
    sglang_enable_dp_attention = True
    sglang_ep_size = 16
    sglang_dp_size = 16
    sglang_moe_dense_tp_size = 1
    sglang_enable_dp_lm_head = True
    sglang_moe_a2a_backend = "none"
    sglang_deepep_mode = None
    sglang_page_size = 64
    sglang_kv_cache_dtype = "fp8_e4m3"
    sglang_nsa_decode_backend = "flashmla_kv"
    sglang_nsa_prefill_backend = "flashmla_sparse"
    sglang_attention_backend = "nsa"
    sglang_cuda_graph_max_bs = 8
    sglang_disable_overlap_schedule = True
    sglang_max_running_requests = 512
    sglang_watchdog_timeout = 3600
    # Keep the initial Megatron-to-SGLang weight update under H100 memory
    # pressure; EAGLE's draft model adds several resident GB per GPU.
    sglang_speculative_algorithm = None
    sglang_speculative_num_steps = None
    sglang_speculative_eagle_topk = None
    sglang_speculative_num_draft_tokens = None
    sglang_speculative_draft_attention_backend = None

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 4
    conversion_tensor_model_parallel_size = 4
    conversion_pipeline_model_parallel_size = 8
    sequence_parallel = True
    pipeline_model_parallel_size = 8
    decoder_first_pipeline_num_layers = 14
    decoder_last_pipeline_num_layers = 16
    context_parallel_size = 8
    expert_model_parallel_size = 32
    expert_tensor_parallel_size = 1
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 4096
    # Upstream uses 1024 for throughput, but with colocated SGLang the actor
    # train step needs smaller packed-token microbatches to leave headroom for
    # MoE and fused cross-entropy backward scratch/autotune allocations.
    data_pad_size_multiplier = 64
    log_probs_chunk_size = 16384
    train_memory_margin_bytes = 0
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    moe_token_dispatcher_type = "alltoall"

    # ── Algorithm ─────────────────────────────────────────────────────────────
    advantage_estimator = "grpo"
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_tis = True
    tis_clip_low = 0.5
    tis_clip = 2.0

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
    wandb_group = "glm5.2-744b-a40b-dapo-math-32n"
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        """Download DAPO-Math-17k from Hugging Face to the data volume."""
        import os
        from huggingface_hub import snapshot_download

        os.makedirs(f"{DATA_PATH}/dapo-math-17k", exist_ok=True)
        snapshot_download(
            repo_id="zhuzilin/dapo-math-17k",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/dapo-math-17k",
        )


slime = _Slime()
