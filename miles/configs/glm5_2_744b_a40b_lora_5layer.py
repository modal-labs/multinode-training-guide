"""GLM-5.2 (744B-A40B arch, 5-layer prune) LoRA GRPO — single node, colocated.

Smoke test for the GLM-5.2 bridge-mode DSA LoRA path. ``Pinaster/GLM-5.2_5layer``
is a 5-layer prune (3 dense + 2 MoE) of GLM-5.2 that keeps one computing + one
skip layer, so it exercises the same DSA cross-layer index-sharing, MoE, bridge
LoRA, and sglang MoE-LoRA serving path as the full 744B model at toy cost.

Ports ``scripts/run_glm5_2_744b_a40b_lora.py`` (which the guide launcher does NOT
run directly) into config attributes: the model ``.sh`` supplies architecture
only, every LoRA/DSA/sglang flag is set here and forwarded by ``cli_args()``.

Requires a miles image built after PR #1559 (GLM-5/5.1/5.2 LoRA) and PR #1593
(bridge-LoRA recompute fix); the repo default dev-202605291323 predates both.

Launched by the dedicated smoke harness (pinned to this config):
    uv run modal run miles/modal_train_glm_test.py::download_model
    uv run modal run miles/modal_train_glm_test.py::download_data
    uv run modal run miles/modal_train_glm_test.py::train
"""

from configs.base import ModalConfig, MilesConfig, DATA_PATH, CHECKPOINTS_PATH, HF_CACHE_PATH

modal = ModalConfig(
    docker_image="radixark/miles:dev-202607090055",
    gpu="H200",
    memory=(1024, int(2 * 1024 * 1024)),
    image_run_commands=[
        f"rm -rf {HF_CACHE_PATH} 2>/dev/null || true",
        "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
    ],
    image_env={"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"},
)


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/glm5.2-744B-A40B_5layer_lora.sh"

    environment = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "SGLANG_NSA_FORCE_MLA": "1",
    }

    hf_checkpoint = "Pinaster/GLM-5.2_5layer"
    megatron_to_hf_mode = "bridge"
    dsa_attention_backend = "tilelang"
    qkv_format = "thd"
    micro_batch_size = 1
    save = f"{CHECKPOINTS_PATH}/GLM-5.2_5layer-lora-ckpt"
    save_interval = 1

    actor_num_nodes = 1
    actor_num_gpus_per_node = 4
    colocate = True
    use_miles_router = True
    calculate_per_token_loss = True
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    expert_model_parallel_size = 4
    expert_tensor_parallel_size = 1

    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.0
    target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"
    experts_shared_outer_loras = True
    lora_base_cpu_backup = True
    no_gradient_accumulation_fusion = True

    prompt_data = f"{DATA_PATH}/gsm8k/train.parquet"
    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"

    num_rollout = 1
    rollout_batch_size = 4
    n_samples_per_prompt = 4
    rollout_max_response_len = 512
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

    rollout_num_gpus_per_engine = 2
    sglang_mem_fraction_static = 0.5
    sglang_enable_dp_attention = True
    sglang_ep_size = 2
    sglang_dp_size = 2
    sglang_moe_dense_tp_size = 1
    sglang_enable_dp_lm_head = True
    sglang_attention_backend = "nsa"
    sglang_nsa_decode_backend = "flashmla_sparse"
    sglang_nsa_prefill_backend = "flashmla_sparse"
    sglang_page_size = 64
    sglang_cuda_graph_max_bs = 64
    sglang_max_running_requests = 512
    sglang_chunked_prefill_size = 4096
    sglang_watchdog_timeout = 3600
    sglang_moe_runner_backend = "triton"
    sglang_disable_shared_experts_fusion = True
    sglang_max_lora_rank = 16
    sglang_lora_backend = "triton"

    use_wandb = True
    wandb_project = "miles-run_glm5_2_744b_a40b_lora"
    wandb_group = "glm5.2-5layer-lora"
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        import os

        from huggingface_hub import snapshot_download

        os.makedirs(f"{DATA_PATH}/gsm8k", exist_ok=True)
        snapshot_download(
            repo_id="zhuzilin/gsm8k",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/gsm8k",
        )


miles = _Miles()
