"""Kimi-K2.5 — 8 H200 nodes (8 GPUs each), colocated. NOT real full-param:
trains only the last 3 layers via ``only_train_params_name_list``.

Run:
    EXPERIMENT_CONFIG=kimi_k25 modal run -d miles/modal_train.py::train

To train the actual full-param Kimi-K2.5, you need **at least 32 nodes (256
H200s)** for the parallelism to close. Reference recipe (script + parallelism
config):
    https://gist.github.com/GeLee-Q/aa8a5336fa48c5934172aaa6e25ef5e7

To convert this config to real full-param:
  1. Remove / set ``only_train_params_name_list = None`` so all layers train.
  2. Bump ``actor_num_nodes`` to 32 and re-size the parallelism accordingly
     (``tensor_model_parallel_size``, ``pipeline_model_parallel_size``,
     ``context_parallel_size``, ``expert_model_parallel_size``) — see gist.

Knobs to tune (train/rollout throughput):
  - ``max_tokens_per_gpu`` (+ ``log_probs_max_tokens_per_gpu``)
  - ``rollout_max_response_len`` (increase if you see high truncated_ratio)
  - ``rollout_batch_size``
  - ``n_samples_per_prompt``
  - ``sglang_mem_fraction_static``
"""

from configs.base import ModalConfig, MilesConfig, DATA_PATH, CHECKPOINTS_PATH

SGLANG_SYNC_SHA = "b51b26377b72f8fb839d6d3b56ad2ead840bda95"
MEGATRON_BRIDGE_SHA = "3fd3768045422d0aa5c97e90a4e6c659aea9acb9"
MILES_SHA = "cc077c4e03cc806cfcafce61a0667a8aa6636777"

modal = ModalConfig(
    gpu="H200",
    memory=(1024, int(2 * 1024 * 1024)),
    patch_files=[
        "patches/megatron_bridge_kimi_vl.patch",
        "patches/miles_lora.patch",
        "patches/sglang_lora.patch",
    ],
    image_run_commands=[
        # Remove pip nvidia-cudnn — TE loads system cuDNN via absolute paths and
        # the pip version has H200 symbol mismatches.
        "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
        f"cd /sgl-workspace/sglang && git fetch origin sglang-miles && git checkout {SGLANG_SYNC_SHA}",
        "cd /sgl-workspace/sglang && git update-index --refresh && git apply --3way /tmp/sglang_lora.patch && if grep -R -n '^<<<<<<< ' .; then echo 'Patch failed to apply cleanly. Please resolve conflicts.' && exit 1; fi",
        f"uv pip install --system --no-deps --no-build-isolation git+https://github.com/radixark/Megatron-Bridge.git@{MEGATRON_BRIDGE_SHA}",
        "cd /usr/local/lib/python3.12/dist-packages && patch -p2 --no-backup-if-mismatch < /tmp/megatron_bridge_kimi_vl.patch",
        f"cd /root/miles && git fetch && git checkout {MILES_SHA} && git update-index --refresh && git apply --3way /tmp/miles_lora.patch && if grep -R -n '^<<<<<<< ' .; then echo 'Patch failed to apply cleanly. Please resolve conflicts.' && exit 1; fi",
    ],
)


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/kimi-k2-thinking.sh"

    environment = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NCCL_TIMEOUT": "3600",
        "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
        "OPEN_TRAINING_INT4_GROUP_SIZE": "32",
    }
    hf_checkpoint = "moonshotai/Kimi-K2.5"
    ref_load = f"{CHECKPOINTS_PATH}/Kimi-K2.5-bf16"
    megatron_to_hf_mode = "bridge"

    only_train_params_name_list = ["layers\\.58\\.", "layers\\.59\\.", "layers\\.60\\."]

    actor_num_nodes = 8
    actor_num_gpus_per_node = 8
    colocate = True
    calculate_per_token_loss = True
    use_miles_router = True
    skip_eval_before_train = True
    update_weight_buffer_size = 4 * 512 * 1024 * 1024

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"

    num_rollout = 5
    rollout_batch_size = 32
    n_samples_per_prompt = 8
    rollout_max_response_len = 16384
    rollout_temperature = 1
    sglang_cuda_graph_bs = [1, 2, 4, 8] + list(range(16, 129, 8))
    # dynamic_sampling_filter_path = (
    #     "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    # )
    global_batch_size = 256

    advantage_estimator = "grpo"
    kl_loss_coef = 0.00
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.00
    eps_clip = 0.2
    eps_clip_high = 0.28
    # use_kl_loss = True
    # use_tis = True

    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True
    use_distributed_optimizer = True

    train_backend = "megatron"
    tensor_model_parallel_size = 8
    sequence_parallel = True
    pipeline_model_parallel_size = 2
    context_parallel_size = 4
    expert_model_parallel_size = 32
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 30

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
    no_check_for_nan_in_loss_and_grad = True

    rollout_num_gpus_per_engine = 8
    sglang_mem_fraction_static = 0.7
    sglang_ep_size = 8
    sglang_server_concurrency = 1024

    use_wandb = True
    wandb_project = "miles-kimi-k25"
    wandb_group = "kimi-k25"
    disable_wandb_random_suffix = True

    def prepare_model(self) -> None:
        # Kimi-K2.5 currently needs this upstream source patch before use:
        # https://huggingface.co/moonshotai/Kimi-K2.5/discussions/91
        from pathlib import Path

        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(repo_id=self.hf_checkpoint))
        model_file = model_dir / "modeling_kimi_k25.py"
        src = model_file.read_text()

        ctor_old = """    def __init__(
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = 'spatial_temporal') -> None:
"""
        ctor_new = """    def __init__(
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = 'spatial_temporal',
        use_deterministic_attn: bool = False,
) -> None:
"""
        layer_old = """            MoonViTEncoderLayer(
                **block_cfg,
                use_deterministic_attn=self.use_deterministic_attn)
"""
        layer_new = """            MoonViTEncoderLayer(
                **block_cfg,
                use_deterministic_attn=use_deterministic_attn)
"""

        if "use_deterministic_attn: bool = False" in src:
            return

        if ctor_old not in src or layer_old not in src:
            raise RuntimeError(
                f"Unexpected {model_file} contents; Kimi-K2.5 patch from HF discussion #91 "
                "could not be applied cleanly."
            )

        src = src.replace(ctor_old, ctor_new, 1)
        src = src.replace(layer_old, layer_new, 1)
        model_file.write_text(src)

    def prepare_data(self) -> None:
        import os
        from huggingface_hub import snapshot_download

        os.makedirs(f"{DATA_PATH}/dapo-math-17k", exist_ok=True)
        snapshot_download(
            repo_id="zhuzilin/dapo-math-17k",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/dapo-math-17k",
        )


miles = _Miles()
