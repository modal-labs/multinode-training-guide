"""Qwen3.6-27B agentic RL on Snorkel dataset 1: plain GRPO, six nodes.

This recipe is self-contained: it inherits only ``SlimeConfig`` and spells out
every model, topology, rollout, optimization, checkpoint, and environment
parameter. It has no dependency on a SWE-rebench experiment config.

Data is split deterministically by repository, then reduced to a
language-balanced 300-task subset: 240 train and 60 eval. A repository never
appears on both sides. Set ``SNORKEL_TOTAL=full`` to use the complete split
instead. The old 500-task baseline is not comparable;
``skip_eval_before_train=False`` records a fresh step-0 anchor on the selected
holdout.

This is plain GRPO, not DAPO. Each selected prompt still gets eight trajectories,
but uniform-reward groups stay in the batch and naturally contribute zero
advantage. There is no dynamic filter, rejection, oversampling, or refill loop.

Every Snorkel task builds a Dockerfile-backed sandbox. All sandbox operations
pass through one RolloutManager Modal client, so concurrency is explicitly
capped at 16 per SGLang engine: 16 engines x 16 = 256 in-flight episodes.

Prepare data:

    EXPERIMENT_CONFIG=w_qwen3_6_27b_snorkel_1_noncolocate_5n \
        uv run --no-dev modal run slime/modal_train.py::download_data

Launch a fresh run:

    EXPERIMENT_CONFIG=w_qwen3_6_27b_snorkel_1_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train

Resume an earlier run by its complete tag:

    RESUME=qwen3.6-27b-snorkel-1-plain-300-noncolocate-5n-<stamp> \
    EXPERIMENT_CONFIG=w_qwen3_6_27b_snorkel_1_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

import os
from datetime import datetime

from configs.base import CHECKPOINTS_PATH, DATA_PATH, HF_CACHE_PATH, ModalConfig, SlimeConfig
from configs.datasets import pull
from configs.snorkel_1_data import (
    KEY,
    ROOT,
    SMALL_TOTAL,
    convert,
    eval_split_path,
    report,
    split,
    train_pool_path,
    unpack,
)

_LAUNCH_STAMP = os.environ.get("LAUNCH_STAMP") or f"{datetime.now():%Y%m%d-%H%M%S}"
_RESUME = os.environ.get("RESUME")
_TOTAL_RAW = os.environ.get("SNORKEL_TOTAL", str(SMALL_TOTAL))
if _TOTAL_RAW not in {str(SMALL_TOTAL), "full"}:
    raise ValueError(f"SNORKEL_TOTAL must be {SMALL_TOTAL!r} or 'full', got {_TOTAL_RAW!r}")
_DATASET_TOTAL = None if _TOTAL_RAW == "full" else SMALL_TOTAL
_SET_TAG = "full" if _DATASET_TOTAL is None else str(_DATASET_TOTAL)
_RUN_TAG = (
    f"{os.environ.get('WANDB_GROUP') or f'qwen3.6-27b-snorkel-1-plain-{_SET_TAG}-noncolocate-5n'}"
    f"-{_LAUNCH_STAMP}"
)

_TRAIN_POOL = train_pool_path(_DATASET_TOTAL)
_EVAL_SPLIT = eval_split_path(_DATASET_TOTAL)

_IMAGE_ENV = {
    key: value
    for key in (
        "WANDB_PROJECT",
        "WANDB_GROUP",
        "RESUME",
        "SNORKEL_BATCHES",
        "SNORKEL_TOTAL",
        "SNORKEL_UNPACK_WORKERS",
    )
    if (value := os.environ.get(key)) is not None
} | {"LAUNCH_STAMP": _LAUNCH_STAMP}

modal = ModalConfig(
    gpu="H200",
    memory=(1024, int(2 * 1024 * 1024)),
    ephemeral_disk=2 * 1024 * 1024,
    local_slime="/Users/shariqmobin/Documents/code/work/modal-projects/slime",
    image_run_commands=[
        f"rm -rf {HF_CACHE_PATH}",
        "apt-get update && apt-get install -y --no-install-recommends rdma-core libibverbs1 ibverbs-providers",
        "uv pip install --system modal mini-swe-agent datasets",
    ],
    image_env={"MSWEA_SILENT_STARTUP": "1", **_IMAGE_ENV},
)


def _eval_entry(name: str, path: str) -> dict:
    return {"name": name, "path": path, "metadata_overrides": {"eval_dataset": name}}


class _Slime(SlimeConfig):
    # ── Model ─────────────────────────────────────────────────────────────────
    slime_model_script = "scripts/models/qwen3.5-27B.sh"
    make_vocab_size_divisible_by = 32
    hf_checkpoint = "Qwen/Qwen3.6-27B"
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-27B_torch_dist"

    # ── Sync noncolocate topology: 2 train + 4 rollout nodes ─────────────────
    async_mode = False
    colocate = False
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    rollout_num_gpus = 32
    update_weights_interval = 1
    update_weight_buffer_size = 2147483648

    # ── Agentic rollout ───────────────────────────────────────────────────────
    custom_generate_function_path = "agentic_rl.generate.generate"
    custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"
    custom_config_path = {
        "agentic_max_steps": 75,
        "agentic_episode_timeout": 1800,
        "agentic_eval_timeout": 300,
        "agentic_exec_timeout": 120,
        "router_policy": "consistent_hashing",
    }
    metadata_key = "metadata"
    prompt_data = _TRAIN_POOL
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = False
    rollout_shuffle = True
    rm_type = None
    balance_data = True

    # ── Plain GRPO rollout sizing ─────────────────────────────────────────────
    num_rollout = 500
    rollout_batch_size = 32
    rollout_max_response_len = 8192
    rollout_temperature = 1.0
    n_samples_per_prompt = 8
    num_steps_per_rollout = 1
    global_batch_size = 256
    micro_batch_size = 1
    rollout_max_context_len = 32768 * 2
    sglang_reasoning_parser = "qwen3"
    sglang_tool_call_parser = "qwen3_coder"
    # Deliberately no dynamic_sampling_filter_path or over_sampling_batch_size.

    # ── SGLang: 16 TP2 engines, capped at 256 sandbox episodes ────────────────
    rollout_num_gpus_per_engine = 2
    sglang_server_concurrency = 16
    sglang_mem_fraction_static = 0.85
    sglang_cuda_graph_bs = [1, 2, 4, 8, 16] + list(range(24, 257, 8))
    sglang_mamba_scheduler_strategy = "extra_buffer"
    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 3
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 4
    sglang_enable_dp_attention = False
    sglang_disable_custom_all_reduce = False

    # ── Eval: the language-balanced, repo-disjoint 20% holdout ────────────────
    eval_interval = 5
    skip_eval_before_train = False
    eval_max_response_len = 8192
    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": [_eval_entry(f"snorkel1_{_SET_TAG}_repo20", _EVAL_SPLIT)],
    }

    # ── Dense 27B Megatron training: TP4 x CP2 x DP2 ─────────────────────────
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 2
    expert_model_parallel_size = 1
    expert_tensor_parallel_size = 1
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 32768
    log_probs_chunk_size = 1024
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"

    # ── Checkpointing and rollout artifacts ──────────────────────────────────
    save_debug_rollout_data = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    save = f"{CHECKPOINTS_PATH}/swe_ckpts/{_RESUME or _RUN_TAG}"
    save_interval = 5
    load = save

    # ── GRPO ──────────────────────────────────────────────────────────────────
    advantage_estimator = "grpo"
    use_kl_loss = False
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0
    eps_clip = 0.2
    eps_clip_high = 0.28

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = "adam"
    lr = 4e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    # ── Runtime environment ───────────────────────────────────────────────────
    environment = {
        "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "MODAL_ENVIRONMENT": "shariq-dev",
        "ASYNC_RL_TASK_ROOT": f"{DATA_PATH}",
        "SLIME_AGENT_SANDBOX_CPU": "2",
        "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
        "ASYNC_RL_REWARD_SHAPE": "binary",
    }

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = False
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_group = _RUN_TAG
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        """Materialize the full and 300-task repo-disjoint 80/20 splits."""
        pull(KEY)
        unpack(ROOT)
        convert(ROOT)
        split(ROOT)
        report(ROOT)


slime = _Slime()
