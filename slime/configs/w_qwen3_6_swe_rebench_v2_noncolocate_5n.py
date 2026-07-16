"""Qwen3.6-35B-A3B SWE-rebench-V2 (Python) agentic RL — noncolocate, five nodes.

Scale-out sibling of ``w_qwen3_6_swe_rebench_v2_noncolocate_3n``: same data
(SWE-rebench-V2 Python subset), model, algorithm, and optimizer; the **topology**
grows the rollout fleet (2 → 4 rollout nodes, 8 → 16 engines) and this config
layers on an experimental SGLang serving recipe (dp-attention + HiCache) plus TIS.

Self-contained: inherits only ``SlimeConfig`` and spells every arg out inline (no
inheritance from another experiment config), so this recipe can be tuned in
isolation.

Topology: 1 training node (8 GPU, Megatron) + 4 rollout nodes (32 GPU, 16× TP2
SGLang engines behind sgl-router) = 5 nodes / 40 GPU. Sync noncolocate
(``async_mode=False``), matching the rebench lineage — set ``async_mode=True`` for
rollout/train overlap (the SWE-noncolocate base default).

Data: FULL harbor build of ``nebius/SWE-rebench-V2`` (Python subset, ~7.2k tasks,
no in-distribution holdout), published by
``agentic_rl.environment.convert2slime.swerebench``; eval is a transfer slice on
the published ``swegym_lite`` + ``swebench_verified`` held-out sets.

    EXPERIMENT_CONFIG=w_qwen3_6_swe_rebench_v2_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

import os

from configs.base import (
    ModalConfig,
    SlimeConfig,
    DATA_PATH,
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
    run_tag,
)
from configs.datasets import eval_datasets, pull, subsample, train_path

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-rebench-v2-noncolocate-5n")

# Datasets (one HF repo each; see configs/datasets.py). ALL swe_rebench_v2 Python
# tasks are training (no in-distribution holdout), so eval is a transfer slice on
# the published swegym_lite + swebench_verified held-out sets.
_TRAIN = "swe_rebench_v2"
_EVAL = [("swegym_lite", None), ("swebench_verified", None)]  # transfer/generalization eval; [] to disable

_WANDB_IMAGE_ENV = {
    k: v for k in ("WANDB_PROJECT", "WANDB_GROUP") if (v := os.environ.get(k)) is not None
}

modal = ModalConfig(
    gpu="H200",
    local_slime="/Users/junlin/Documents/Research/async-rl/slime",
    image_run_commands=[
        f"rm -rf {HF_CACHE_PATH}",
        "apt-get update && apt-get install -y --no-install-recommends rdma-core libibverbs1 ibverbs-providers",
        "uv pip install --system modal mini-swe-agent datasets",
    ],
    image_env={"MSWEA_SILENT_STARTUP": "1", **_WANDB_IMAGE_ENV},  # no mini-swe banner in rollout logs
)


class _Slime(SlimeConfig):
    # qwen3.5 architecture (hybrid gated-deltanet + full attention).
    slime_model_script = "scripts/models/qwen3.5-35B-A3B.sh"
    make_vocab_size_divisible_by = 32

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3.6-35B-A3B"
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-35B-A3B_torch_dist"
    # ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-35B-A3B_torch_dist_tp1"

    # ── Noncolocate, sync (rollout & training on separate nodes) ───────────────
    # async_mode=False = sync (rollout step, then a train step; no overlap), as in
    # the rebench lineage. Flip to True for fully-overlapped async on these GPUs.
    async_mode = False
    colocate = False
    actor_num_nodes = 1            # 1 training node (8 GPU, Megatron)
    actor_num_gpus_per_node = 8
    rollout_num_gpus = 32          # 4 rollout nodes → 32 // 2 = 16 TP2 engines
    update_weights_interval = 2    # resync weights every 2 rollout steps
    update_weight_buffer_size = 2147483648  # bucket the update like upstream CI

    # ── Custom agentic rollout (reward computed inline; no rm_type) ──────────
    custom_generate_function_path = "agentic_rl.generate.generate"
    custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"
    # Episode limits read off args (launcher materializes this dict to a temp YAML).
    custom_config_path = {
        "agentic_max_steps": 75,
        "agentic_episode_timeout": 1800,
        "agentic_eval_timeout": 300,
        "agentic_exec_timeout": 120,  # per-command sandbox exec
        "router_policy": "consistent_hashing",
    }
    metadata_key = "metadata"
    # Harbor build of SWE-rebench-V2, pulled from HF into /data/swe_rebench_v2/.
    prompt_data = train_path(_TRAIN)
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = False  # the adapter renders the chat template itself
    rollout_shuffle = True
    rm_type = None  # reward from the task env rollout (env/harbor.py), not a reward model
    balance_data = True

    # ── Rollout sizing ────────────────────────────────────────────────────────
    num_rollout = 500
    rollout_batch_size = 64
    rollout_max_response_len = 8192
    rollout_temperature = 1.0
    n_samples_per_prompt = 16
    num_steps_per_rollout = 1
    global_batch_size = 1024  # rollout_batch_size * n_samples_per_prompt // steps
    micro_batch_size = 1
    rollout_max_context_len = 32768 * 2  # 64k multi-turn prompt+response budget
    sglang_reasoning_parser = "qwen3"  # strip <think> blocks
    # mini-swe-agent v2 needs the model-matched parser for native tool-calls.
    sglang_tool_call_parser = "qwen3_coder"

    # ── Rollout engines: 4 rollout nodes, 16× TP2 behind sgl-router ───────────
    rollout_num_gpus_per_engine = 2   # TP2 → 32 // 2 = 16 engines (sgl-router load-balanced)
    sglang_mem_fraction_static = 0.85
    sglang_cuda_graph_bs = [1, 2, 4, 8, 16] + list(range(24, 257, 8))
    sglang_cuda_graph_max_bs = 64
    sglang_max_running_requests = 512
    use_fault_tolerance = True

    # Required for gated-deltanet; extra_buffer also radix-caches mamba states
    # across turns (prefix-cache health is the multi-turn bottleneck).
    sglang_mamba_scheduler_strategy = "extra_buffer"

    # EAGLE speculative decoding off the MTP head (decode-latency win); disable
    # this block first if the engine looks off.
    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 3
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 4

    # ── HiCache: host-memory extension of the radix prefix cache ──────────────
    sglang_enable_hierarchical_cache = True
    sglang_hicache_ratio = 1.0
    sglang_hicache_write_policy = "write_through"
    sglang_page_size = 64  # HiCache transfers are page-granular

    # ── dp-attention ON (experimental) ─────────────────────────────────────────
    # NB: this REVERSES the rollout-perf study's recommendation (see
    # swe_sglang_rollout_perf_profile / distilled.md F4–F9, where dp-attention OFF
    # was the single biggest latency lever). Kept as the 5n experiment; set
    # sglang_enable_dp_attention=False and drop the dp_size / dp_lm_head /
    # moe_dense_tp flags to return to the profiled pure-TP2 recipe.
    sglang_enable_dp_attention = True
    sglang_disable_custom_all_reduce = False
    sglang_dp_size = 1
    sglang_enable_dp_lm_head = True
    sglang_moe_dense_tp_size = 1

    # ── Eval ──────────────────────────────────────────────────────────────────
    # Each pass blocks the train loop on the shared engines, so keep subsets small
    # (full sweeps: w_qwen3_swe_eval). Only eval_interval=None is "off".
    eval_interval = 5
    skip_eval_before_train = True

    eval_max_response_len = 8192
    eval_config = {
        "defaults": {
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.6,  # low-but-nonzero: Qwen3 degenerates at greedy
            "top_p": 1.0,
        },
        # Built from _EVAL (key, n) specs; n subsamples eval.jsonl in download_data.
        "datasets": eval_datasets(_EVAL),
    }

    # ── Training ──────────────────────────────────────────────────────────────
    # num_query_groups=2 caps TP at 2; EP=8 divides the 256 experts; CP=2 matches
    # the upstream 35B CI test.
    tensor_model_parallel_size = 2
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    # CP shards the sequence; max_tokens_per_gpu must be >= the longest sample's
    # per-CP-rank token count or dynamic batching can't place it.
    context_parallel_size = 2
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    # MoE dispatch: flex + DeepEP (config args win over the model script's alltoall).
    moe_token_dispatcher_type = "flex"
    moe_enable_deepep = True  # training-side expert all-to-all (rollout-side deepep is off)
    use_dynamic_batch_size = True
    # 64k rollout_max_context_len / CP2 = 32k per rank; this budget covers the
    # longest sample exactly.
    max_tokens_per_gpu = 16384 * 2
    log_probs_chunk_size = 1024
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    # mtp_num_layers = 1
    # enable_mtp_training = True
    # mtp_loss_scaling_factor = 0.2
    # Dump every rollout under the W&B group subdir; relaunches overwrite.
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    # ── Algorithm ─────────────────────────────────────────────────────────────
    advantage_estimator = "grpo"
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0
    eps_clip = 0.2
    eps_clip_high = 0.28
    # Truncated importance sampling: corrects for the (up to
    # update_weights_interval-step) staleness between rollout and train weights.
    use_tis = True
    tis_clip = 2.0
    tis_clip_low = 0.5

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

    # ── Environment: PYTHONPATH + Modal sandbox knobs reach the Ray workers ──
    # In-process design: no adapter/tunnel/in-sandbox runner. Episode limits live
    # in custom_config_path; only sandbox/runtime knobs belong here.
    environment = {
        "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "MODAL_ENVIRONMENT": "junlin-dev",  # env the agent sandboxes boot in
        # harbor rows resolve relative task_path here; evalset.py uses the same root.
        "ASYNC_RL_TASK_ROOT": f"{DATA_PATH}",
        "SLIME_AGENT_SANDBOX_CPU": "2",
        "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
        "ASYNC_RL_REWARD_SHAPE": "binary",
    }

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_group = _RUN_TAG
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        """Pull the SWE-rebench-V2 repo into /data/swe_rebench_v2/ and subsample
        the eval slice. Conversion/publish is offline (convert2slime/swerebench.py)."""
        for key in {_TRAIN, *(k for k, _ in _EVAL)}:
            pull(key)  # whole repo (train + eval + tasks) into /data/<key>/
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)


slime = _Slime()
