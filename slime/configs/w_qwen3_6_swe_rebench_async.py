"""Qwen3.6-35B-A3B (qwen3.5 arch MoE) — fully-async agentic SWE RL on SWE-rebench-V2.

Standalone config (inherits SlimeConfig directly). Analog of
``glm47_flash_swe_rebench_async``: identical fully-async pipeline (mini-swe agent
loop in Modal sandboxes via agentic_rl/, SWE-rebench native fresh-sandbox grading
through ``SweRebenchEnv``) and GRPO+TIS training, swapped to Qwen3.6's qwen3.5
architecture (hybrid gated-deltanet + full attention, 248k vocab) and the
dp-attention-OFF / small-TP2-engine rollout recipe the perf study found best for
Qwen3.6 SWE (see agentic_rl/profiles/swe_sglang_rollout_perf_profile). Non-colocated
on whfb H200: 2 actor nodes + 2 rollout nodes = 4 nodes.

Rows carry metadata.task_type="swerebench" → SweRebenchEnv (run one mini-swe episode,
capture the git diff, grade in a FRESH sandbox vs the held-out test patch). No task
tree / ASYNC_RL_TASK_ROOT needed.
"""
import os
from configs.base import ModalConfig, SlimeConfig, DATA_PATH, CHECKPOINTS_PATH, HF_CACHE_PATH
from configs.datasets import pull, train_path


_WANDB_IMAGE_ENV = {
    k: v for k in ("WANDB_PROJECT", "WANDB_GROUP") if (v := os.environ.get(k)) is not None
}
_TRAIN = "swe_rebench_v2"

# whfb H200 (141GB, RoCE fabric). Install the mlx5 verbs provider so NCCL can drive the
# RoCE NICs cross-node; ship slime + agentic_rl/ and the sandbox/agent deps into the image.
modal = ModalConfig(
    gpu="H200",
    # cloud="whfb",
    memory=(1024, int(2 * 1024 * 1024)),
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
    async_mode = True
    # /root/slime on PYTHONPATH so the Ray rollout workers can import agentic_rl.
    environment = {**SlimeConfig.environment, "PYTHONPATH": "/root/Megatron-LM/:/root/slime"}

    # ── Model / checkpoint ──────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3.6-35B-A3B"
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-35B-A3B_torch_dist"

    # ── Infrastructure (non-colocated: 2 actor nodes + 2 rollout nodes) ──────────
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 32  # 4 rollout nodes; 8× TP2 engines (dp-attention off)

    # ── Parallelism. num_query_groups=2 caps TP at 2; CP4 gives 16k tokens/rank at
    # 64k context (CP2 doubles that and risks OOM). EP8/ETP1 for the 256-expert MoE.
    # TP×CP=8 per replica → DP2 across the 16 training GPUs.
    tensor_model_parallel_size = 2
    pipeline_model_parallel_size = 1
    context_parallel_size = 4
    sequence_parallel = True
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    moe_token_dispatcher_type = "flex"
    moe_enable_deepep = True  # training-side expert all-to-all (rollout-side deepep is off)

    # ── Data: SWE-rebench-V2 python/pytest subset (built by download_data) ───────
    prompt_data = train_path(_TRAIN)
    input_key = "input"
    label_key = "label"  # instance_id, consumed by the grader
    apply_chat_template = False  # the agent harness builds its own conversation
    rm_type = "deepscaler"  # unused (reward comes from agentic_rl SweRebenchEnv), kept for a valid default

    # ── Rollout / sglang: dp-attention OFF + small TP2 engines (Qwen3.6 SWE recipe) ─
    rollout_function_path = "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
    custom_generate_function_path = "agentic_rl.generate.generate"  # mini-swe agent loop
    custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"  # agentic/* + async/*
    rollout_num_gpus_per_engine = 2  # TP2 → 16 // 2 = 8 engines (sgl-router load-balanced)
    sglang_server_concurrency = 128  # in-flight episodes per engine
    rollout_max_context_len = 65536  # served context per episode
    rollout_max_prompt_len = 16384  # problem-statement filter
    rollout_max_response_len = 4096  # per-turn generation cap
    rollout_temperature = 1.0
    rollout_shuffle = True
    sglang_reasoning_parser = "qwen3"  # strip <think> blocks
    sglang_tool_call_parser = "qwen3_coder"  # native bash tool-calls (model-matched parser)
    sglang_mem_fraction_static = 0.7
    # dp-attention OFF → pure TP; the single biggest Qwen3.6 rollout lever (unifies the
    # KV pool, drops per-token all-gather). Leave the DP/EP layout flags unset.
    sglang_enable_dp_attention = False
    sglang_disable_custom_all_reduce = False
    sglang_cuda_graph_bs = [1, 2, 4, 8, 16] + list(range(24, 257, 8))
    use_fault_tolerance = True
    # Gated-deltanet state pool; extra_buffer also radix-caches mamba states across
    # turns (prefix-cache health is the rollout bottleneck).
    sglang_mamba_scheduler_strategy = "extra_buffer"
    # EAGLE speculative decode off the MTP head.
    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 3
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 4

    # Agent episode limits (loaded onto `args` via setattr; read by agentic_rl.generate).
    custom_config_path = {
        "agentic_max_steps": 125,  # turns/episode; long tail hits this, context never truncates
        "agentic_episode_timeout": 1800,  # wall cap on the agent loop
        "agentic_exec_timeout": 120,  # per-command sandbox exec
        "agentic_grade_timeout": 1800,  # held-out test run
        "agentic_query_timeout": 600,  # per-turn sglang query; bounds hung/queued generations
        "agentic_max_boot_retries": 3,  # then ship a masked reward-0 sample (bad-image guard)
        "router_policy": "consistent_hashing",  # pin an episode's turns to one worker → prefix-cache hits
    }

    # ── Training ────────────────────────────────────────────────────────────────
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 16384  # per-rank token budget; 64k context / CP4
    log_probs_chunk_size = 8192  # chunk the logprob/entropy clone for headroom
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"

    # ── Optimizer ───────────────────────────────────────────────────────────────
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    # ── Algorithm: GRPO + TIS off-policy correction ─────────────────────────────
    advantage_estimator = "grpo"
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0
    eps_clip = 0.2
    eps_clip_high = 0.28  # asymmetric clip for stale samples
    use_tis = True  # clamp exp(actor_logp − rollout_logp) to [0, tis_clip]; without it staleness blows up grads
    tis_clip = 2.0
    balance_data = True  # even DP load despite wide trajectory-length variance
    update_weights_interval = 3

    # ── Sizing. global_batch_size = rollout_batch_size × n_samples_per_prompt → one
    # optimizer step per rollout collection. 16 prompts damps the sparse-reward gradient
    # variance (var ∝ 1/prompts); 8 samples/prompt for a strong GRPO group.
    num_rollout = 100
    rollout_batch_size = 16
    n_samples_per_prompt = 8
    global_batch_size = 128
    eval_interval = None  # no eval in fully-async
    skip_eval_before_train = True

    # ── WandB ───────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_group = os.environ.get("WANDB_GROUP")
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        pull(_TRAIN)


slime = _Slime()
