"""GLM-4.7-Flash (30B-A3B MoE) — fully-async agentic SWE RL on SWE-rebench-V2.

Standalone config (inherits SlimeConfig directly). Rollout = mini-swe agent loop in
Modal sandboxes (agentic_rl/, shipped via local_slime, wired through slime's custom-*
hooks — no slime-core edits); training = Megatron GRPO. Non-colocated on whfb H200:
2 actor nodes + 8 rollout engines = 10 nodes. Checkpoint is converted for TP2/PP2.

Rows carry metadata.task_type="swerebench" → agentic_rl.environment.swerebench.SweRebenchEnv
(SWE-rebench native: run one mini-swe episode, capture the git diff, grade in a FRESH
sandbox against the held-out test patch). No task tree / ASYNC_RL_TASK_ROOT needed.
"""
import os
from configs.base import ModalConfig, SlimeConfig, DATA_PATH, CHECKPOINTS_PATH, HF_CACHE_PATH

_WANDB_IMAGE_ENV = {
    k: v for k in ("WANDB_PROJECT", "WANDB_GROUP") if (v := os.environ.get(k)) is not None
}
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
    slime_model_script = "scripts/models/glm4.7-30B-A3B.sh"
    async_mode = True
    # /root/slime on PYTHONPATH so the Ray rollout workers can import agentic_rl.
    environment = {**SlimeConfig.environment, "PYTHONPATH": "/root/Megatron-LM/:/root/slime"}

    # ── Model / checkpoint ──────────────────────────────────────────────────────
    hf_checkpoint = "zai-org/GLM-4.7-Flash"
    ref_load = f"{CHECKPOINTS_PATH}/GLM-4.7-Flash_torch_dist_tp2pp2"

    # ── Infrastructure (non-colocated: 2 actor nodes + 4 rollout engines) ────────
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 32  # 4 sglang engines × 8 GPU

    # ── Parallelism. TP2/PP2 are checkpoint-fixed; EP8/ETP1 for the MoE. CP4 gives 16k
    # tokens/rank at 64k context (CP2 OOMs at 80GB). TP×PP×CP=16 = 2 nodes → DP=1.
    tensor_model_parallel_size = 2
    pipeline_model_parallel_size = 2
    context_parallel_size = 4
    sequence_parallel = True
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 23  # GLM-4.7-Flash layer split for PP2

    # ── Data: SWE-rebench-V2 python/pytest subset (built by download_data) ───────
    prompt_data = f"{DATA_PATH}/swe_rebench_v2_python/train.jsonl"
    input_key = "input"
    label_key = "label"  # instance_id, consumed by the grader
    apply_chat_template = False  # the agent harness builds its own conversation
    rm_type = "deepscaler"  # unused (reward comes from agentic_rl SweRebenchEnv), kept for a valid default

    # ── Rollout / sglang ────────────────────────────────────────────────────────
    rollout_function_path = "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
    custom_generate_function_path = "agentic_rl.generate.generate"  # mini-swe agent loop
    custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"  # agentic/* + async/*
    # Native tool-call + reasoning parsers. The mini-swe harness parses tool calls
    # client-side via these (agentic_rl.model.RecordingModel); without them every turn
    # rejects with "no tool call" → rollback → 0 trained tokens. GLM-4.7-Flash emits
    # the COMPACT (no-newline) form
    #   <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call> and <think>…
    # The "glm45" tool parser's regex REQUIRES newlines around the args, so it sees the
    # <tool_call> token (has_tool_call=True) but extracts ZERO calls → "No tool calls found"
    # on every turn (100% turns=0, verified offline against sglang 0.5.12.post1). The
    # dedicated "glm47" detector (Glm47MoeDetector: "for GLM-4.7 and GLM-5") parses the
    # compact form. Reasoning is unchanged (<think></think>), so keep "glm45" there —
    # there is no "glm47" reasoning parser.
    sglang_reasoning_parser = "glm45"  # strip <think> blocks (no glm47 variant)
    sglang_tool_call_parser = "glm47"  # GLM-4.7/5 compact <tool_call> form (glm45 needs newlines)
    rollout_num_gpus_per_engine = 8
    sglang_server_concurrency = 128  # in-flight episodes per engine
    rollout_max_context_len = 65536  # served context per episode
    rollout_max_prompt_len = 16384  # problem-statement filter
    rollout_max_response_len = 4096  # per-turn generation cap
    rollout_temperature = 1.0
    rollout_shuffle = True
    sglang_mem_fraction_static = 0.7  # headroom for HiCache staging + EAGLE draft (0.85 OOMs)
    sglang_enable_dp_attention = True
    sglang_dp_size = 8
    sglang_enable_dp_lm_head = True
    sglang_moe_dense_tp_size = 1
    sglang_cuda_graph_max_bs = 64
    sglang_max_running_requests = 512
    use_fault_tolerance = True
    # EAGLE speculative decode (~2.3 accept len).
    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 3
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 4
    # HiCache: tier the radix prefix cache GPU→host so idle multi-turn prefixes survive
    # eviction (keeps ~0.8 prefix-hit under load). ratio=1.0: host pool = GPU pool — 3.0
    # host-OOMs on H200 (its GPU KV pool is ~2× H100's).
    sglang_enable_hierarchical_cache = True
    sglang_hicache_ratio = 1.0
    sglang_hicache_write_policy = "write_through"
    sglang_page_size = 64  # HiCache transfers are page-granular

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
    max_tokens_per_gpu = 16384  # per-rank token budget; 32768 OOMs at 64k/CP4
    log_probs_chunk_size = 8192  # chunk the logprob/entropy clone for headroom
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    moe_token_dispatcher_type = "flex"
    moe_enable_deepep = True  # training-side expert all-to-all (rollout-side deepep is off)
    mtp_num_layers = 1
    enable_mtp_training = True
    mtp_loss_scaling_factor = 0.2

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
        """SWE-rebench-V2 → prompt_data jsonl, filtered to gradable python/pytest tasks:
        require ≥1 failing test (empty FAIL_TO_PASS would auto-resolve via all([])) and drop
        3 private images. ~7,240 instances. metadata.task_type pins the SweRebenchEnv."""
        import json
        import os

        from datasets import load_dataset

        BAD = {"dhi__mikeio-227", "dhi__mikeio-442", "dhi__mikeio-690"}  # private images
        out = f"{DATA_PATH}/swe_rebench_v2_python/train.jsonl"
        keys = ("instance_id", "repo", "base_commit", "image_name", "problem_statement",
                "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "install_config", "language")
        os.makedirs(os.path.dirname(out), exist_ok=True)

        def keep(r) -> bool:
            if r["language"] != "python" or r["install_config"]["log_parser"] != "parse_log_pytest":
                return False
            if r["instance_id"] in BAD:
                return False
            return len(r["FAIL_TO_PASS"]) >= 1

        ds = load_dataset("nebius/SWE-rebench-V2", split="train").filter(keep)
        with open(out, "w") as f:
            for r in ds:
                row = {
                    "input": r["problem_statement"],
                    "label": r["instance_id"],
                    "metadata": {"task_type": "swerebench", **{k: r[k] for k in keys}},
                }
                f.write(json.dumps(row) + "\n")
        print(f"download_data: wrote {len(ds)} filtered instances to {out}")


slime = _Slime()
