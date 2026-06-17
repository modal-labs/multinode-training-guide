"""Qwen3-30B-A3B SWE agentic RL on SWE-Gym-Lite — async, non-colocated (2× H200:8).

Self-contained (model/optimizer/parallelism copied from ``w_qwen3_dapo``).
Rollout is the mini-swe-agent driver in ``async_rl_research`` (Modal sandboxes,
reward graded in a clean sandbox), wired via ``--custom-generate-function-path``.
The agent dials back to the host adapter through a ``modal.forward`` tunnel
(``MODAL_EXPOSE_ADAPTER=1``); ``async_rl_research`` ships via the ``local_slime``
overlay and is added to ``PYTHONPATH`` so Ray rollout workers can import it.
Checkpoint: reuse ``w_qwen3_dapo``'s ``Qwen3-30B-A3B_torch_dist`` (TP=4, PP=1).
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

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3-30b-a3b-swe-gym-lite-async")

_WANDB_IMAGE_ENV = {
    k: v for k in ("WANDB_PROJECT", "WANDB_GROUP") if (v := os.environ.get(k)) is not None
}

modal = ModalConfig(
    gpu="H200",
    local_slime="/Users/junlin/Documents/Research/async-rl/slime",
    image_env=_WANDB_IMAGE_ENV,
    # Install modal into site-packages (Ray workers override PYTHONPATH); wipe
    # HF_CACHE_PATH so the volume can mount onto an empty path.
    image_run_commands=["pip install modal", f"rm -rf {HF_CACHE_PATH}"],
)


class _Slime(SlimeConfig):
    slime_model_script = "scripts/models/qwen3-30B-A3B.sh"
    make_vocab_size_divisible_by = 32

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3-30B-A3B"
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3-30B-A3B_torch_dist"

    # ── Async / non-colocate ────────────────────────────────────────────────
    # SGLang serves continuously during multi-turn dial-backs, so rollout GPUs
    # are separate from training.
    async_mode = True
    update_weights_interval = 3
    colocate = False
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    rollout_num_gpus = 8  # → total_nodes() == 2 (8 actor + 8 rollout)

    # ── Custom agentic rollout (reward computed inline; no rm_type) ──────────
    custom_generate_function_path = "async_rl_research.generate.generate"
    metadata_key = "metadata"
    prompt_data = f"{DATA_PATH}/swe_gym_lite/swe_gym_lite.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = False  # the adapter renders the chat template itself
    rollout_shuffle = True
    rm_type = None  # reward from the task env rollout (env/swe_gym.py), not a reward model
    balance_data = True

    # ── Rollout sizing (start small to validate the pipeline) ────────────────
    num_rollout = 3000
    rollout_batch_size = 32
    rollout_max_response_len = 8192  # caps ONE model turn
    rollout_temperature = 1.0
    n_samples_per_prompt = 8
    num_steps_per_rollout = 1
    global_batch_size = 256  # rollout_batch_size * n_samples_per_prompt // steps
    micro_batch_size = 1
    rollout_max_context_len = 32768  # multi-turn prompt+response budget
    sglang_reasoning_parser = "qwen3"  # strip <think> blocks
    # mini-swe-agent v2 needs the model-matched parser for native tool-calls
    # (Qwen3 emits hermes-style <tool_call> JSON -> "qwen25").
    sglang_tool_call_parser = "qwen25"
    rollout_num_gpus_per_engine = 8
    sglang_mem_fraction_static = 0.85
    sglang_cuda_graph_bs = [1, 2, 4, 8, 16] + list(range(24, 257, 8))

    sglang_enable_dp_attention = False
    # sglang_dp_size = 8
    sglang_ep_size = 8

    sglang_disable_custom_all_reduce = True

    # ── Eval ──────────────────────────────────────────────────────────────────
    # Subsets built with `python -m async_rl_research.evalset`. No step-0
    # baseline in async mode (baseline once with w_qwen3_swe_eval); each pass
    # blocks the train loop on the shared engines, so keep subsets small. Only
    # eval_interval=None is "off".
    eval_interval = None  # flip on (e.g. 20) once /data/evalsets/v0 is built
    eval_max_response_len = 16384
    eval_config = {
        "defaults": {
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.6,  # low-but-nonzero: Qwen3 degenerates at greedy
            "top_p": 1.0,
        },
        # metadata_overrides keeps per-dataset attribution in the flattened dump.
        "datasets": [
            {
                "name": "usaco_50",
                "path": f"{DATA_PATH}/evalsets/v0/usaco_50.jsonl",
                "metadata_overrides": {"eval_dataset": "usaco_50"},
            },
            # {
            #     "name": "swebench_verified_50",
            #     "path": f"{DATA_PATH}/evalsets/v0/swebench_verified_50.jsonl",
            #     "metadata_overrides": {"eval_dataset": "swebench_verified_50"},
            # },
        ],
    }

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    # CP shards the sequence; max_tokens_per_gpu must be >= the longest sample's
    # per-CP-rank token count or dynamic batching can't place it.
    context_parallel_size = 2
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    use_dynamic_batch_size = True
    max_tokens_per_gpu = rollout_max_context_len // context_parallel_size  # 16384
    log_probs_chunk_size = 1024
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
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
    # NOTE: async_rl_research reads AGENT_*/MODAL_* names, not the upstream SWE_*.
    environment = {
        "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "MODAL_EXPOSE_ADAPTER": "1",  # sandboxes reach adapter via forward tunnel
        "MODAL_ENVIRONMENT": "junlin-dev",  # env the agent sandboxes boot in
        # harbor rows resolve relative task_path here; evalset.py uses the same root.
        "ASYNC_RL_TASK_ROOT": f"{DATA_PATH}",
        "ASYNC_RL_AGENT_DRIVER": "async_rl_research.agent.mini_swe_agent",
        "AGENT_TIME_BUDGET_SEC": "1800",  # wallclock per agent run
        "AGENT_EVAL_TIMEOUT_SEC": "600",  # wallclock cap on the evaluator sandbox
        "MODAL_BOOT_CONCURRENCY": "8",  # max concurrent sandbox creates
        "SLIME_AGENT_SANDBOX_CPU": "2",
        "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
        # Authenticated Docker Hub pulls for per-instance SWE images avoid the
        # anonymous pull limit. Points at a modal.Secret with REGISTRY_USERNAME/
        # PASSWORD (modal secret create dockerhub-creds REGISTRY_USERNAME=<user>
        # REGISTRY_PASSWORD=<token> --env <your-env>).
        # "MODAL_REGISTRY_SECRET": "dockerhub-creds",
        "SHIM_PORT": "18002",
    }

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = os.environ.get("WANDB_PROJECT") or "Modal"
    wandb_group = _RUN_TAG
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        """Pull SWE-Gym-Lite from HF and convert to slime prompt JSONL.

        Run on the slime-data volume: ``modal run slime/modal_train.py::download_data``.
        """
        import os
        import sys

        sys.path.insert(0, "/root/slime")  # local_slime overlay → async_rl_research
        from async_rl_research.environment.convert2slime.swe_gym import load_hf, write_jsonl

        out_dir = f"{DATA_PATH}/swe_gym_lite"
        os.makedirs(out_dir, exist_ok=True)
        rows = load_hf("train", lite=True, limit=None)
        count = write_jsonl(rows, f"{out_dir}/swe_gym_lite.jsonl")
        print(f"wrote {count} SWE-Gym-Lite rows -> {out_dir}/swe_gym_lite.jsonl")


slime = _Slime()
