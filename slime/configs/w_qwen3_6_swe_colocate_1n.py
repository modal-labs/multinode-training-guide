"""Qwen3.6-35B-A3B SWE agentic RL — colocated, single node (1× H200:8).

Port of ``w_qwen3_swe_colocate_1n`` to Qwen3.6-35B-A3B, self-contained so it can
be tuned independently. Qwen3.6 uses the qwen3.5 architecture (hybrid
gated-deltanet + full-attention, gated attention output, 248k vocab) via
``scripts/models/qwen3.5-35B-A3B.sh`` + ``slime_plugins.models.qwen3_5``.
Colocated sync: each step runs rollout, then the engine offloads and Megatron
trains.

Model-driven deltas vs the 30B-A3B version:
  - TP 4 -> 2: full-attention layers have num_query_groups=2 (Megatron needs
    num_query_groups % TP == 0).
  - mamba scheduler "extra_buffer" for the deltanet state pool; also
    radix-caches mamba states (critical for multi-turn re-prefill).
  - EAGLE speculative decoding off the MTP head (latency win on decode-bound
    rollout; qwen3_5 bridge keeps the draft head fresh across weight updates).
  - MoE dispatch: flex + DeepEP (config flags come after the model script's
    alltoall MODEL_ARGS, so they win).
  - New torch_dist conversion required (one-time):
        EXPERIMENT_CONFIG=w_qwen3_6_swe_colocate_1n \
        modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint

Verify on the first run: tool-call/reasoning parsers (suspect first if the agent
format-errors in a loop on turn 1); max_tokens_per_gpu >= context_len/CP (raise
CP to 4 if training OOMs); mem_fraction_static=0.7 (drop toward 0.5 if startup
OOMs during cuda-graph capture). Training rows are the harbor build of
SWE-Gym-Lite (``env/harbor.py``).
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
_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-gym-lite-colocate-1n")


HF_EVAL_REPO = "junlin-modal/agentic-rl-evalsets"

_WANDB_IMAGE_ENV = {
    k: v for k in ("WANDB_PROJECT", "WANDB_GROUP") if (v := os.environ.get(k)) is not None
}

modal = ModalConfig(
    gpu="H200",
    local_slime="/Users/junlin/Documents/Research/async-rl/slime",
    image_env=_WANDB_IMAGE_ENV,
    image_run_commands=["pip install modal", f"rm -rf {HF_CACHE_PATH}"],
)


class _Slime(SlimeConfig):
    # qwen3.5 architecture (hybrid gated-deltanet + full attention).
    slime_model_script = "scripts/models/qwen3.5-35B-A3B.sh"
    make_vocab_size_divisible_by = 32

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3.6-35B-A3B"
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-35B-A3B_torch_dist"

    # ── Colocate / sync ───────────────────────────────────────────────────────
    async_mode = False
    colocate = True
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    update_weights_interval = 1  # sync: fresh weights every step
    update_weight_buffer_size = 2147483648  # bucket the update like upstream CI

    # ── Custom agentic rollout (reward computed inline; no rm_type) ──────────
    custom_generate_function_path = "async_rl_research.generate.generate"
    metadata_key = "metadata"
    # Harbor build of SWE-Gym-Lite (download_data re-roots task_path under /data).
    prompt_data = f"{DATA_PATH}/swegym_lite/swegym_lite.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = False  # the adapter renders the chat template itself
    rollout_shuffle = True
    rm_type = None  # reward from the task env rollout (env/harbor.py), not a reward model
    balance_data = True

    # ── Rollout sizing ────────────────────────────────────────────────────────
    num_rollout = 500
    rollout_batch_size = 32
    rollout_max_response_len = 8192
    rollout_temperature = 1.0
    n_samples_per_prompt = 8
    num_steps_per_rollout = 1
    global_batch_size = 256  # rollout_batch_size * n_samples_per_prompt // steps
    micro_batch_size = 1
    rollout_max_context_len = 32768  # multi-turn prompt+response budget
    sglang_reasoning_parser = "qwen3"  # strip <think> blocks
    # mini-swe-agent v2 needs the model-matched parser for native tool-calls.
    sglang_tool_call_parser = "qwen3_coder"
    rollout_num_gpus_per_engine = 8

    # ── Engine sizing under colocation ────────────────────────────────────────
    # Megatron residuals share the 141GB, so the static pool shrinks vs a
    # dedicated rollout node. See docstring for the OOM playbook.
    sglang_mem_fraction_static = 0.7
    sglang_cuda_graph_bs = [1, 2, 4, 8, 16] + list(range(24, 257, 8))

    # Required for gated-deltanet; extra_buffer also radix-caches mamba states
    # across turns (prefix-cache health is the colocate bottleneck).
    sglang_mamba_scheduler_strategy = "extra_buffer"

    # EAGLE speculative decoding off the MTP head (decode-latency win); disable
    # this block first if the engine looks off.
    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 3
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 4

    sglang_enable_dp_attention = False
    # sglang_dp_size = 8
    # sglang_ep_size = 8

    sglang_disable_custom_all_reduce = True

    # ── Eval ──────────────────────────────────────────────────────────────────
    # Subsets built with `python -m async_rl_research.evalset`. Each pass blocks
    # the train loop on the shared engines, so keep subsets small (full sweeps:
    # w_qwen3_swe_eval). Only eval_interval=None is "off".
    eval_interval = 5
    skip_eval_before_train = True

    eval_max_response_len = 8192
    eval_config = {
        "defaults": {
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.6,  # low-but-nonzero: Qwen3 degenerates at greedy
            "top_p": 1.0,
        },
        # metadata_overrides keeps per-dataset attribution in the flattened dump.
        "datasets": [
            # In-distribution: held-out subsample of the SWE-Gym-Lite train set.
            {
                "name": "swe_gym_lite_100",
                "path": f"{DATA_PATH}/evalsets/v0/swe_gym_lite_100.jsonl",
                "metadata_overrides": {"eval_dataset": "swe_gym_lite_100"},
            },
            # Held-out transfer: competitive-programming (harbor USACO).
            {
                "name": "usaco_50",
                "path": f"{DATA_PATH}/evalsets/v0/usaco_50.jsonl",
                "metadata_overrides": {"eval_dataset": "usaco_50"},
            },
        ],
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
    moe_enable_deepep = True
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
        "MODAL_BOOT_TIMEOUT_SEC": "600",  # per-sandbox boot/image-pull cap
        # "ASYNC_RL_TASK_TIMEOUT_OVERRIDE": "1",  # override using task.toml timeouts
        "SLIME_AGENT_SANDBOX_CPU": "2",
        "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
        # "MODAL_REGISTRY_SECRET": "dockerhub-creds",
        "SHIM_PORT": "18002",
        "MSWE_STEP_LIMIT": "75",
    }

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = os.environ.get("WANDB_PROJECT") or "Modal"
    wandb_group = _RUN_TAG
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        """Stage the run's data: harbor SWE-Gym-Lite train set + eval slices.

        Sparse-clone harbor-datasets (swegym-lite subtree), convert into
        /data/swegym_lite, re-root task_path under /data, then pull the
        swe_gym_lite_100 + usaco_50 eval slices (eval-pinned copies win on overlap).

        Run on the slime-data volume: ``modal run slime/modal_train.py::download_data``.
        """
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        sys.path.insert(0, "/root/slime")  # local_slime overlay → async_rl_research
        from async_rl_research.environment.convert2slime.harbor import main as convert_main

        clone = tempfile.mkdtemp(prefix="harbor-datasets-")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "https://github.com/harbor-framework/harbor-datasets.git", clone],
            check=True,
        )
        subprocess.run(["git", "-C", clone, "sparse-checkout", "set", "datasets/swegym-lite"], check=True)
        convert_main(
            ["--tasks-dir", f"{clone}/datasets/swegym-lite",
             "--out-dir", f"{DATA_PATH}/swegym_lite", "--name", "swegym_lite"]
        )

        # Converter writes task_path as "tasks/<id>"; re-root to
        # "swegym_lite/tasks/<id>" to resolve under ASYNC_RL_TASK_ROOT=/data.
        jsonl = Path(f"{DATA_PATH}/swegym_lite/swegym_lite.jsonl")
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        for row in rows:
            task_path = row["metadata"]["task_path"]
            if not task_path.startswith("swegym_lite/"):
                row["metadata"]["task_path"] = f"swegym_lite/{task_path}"
        jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"re-rooted {len(rows)} rows' task_path under swegym_lite/ -> {jsonl}")

        # Eval slices: swe_gym_lite_100 + usaco_50 (repo layout mirrors /data).
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=HF_EVAL_REPO,
            repo_type="dataset",
            local_dir=str(DATA_PATH),
            allow_patterns=[
                "evalsets/v0/swe_gym_lite_100.jsonl",
                "evalsets/v0/usaco_50.jsonl",
                "swegym_lite/**",
                "usaco/**",
            ],
        )
        print(f"downloaded swe_gym_lite_100 + usaco_50 eval slices of {HF_EVAL_REPO} -> {path}")


slime = _Slime()
