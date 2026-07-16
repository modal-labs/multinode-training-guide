"""Qwen3.6-35B-A3B on Frontier-CS algorithmic (competitive programming) — noncolocate, three nodes.

Frontier-CS sibling of ``w_qwen3_6_swe_noncolocate_3n``: identical topology and
engine recipe (1 train + 2 rollout nodes, dp-attention OFF, 8× TP2 SGLang behind
sgl-router — see that config's docstring for the perf rationale), only the
**task family** changes (SWE-Gym-Lite → Frontier-CS).

Frontier-CS runs as **harbor tasks** (per-task Dockerfile + in-sandbox
``tests/evaluate.py``). The agent writes ``/app/solution.cpp`` and iterates with
``bash /app/submit.sh``, which POSTs to a **verifier server** (Node + go-judge)
booted once per worker by ``FrontierCsEnv`` (``environment/verifier_server/``). The
2.5 GB of testdata rides on slime-data (``frontier_cs/problems/``); the judge mounts
slime-data and reads it. Final grade = the judge's score on the final solution.cpp
(``rewards.py`` shapes it; ``ASYNC_RL_REWARD_SHAPE`` picks fractional|binary|thresholded).
Rows carry ``task_type=frontier_cs`` (set by the converter), so ``FrontierCsEnv``
runs them — no env wiring needed beyond the verifier knobs below.

Self-contained (inherits only ``SlimeConfig``): the model / checkpoint / agent env /
algorithm / optimizer / training parallelism are spelled out inline — see
``w_qwen3_6_swe_colocate_1n`` for the model rationale (qwen3.5 arch, TP2 cap, EAGLE,
MoE dispatch, the one-time torch_dist conversion) and ``w_qwen3_6_swe_noncolocate_3n``
for the noncolocate engine recipe.

Competitive programming is reasoning-heavy, so ``eval_max_response_len`` is raised
to 24576 (the SWE default 8192 guillotines Qwen3.6's first <think> mid-thought
before any tool call — 32/38 eval problems died at exactly 8192 with 0 tool calls;
cf. ``w_qwen3_6_frontier_cs_eval``). ``rollout_max_response_len`` (the TRAINING
per-turn cap) is left at 8192 to match the existing frontier-cs train configs; bump
it if early training rollouts show the same guillotine.

Prereqs before launch (see agentic_rl/environment/convert2slime/README.md):
publish the frontier-cs dataset repo (jsonl + tasks/ + the 2.5 GB problems/) to HF;
``download_data`` pulls all of it onto slime-data.

    EXPERIMENT_CONFIG=w_qwen3_6_frontier_cs_noncolocate_3n \
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
_RUN_TAG = run_tag("qwen3.6-35b-a3b-frontier-cs-noncolocate-3n")

# Datasets (one HF repo each; see configs/datasets.py). Train on frontier_cs; eval
# the full held-out frontier_cs slice + USACO transfer (50).
_TRAIN = "frontier_cs"
_EVAL = [("frontier_cs", None), ("usaco", 50)]

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

    # ── Async noncolocate (rollout & training on separate nodes) ───────────────
    async_mode = True
    colocate = False
    actor_num_nodes = 1            # 1 training node (8 GPU, Megatron)
    actor_num_gpus_per_node = 8
    update_weights_interval = 2    # resync weights every 2 rollout steps
    update_weight_buffer_size = 2147483648  # bucket the update like upstream CI

    # ── Custom agentic rollout (reward computed inline; no rm_type) ──────────
    custom_generate_function_path = "agentic_rl.generate.generate"
    custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"
    # Episode limits read off args (launcher materializes this dict to a temp YAML).
    custom_config_path = {
        "agentic_max_steps": 75,
        "agentic_episode_timeout": 1800,
        "agentic_eval_timeout": 600,
    }
    metadata_key = "metadata"
    # Frontier-CS algorithmic (harbor; task_type=frontier_cs), pulled into /data/frontier_cs/.
    prompt_data = train_path(_TRAIN)
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = False  # the adapter renders the chat template itself
    rollout_shuffle = True
    rm_type = None  # reward from the task env rollout (FrontierCsEnv judge), not a reward model
    balance_data = True

    # ── Rollout sizing ────────────────────────────────────────────────────────
    num_rollout = 500
    rollout_batch_size = 32
    rollout_max_response_len = 8192   # TRAINING per-turn cap (see docstring on the CP guillotine)
    rollout_temperature = 1.0
    n_samples_per_prompt = 8
    num_steps_per_rollout = 1
    global_batch_size = 256  # rollout_batch_size * n_samples_per_prompt // steps
    micro_batch_size = 1
    rollout_max_context_len = 32768 * 2  # 64k multi-turn prompt+response budget
    sglang_reasoning_parser = "qwen3"  # strip <think> blocks
    # mini-swe-agent v2 needs the model-matched parser for native tool-calls.
    sglang_tool_call_parser = "qwen3_coder"

    # ── Rollout engines: 2 rollout nodes, 8× TP2, dp-attention OFF ────────────
    rollout_num_gpus = 16             # 2 rollout nodes
    rollout_num_gpus_per_engine = 2   # TP2 → 16 // 2 = 8 engines (sgl-router load-balanced)
    sglang_mem_fraction_static = 0.7
    sglang_cuda_graph_bs = [1, 2, 4, 8, 16] + list(range(24, 257, 8))

    # Required for gated-deltanet; extra_buffer also radix-caches mamba states
    # across turns (prefix-cache health is the multi-turn bottleneck).
    sglang_mamba_scheduler_strategy = "extra_buffer"

    # EAGLE speculative decoding off the MTP head (decode-latency win); disable
    # this block first if the engine looks off.
    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 3
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 4

    # dp-attention OFF → pure TP. The DP/EP layout flags (sglang_dp_size /
    # sglang_ep_size / sglang_enable_dp_lm_head) stay UNSET — they are meaningless
    # without dp-attention and SGLang would build a contradictory engine if set.
    sglang_enable_dp_attention = False
    sglang_disable_custom_all_reduce = False

    # ── Eval ──────────────────────────────────────────────────────────────────
    # Each pass blocks the train loop on the shared engines. Only eval_interval=None
    # is "off". Frontier-CS eval is reasoning-heavy → raise the per-turn cap (see
    # docstring); rollout_max_context_len above (64k) already matches the eval recipe.
    eval_interval = 10
    skip_eval_before_train = True

    eval_max_response_len = 24576  # CP needs room to finish <think> + iterate (was 8192)
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
    moe_enable_deepep = True
    use_dynamic_batch_size = True
    # Retained at the colocate_1n base value (32768 // CP = 16384); it does NOT
    # scale with the 64k rollout_max_context_len above. Bump to 32768 if you want
    # the budget to track the full 64k context — at the cost of more activation
    # memory per GPU.
    max_tokens_per_gpu = 16384
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

    # ── Environment: SWE sandbox knobs + Frontier-CS verifier-server wiring ───
    # FrontierCsEnv boots the verifier server (vm_runtime Modal Sandbox) once per
    # worker and exports FRONTIER_CS_JUDGE_URL. Set FRONTIER_CS_JUDGE_URL here to
    # point at a pre-deployed judge instead (skips the per-worker boot).
    # ASYNC_RL_REWARD_SHAPE picks the central reward shape (fractional|binary|thresholded).
    environment = {
        "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "MODAL_ENVIRONMENT": "junlin-dev",  # env the agent sandboxes boot in
        # harbor rows resolve relative task_path here; evalset.py uses the same root.
        "ASYNC_RL_TASK_ROOT": f"{DATA_PATH}",
        "SLIME_AGENT_SANDBOX_CPU": "2",
        "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
        "FRONTIER_CS_JUDGE_URL": os.environ.get("FRONTIER_CS_JUDGE_URL", ""),
        "ASYNC_RL_REWARD_SHAPE": os.environ.get("ASYNC_RL_REWARD_SHAPE", "fractional"),
    }

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_group = _RUN_TAG
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        """Pull frontier_cs (jsonl + tasks/ + the 2.5 GB problems/) + USACO eval onto /data.

        The verifier server mounts slime-data and reads /data/frontier_cs/problems —
        no separate volume. Run: ``modal run slime/modal_train.py::download_data``.
        """
        for key in {_TRAIN, *(k for k, _ in _EVAL)}:
            pull(key)  # whole repo into /data/<key>/ (frontier_cs incl. problems/)
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)


slime = _Slime()
