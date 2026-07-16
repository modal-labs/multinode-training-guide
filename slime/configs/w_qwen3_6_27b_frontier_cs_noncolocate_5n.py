"""Qwen3.6-27B (DENSE) Frontier-CS algorithmic (competitive programming) — noncolocate, six nodes.

Frontier-CS sibling of ``w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n``: identical
**model** (dense 27B), topology (2 train + 4 rollout), engine recipe, algorithm,
and optimizer — only the **task family** and its data wiring change
(SWE-rebench-V2 Python → Frontier-CS). The ``5n`` in the filename is inherited
from the lineage; the actual footprint is 6 nodes.

Frontier-CS runs as **harbor tasks** (per-task Dockerfile + in-sandbox
``tests/evaluate.py``). The agent writes ``/app/solution.cpp`` and iterates with
``bash /app/submit.sh``, which POSTs to a **verifier server** (Node + go-judge)
booted once per worker by ``FrontierCsEnv`` (``environment/verifier_server/``). The
2.5 GB of testdata rides on slime-data (``frontier_cs/problems/``); the judge mounts
slime-data and reads it. Final grade = the judge's score on the final solution.cpp
(``rewards.py`` shapes it; ``ASYNC_RL_REWARD_SHAPE`` picks fractional|binary|thresholded).
Rows carry ``task_type=frontier_cs`` (set by the converter), so ``FrontierCsEnv``
runs them — no env wiring needed beyond the verifier knobs below.

Competitive programming is reasoning-heavy: the per-turn cap guillotines Qwen3.6's
first <think> mid-thought before any tool call (finish=length, 0 tool calls →
ContextLengthExceeded, reward 0, zero usable turns). Eval evidence: 32/38 problems
died at 8192, and 22/38 STILL died at 24576 (eval run 20260707-183540, temp 0.6) —
so both the TRAINING per-turn cap (``rollout_max_response_len``) and the eval cap
(``eval_max_response_len``) are set to 24576 / 49152 respectively. At temp 1.0
training thinks run longer than eval's, so expect a sizable ContextLengthExceeded
fraction even at 24576 — watch rollout/zero_std and the agentic exit_status metrics
and raise toward 49152 if too many groups are all-fail with 0 turns.

Self-contained: inherits only ``SlimeConfig`` and spells every arg out inline (no
inheritance from another experiment config), so the dense recipe can be tuned in
isolation.

Topology: 2 training nodes (16 GPU, Megatron, TP4 × CP2 -> DP2) + 4 rollout nodes
(32 GPU, 16× TP2 SGLang engines behind sgl-router) = 6 nodes / 48 GPU. Sync
noncolocate (``async_mode=False``); set ``async_mode=True`` for rollout/train overlap.

Dense (27B) recipe — see ``w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n`` for the full
rationale (qwen3.5-27B model script, TP4 × CP2 × PP1, EP collapsed to 1, EAGLE, the
one-time HF→Megatron torch_dist conversion).

Prereqs before launch (see agentic_rl/environment/convert2slime/README.md):
publish the frontier-cs dataset repo (jsonl + tasks/ + the 2.5 GB problems/) to HF;
``download_data`` pulls all of it onto slime-data.

Checkpointing + resume (ported from ``w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n_async``):
Modal GPUs last at most 24h and a rollout+eval cadence here is ~25 steps/day, so a
run MUST survive restarts to reach num_rollout. Checkpoints save every
``save_interval`` rollouts under a per-launch save dir; a Modal auto-retry of the
same launch reuses the launch stamp and resumes from the latest checkpoint
automatically, while a new local launch starts fresh. To manually continue an
earlier (e.g. 24h-expired) run, relaunch with RESUME=<its run tag>:

    RESUME=qwen3.6-27b-frontier-cs-noncolocate-5n-<stamp> \
    EXPERIMENT_CONFIG=w_qwen3_6_27b_frontier_cs_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train

Optionally also set WANDB_GROUP=<tag> to keep the W&B group continuous.

Before the first run (one-time HF -> Megatron torch_dist conversion):

    EXPERIMENT_CONFIG=w_qwen3_6_27b_frontier_cs_noncolocate_5n \
        modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint

Launch:

    EXPERIMENT_CONFIG=w_qwen3_6_27b_frontier_cs_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

import os
from datetime import datetime

from configs.base import (
    ModalConfig,
    SlimeConfig,
    DATA_PATH,
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
)
from configs.datasets import eval_datasets, pull, subsample, train_path

# Launch stamp: minted ONCE on the local machine (client-side config import at
# `modal run`) and baked into the image env, so every in-container re-import —
# including Modal auto-retries after a preemption / 24h GPU expiry — reuses the
# SAME stamp. New local launch → new stamp → fresh run; auto-retry → same stamp
# → same save dir → resumes from the latest checkpoint automatically.
_LAUNCH_STAMP = os.environ.get("LAUNCH_STAMP") or f"{datetime.now():%Y%m%d-%H%M%S}"

# W&B run name + launch stamp (stable across auto-retries, unlike run_tag()).
_RUN_TAG = f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-frontier-cs-noncolocate-5n'}-{_LAUNCH_STAMP}"

# Datasets (one HF repo each; see configs/datasets.py). Train on frontier_cs; eval
# the full held-out frontier_cs slice + USACO transfer (50).
_TRAIN = "frontier_cs"
_EVAL = [("frontier_cs", 5), ("usaco", 50)]

# Local env vars baked into the image at launch so the in-container config
# import (train() re-imports this module remotely) sees the same values.
# LAUNCH_STAMP is what makes auto-retry resume work: minted locally above,
# read back from the image env by every container (re)start of this launch.
# The ASYNC_RL_* reward switches must ride along too: the `environment` dict
# below re-reads os.environ IN-CONTAINER, where the local launch shell's values
# don't exist unless baked here.
_IMAGE_ENV = {
    k: v
    for k in (
        "WANDB_PROJECT",
        "WANDB_GROUP",
        "RESUME",
        "ASYNC_RL_REWARD_SHAPE",
        "ASYNC_RL_OUTCOME_REWARD",
        "ASYNC_RL_SOLVED_BONUS",
        "ASYNC_RL_OUTCOME_GAMMA",
        "THINK_CLOSURE",
    )
    if (v := os.environ.get(k)) is not None
} | {"LAUNCH_STAMP": _LAUNCH_STAMP}

# ── Resume switch ──────────────────────────────────────────────────────────────
# Default: every LOCAL launch is FRESH (new stamp → new save dir), but a Modal
# auto-retry of the same launch reuses the stamp and resumes from the latest
# checkpoint. RESUME=<run tag> (the W&B group name) manually continues an
# earlier run — see the resume instructions in the module docstring.
_RESUME = os.environ.get("RESUME")

modal = ModalConfig(
    gpu="H200",
    memory=(1024, int(2 * 1024 * 1024)),
    ephemeral_disk=2 * 1024 * 1024,  # MiB → 2 TiB
    local_slime="/Users/junlin/Documents/Research/async-rl/slime",
    image_run_commands=[
        f"rm -rf {HF_CACHE_PATH}",
        "apt-get update && apt-get install -y --no-install-recommends rdma-core libibverbs1 ibverbs-providers",
        "uv pip install --system modal mini-swe-agent datasets",
    ],
    image_env={"MSWEA_SILENT_STARTUP": "1", **_IMAGE_ENV},  # no mini-swe banner in rollout logs
)


class _Slime(SlimeConfig):
    # qwen3.5 architecture, DENSE 27B variant. Qwen3.6-27B reuses the qwen3.5
    # Megatron spec (hybrid gated-deltanet + full attention, gated attention
    # output, 248k vocab); same "Qwen3.6 HF checkpoint + qwen3.5 model script"
    # pattern the 35B-A3B configs use.
    slime_model_script = "scripts/models/qwen3.5-27B.sh"
    make_vocab_size_divisible_by = 32

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3.6-27B"
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-27B_torch_dist"

    # ── Noncolocate, sync (rollout & training on separate nodes) ───────────────
    # async_mode=False = sync (rollout step, then a train step; no overlap). Flip to
    # True for fully-overlapped async on these GPUs.
    async_mode = False
    colocate = False
    actor_num_nodes = 2           # 2 training nodes (16 GPU, Megatron) → DP2 at TP4×CP2
    actor_num_gpus_per_node = 8
    rollout_num_gpus = 32          # 4 rollout nodes → 32 // 2 = 16 TP2 engines
    update_weights_interval = 1    # resync weights every rollout step
    update_weight_buffer_size = 2147483648  # bucket the update like upstream CI

    # ── Custom agentic rollout (reward computed inline; no rm_type) ──────────
    custom_generate_function_path = "agentic_rl.generate.generate"
    custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"
    # Episode limits read off args (launcher materializes this dict to a temp YAML).
    custom_config_path = {
        "agentic_max_steps": 75,
        "agentic_episode_timeout": 1800,
        "agentic_eval_timeout": 600,  # frontier-cs judge grade is slower than SWE (was 300)
        "agentic_exec_timeout": 120,  # per-command sandbox exec
        # Forced think closure for runaway <think> (agentic_rl/model.py): OFF for
        # training by default; THINK_CLOSURE=1 flips it (baked into _IMAGE_ENV above
        # so the in-container config re-import resolves the same value). A/B'd first
        # on w_qwen3_6_27b_frontier_cs_eval before enabling here.
        "agentic_close_think_on_length": os.environ.get("THINK_CLOSURE", "0") == "1",
        "agentic_max_think_closures": 2,
        "agentic_think_closure_budget": 4096,
        "router_policy": "consistent_hashing",
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
    num_rollout = 100
    rollout_batch_size = 32   # 27B recipe (matches swe_rebench_v2 27B; the 35B-5n runs 64×16)
    rollout_max_response_len = 24576  # TRAINING per-turn cap (see docstring on the CP guillotine; was 8192)
    rollout_temperature = 1.0
    n_samples_per_prompt = 8
    num_steps_per_rollout = 1
    global_batch_size = 256  # rollout_batch_size * n_samples_per_prompt // steps
    micro_batch_size = 1
    rollout_max_context_len = 32768 * 2  # 64k multi-turn prompt+response budget
    sglang_reasoning_parser = "qwen3"  # strip <think> blocks
    # mini-swe-agent v2 needs the model-matched parser for native tool-calls.
    sglang_tool_call_parser = "qwen3_coder"

    # ── Dynamic sampling (DAPO) — DISABLED; uncomment to experiment ────────────
    # Drops prompt groups whose n_samples_per_prompt rollouts all get the SAME
    # reward (all solved or all failed) — those have zero GRPO advantage, hence
    # zero gradient. With over_sampling_batch_size > rollout_batch_size, slime
    # oversamples prompts and refills until it has rollout_batch_size groups with
    # nonzero reward std, so every optimizer step is dense with signal.
    #   Cost tradeoff (noncolocate, expensive sandbox episodes): refilling means
    #   MORE agent rollouts. Early on, all-fail groups dominate, so aggressive
    #   filtering can balloon rollout wall-time — watch perf/rollout_time and
    #   rollout/dynamic_filter/drop_* (and gate the decision on rollout/zero_std/*).
    #   fractional reward (ASYNC_RL_REWARD_SHAPE) already adds within-group variance,
    #   so the zero-std fraction here is lower than it would be under binary reward.
    # dynamic_sampling_filter_path = "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    # over_sampling_batch_size = 48  # > rollout_batch_size (32); start ~1.5× and watch rollout wall

    # ── Rollout engines: 4 rollout nodes, 16× TP2, dp-attention OFF ───────────
    # Dense 27B at TP2 ≈ 27GB weights/GPU — comfortable on H200 (141GB), leaving a
    # large KV pool. If SGLang TP>1 misbehaves on this family (see miles' note on
    # sglang#21039), drop to rollout_num_gpus_per_engine=1 → 32 TP1 engines.
    rollout_num_gpus_per_engine = 2   # TP2 → 32 // 2 = 16 engines (sgl-router load-balanced)
    # Dedicated rollout GPUs, so the pool can be aggressive. TP2 leaves ~27GB
    # weights/GPU on a 141GB H200; 0.85 gives a ~90GB KV pool. Do NOT raise to
    # 0.90: observed OOM at startup — the initial Megatron->SGLang weight push
    # needs ~2.4GiB free per GPU (the 2GiB update_weight_buffer_size bucket) and
    # 0.90 left only 1.55GiB, killing the run on a 600s NCCL barrier timeout.
    sglang_mem_fraction_static = 0.85
    sglang_cuda_graph_bs = [1, 2, 4, 8, 16] + list(range(24, 257, 8))

    # Required for gated-deltanet; extra_buffer also radix-caches mamba states
    # across turns (prefix-cache health is the multi-turn bottleneck).
    sglang_mamba_scheduler_strategy = "extra_buffer"

    # EAGLE speculative decoding off the MTP head (decode-latency win); disable
    # this block first if the engine looks off / fails to start.
    sglang_speculative_algorithm = "EAGLE"
    sglang_speculative_num_steps = 3
    sglang_speculative_eagle_topk = 1
    sglang_speculative_num_draft_tokens = 4

    # dp-attention OFF → pure TP. The DP/EP layout flags (sglang_dp_size /
    # sglang_ep_size / sglang_enable_dp_lm_head) stay UNSET — they are meaningless
    # without dp-attention and SGLang would build a contradictory engine if set.
    sglang_enable_dp_attention = False
    sglang_disable_custom_all_reduce = False
    qwen_gdn_backend = "flashqla" 

    # ── Eval ──────────────────────────────────────────────────────────────────
    # Each pass blocks the train loop on the shared engines (full sweep:
    # w_qwen3_6_27b_frontier_cs_eval). Only eval_interval=None is "off". Frontier-CS
    # eval is reasoning-heavy → raise the per-turn cap (see docstring); the 64k
    # rollout_max_context_len above already matches the eval recipe.
    eval_interval = 10
    skip_eval_before_train = True

    eval_max_response_len = 49152  # CP needs room to finish <think> + iterate (8192 → 24576 → 49152)
    eval_config = {
        "defaults": {
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.6,  # low-but-nonzero: Qwen3 degenerates at greedy
            "top_p": 1.0,
        },
        # Built from _EVAL (key, n) specs; n subsamples eval.jsonl in download_data.
        "datasets": eval_datasets(_EVAL),
    }

    # ── Training (DENSE 27B parallelism, 2 nodes / 16 GPU) ────────────────────
    # Dense backbone, so no expert parallelism. num_query_groups=4 (vs 2 on the
    # 35B-A3B) lifts the TP cap to 4 (Megatron needs num_query_groups % TP == 0);
    # TP4 shards the 27B weights/grads 4-way (~13.5GB bf16 weights + 27GB fp32
    # grads per GPU), leaving ample activation headroom. TP4 × CP2 × PP1 over
    # 16 GPUs → DP2, so each optimizer step splits the global batch across two
    # model replicas (TP stays intra-node; only DP grad all-reduce crosses nodes).
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    # CP shards the sequence; max_tokens_per_gpu must be >= the longest sample's
    # per-CP-rank token count or dynamic batching can't place it.
    context_parallel_size = 2
    # Dense model → no MoE: EP/ETP collapse to 1 and the MoE dispatch flags
    # (moe_token_dispatcher_type / moe_enable_deepep) the 35B-A3B sets are dropped.
    expert_model_parallel_size = 1
    expert_tensor_parallel_size = 1
    use_dynamic_batch_size = True
    # Must fit the longest sample per CP rank: 64k context / CP2 = 32k tokens/rank.
    # (16384 would fail to place the longest episodes.) Matches the SWE-rebench
    # 27B config; full recompute keeps activations in check at TP4.
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
    use_rollout_logprobs=True
    # Dump every rollout under the W&B group subdir; relaunches overwrite.
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    # ── Checkpointing (fresh per local launch; auto-retry resumes — see top) ──
    # Checkpoints save under this launch's run tag, so re-running the config
    # never silently continues an old run. load = save: on a fresh launch the
    # dir is empty and slime falls back to ref_load; on a Modal auto-retry the
    # stamp (and thus the dir) is reused, so the latest checkpoint is loaded.
    # RESUME=<tag> points both at an earlier launch's dir instead.
    save = f"{CHECKPOINTS_PATH}/swe_ckpts/{_RESUME or _RUN_TAG}"
    # Every 5 rollouts ≈ 3h at ~35 min/step: a 24h GPU expiry then costs at most
    # ~3h of progress on the auto-retry resume (save itself is minutes).
    save_interval = 15
    load = save

    # ── Algorithm ─────────────────────────────────────────────────────────────
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
        # 4 CPUs: straggler episodes are ~76% in-sandbox tool exec (g++ compiles +
        # 50-200-iteration stress loops), CPU-bound at 2 (rollout 11/12 dump profile).
        "SLIME_AGENT_SANDBOX_CPU": "4",
        "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
        "FRONTIER_CS_JUDGE_URL": os.environ.get("FRONTIER_CS_JUDGE_URL", ""),
        "ASYNC_RL_REWARD_SHAPE": os.environ.get("ASYNC_RL_REWARD_SHAPE", "fractional"),
        # Episode-level OUTCOME reward (rewards.py): base strategy final|best|disc_sum,
        # additive solved bonus (Kevin uses 0.3), disc_sum discount (Kevin: 0.4).
        # Defaults are the identity — bit-identical to the pre-interface reward.
        "ASYNC_RL_OUTCOME_REWARD": os.environ.get("ASYNC_RL_OUTCOME_REWARD", "final"),
        "ASYNC_RL_SOLVED_BONUS": os.environ.get("ASYNC_RL_SOLVED_BONUS", "0"),
        "ASYNC_RL_OUTCOME_GAMMA": os.environ.get("ASYNC_RL_OUTCOME_GAMMA", "0.4"),
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
