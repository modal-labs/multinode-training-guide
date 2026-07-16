"""Qwen3.6-27B (DENSE) SWE-rebench-V2 (Python) agentic RL — noncolocate, FULLY-ASYNC.

Async duplicate of ``w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n``: same model,
data, topology, engines, algorithm, and optimizer — only the execution mode
changes, using the async recipe extracted from ``qwen36_35b_a3b_agentic``
(the fully-async 35B-A3B run):

  * ``async_mode = True`` → train_async.py (trainer consumes batch N while the
    rollout pool generates N+1; no rollout/train serialization).
  * ``rollout_function_path`` → ``fully_async_rollout.generate_rollout_fully_async``:
    a background worker keeps a fixed pool of in-flight episodes across rollout
    boundaries (pool = sglang_server_concurrency × num_engines); weight-update
    ABORTED groups are requeued, never trained on.
  * ``use_rollout_logprobs = True``: train directly on the rollout (sglang)
    logprobs as the PPO old-policy — the 35B-A3B run MEASURED rollout≈trainer
    (ois=1.0, ppo_kl=0.0016), so this replaces TIS as the off-policy handling.
  * ``no_check_for_nan_in_loss_and_grad``: off-policy RL occasionally produces a
    NaN/inf grad; let slime skip that step instead of Megatron raising.
  * ``use_fault_tolerance = True`` + ``agentic_query_timeout`` /
    ``agentic_max_boot_retries``: keep a hung engine/sandbox from wedging the pool.

Ported into the local slime checkout for this config (fully_async_rollout.py
+ arguments.py, mirroring the 35B-A3B recipe):
  * ``rollout_max_staleness = 4``: caps the worker's in-flight pool at
    staleness x rollout_batch_size = 128 groups (instead of the engine cap of
    sglang_server_concurrency x engines = 1024), so a group is trained at most
    ~4 weight updates after it started generating (Little's law: lag =
    in-flight / consumed-per-step). The first run's version_lag/max grew to 25+
    without this.
  * dynamic sampling (DAPO): the async collector now applies
    ``dynamic_sampling_filter_path`` to each completed group and keeps pulling
    until rollout_batch_size PASSING groups are gathered; zero-std groups are
    dropped, never trained. Over-generation is free (the pool generates
    continuously). ``dynamic_sampling/raw_reward_all`` logs the UNBIASED
    pre-filter mean reward — watch THAT for learning progress, since the
    post-filter rollout/raw_reward is pinned near 0.5 by construction.

NOT carried over (this local slime checkout doesn't have them):
``rollout_pause_generation_mode`` and the ``agentic_ramp_window`` /
``agentic_overlong_*`` knobs.

Also changed to make fully-async actually run on this checkout:
  * eval OFF (``eval_interval = None``): ``generate_rollout_fully_async`` raises
    on evaluation and ``eval_function_path`` defaults to the rollout function.
    Sweep checkpoints with ``w_qwen3_6_27b_swe_rebench_v2_prefilter_eval`` /
    an eval config instead.
  * checkpointing ON (``save`` / ``save_interval`` / ``load``): the first async
    run lost 26 steps when Modal preempted a node and auto-retried with nothing
    saved. Save dir is keyed by a per-launch stamp baked into the image env, so
    a Modal auto-retry RESUMES from the latest checkpoint, while a new local
    launch starts fresh. ``RESUME=<run tag>`` manually continues an earlier
    run — see the resume switch near the top of this file.

Training data is prefiltered to the mixed-outcome tasks recorded by
``w_qwen3_6_27b_swe_rebench_v2_prefilter_eval`` — run that config's workflow
before this config's ::download_data.

Before the first run (one-time HF -> Megatron torch_dist conversion):

    EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n_async \
        modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint

Launch:

    EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n_async \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

import json
import os
from datetime import datetime
from pathlib import Path

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
# including Modal auto-retries after a preemption — reuses the SAME stamp.
# New local launch → new stamp → fresh run; auto-retry → same stamp → same
# save dir → resumes from the latest checkpoint automatically.
_LAUNCH_STAMP = os.environ.get("LAUNCH_STAMP") or f"{datetime.now():%Y%m%d-%H%M%S}"

# W&B run name + launch stamp (stable across auto-retries, unlike run_tag()).
_RUN_TAG = f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-swe-rebench-v2-noncolocate-5n-async'}-{_LAUNCH_STAMP}"

# Datasets (one HF repo each; see configs/datasets.py). swe_rebench_v2 Python
# tasks are training (no in-distribution holdout); the transfer eval slice the
# sync config runs is disabled here (fully-async has no eval branch) but the
# specs are kept so download_data still stages the sets.
_TRAIN = "swe_rebench_v2"
_EVAL = [("swegym_lite", None), ("swebench_verified", None)]  # staged, not run (eval off in fully-async)

# Train only on the mixed-outcome tasks (0 < solved < 16 base-model rollouts →
# nonzero GRPO advantage); ids come from w_qwen3_6_27b_swe_rebench_v2_prefilter_eval.
_PREFILTER_IDS = f"{DATA_PATH}/{_TRAIN}/prefilter_ids.json"
_PREFILTER_TRAIN = f"{DATA_PATH}/{_TRAIN}/train.prefilter.jsonl"

# Local env vars baked into the image at launch so the in-container config
# import (train() re-imports this module remotely) sees the same values.
# LAUNCH_STAMP is what makes auto-retry resume work: minted locally above,
# read back from the image env by every container (re)start of this launch.
_IMAGE_ENV = {
    k: v for k in ("WANDB_PROJECT", "WANDB_GROUP", "RESUME") if (v := os.environ.get(k)) is not None
} | {"LAUNCH_STAMP": _LAUNCH_STAMP}

# ── Resume switch ──────────────────────────────────────────────────────────────
# Default: every LOCAL launch is FRESH (new stamp → new save dir), but a Modal
# auto-retry of the same launch reuses the stamp and resumes from the latest
# checkpoint. To manually continue an earlier run, relaunch with
# RESUME=<its run tag> (the W&B group name, e.g.
# "qwen3.6-27b-swe-rebench-v2-noncolocate-5n-async-20260708-1830xx"):
#
#     RESUME=<tag> EXPERIMENT_CONFIG=... uv run --no-dev modal run -d slime/modal_train.py::train
#
# Optionally also set WANDB_GROUP=<tag> to keep the W&B group continuous.
_RESUME = os.environ.get("RESUME")

modal = ModalConfig(
    gpu="H200",
    memory=(1024, int(2 * 1024 * 1024)),
    # Both prior runs died at the FIRST checkpoint save (~rollout 19, ~5-8h in):
    # /tmp/ray hit the 512 GiB default container-disk quota (Ray object spilling
    # + engine logs accumulate; the ~112 GB save spike tips it over) → missed
    # heartbeats → node marked dead. 2 TiB gives ample headroom.
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

    # ── Noncolocate, FULLY-ASYNC (rollout & training overlap, separate nodes) ──
    # async_mode=True → train_async.py: the trainer consumes rollout N while the
    # background pool generates N+1.
    async_mode = True
    # Staleness window: in-flight pool = 4 × rollout_batch_size = 128 groups
    # (was 1024 = sglang_server_concurrency × engines, which let version_lag/max
    # climb to 25+). 128 groups × 8 samples = 1024 episodes still saturates the
    # engines' per-sample concurrency budget exactly.
    rollout_max_staleness = 4
    colocate = False
    actor_num_nodes = 2           # 2 training nodes (16 GPU, Megatron) → DP2 at TP4×CP2
    actor_num_gpus_per_node = 8
    rollout_num_gpus = 32          # 4 rollout nodes → 32 // 2 = 16 TP2 engines
    update_weights_interval = 1    # resync every step; in-flight episodes carry mixed weight_versions
    update_weight_buffer_size = 2147483648  # bucket the update like upstream CI

    # ── Custom agentic rollout (reward computed inline; no rm_type) ──────────
    # Fully-async worker (from the 35B-A3B agentic recipe): keeps a fixed pool of
    # in-flight episodes across rollout boundaries; weight-update ABORTED groups
    # are requeued to the buffer, never shipped to training.
    rollout_function_path = "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
    custom_generate_function_path = "agentic_rl.generate.generate"
    custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"
    use_fault_tolerance = True  # rollout health monitoring; a dead engine can't wedge the pool
    # Episode limits read off args (launcher materializes this dict to a temp YAML).
    custom_config_path = {
        "agentic_max_steps": 75,
        "agentic_episode_timeout": 1800,
        "agentic_eval_timeout": 300,
        "agentic_exec_timeout": 120,  # per-command sandbox exec
        "agentic_query_timeout": 600,  # per-turn /generate cap — bounds hung generations in the async pool
        "agentic_max_boot_retries": 3,  # then ship a masked reward-0 sample (bad-image guard)
        "router_policy": "consistent_hashing",
    }
    metadata_key = "metadata"
    # Harbor build of SWE-rebench-V2, pulled from HF into /data/swe_rebench_v2/
    # then PREFILTERED to the mixed-outcome subset (see download_data).
    prompt_data = _PREFILTER_TRAIN
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
    rollout_max_context_len = 32768 * 2  # 64k multi-turn prompt+response budget
    sglang_reasoning_parser = "qwen3"  # strip <think> blocks
    # mini-swe-agent v2 needs the model-matched parser for native tool-calls.
    sglang_tool_call_parser = "qwen3_coder"

    # ── Dynamic sampling (DAPO): discard zero-variance groups ──────────────────
    # The async collector applies this filter to each completed group and keeps
    # pulling from the pool until rollout_batch_size PASSING groups are gathered
    # (first run: 13-20 of 32 groups per step were zero-std → no gradient).
    # over_sampling_batch_size is meaningless here — the pool over-generates
    # continuously by design. Unbiased reward: dynamic_sampling/raw_reward_all.
    dynamic_sampling_filter_path = "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"

    # ── Rollout engines: 4 rollout nodes, 16× TP2, dp-attention OFF ───────────
    # Dense 27B at TP2 ≈ 27GB weights/GPU — comfortable on H200 (141GB), leaving a
    # large KV pool. If SGLang TP>1 misbehaves on this family (see miles' note on
    # sglang#21039), drop to rollout_num_gpus_per_engine=1 → 32 TP1 engines.
    rollout_num_gpus_per_engine = 2   # TP2 → 32 // 2 = 16 engines (sgl-router load-balanced)
    # Per-engine in-flight cap (× 16 engines = 1024 concurrent episodes / Modal
    # sandboxes). The fully-async pool is min(this × engines, staleness window)
    # = min(1024, 128 groups × 8) — the two bounds coincide at 1024 episodes.
    # The slime default (512) would mean 8192 concurrent sandboxes here.
    sglang_server_concurrency = 64
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

    # EAGLE speculative decoding off the native MTP head. The dense 27B DOES ship
    # one (mtp.safetensors in Qwen/Qwen3.6-27B); SGLang ≥0.5.10 serves it via
    # EAGLE/NEXTN with the same steps/topk/draft settings Qwen recommends, and it
    # requires mamba_scheduler_strategy="extra_buffer" (set above).
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

    # ── Eval: OFF (fully-async) ───────────────────────────────────────────────
    # generate_rollout_fully_async raises on evaluation=True, and slime defaults
    # eval_function_path to the rollout function — so any eval pass would crash
    # the run. Sweep checkpoints with the eval configs instead. (To eval in-run,
    # set eval_function_path = "slime.rollout.sglang_rollout.generate_rollout"
    # and accept that eval serializes against the async pool.)
    eval_interval = None
    skip_eval_before_train = True

    eval_max_response_len = 8192
    eval_config = {
        "defaults": {
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.6,  # low-but-nonzero: Qwen3 degenerates at greedy
            "top_p": 1.0,
        },
        # Built from _EVAL (key, n) specs; unused while eval_interval is None.
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
    # (16384 would fail to place the longest episodes.) Matches the 35B-A3B
    # noncolocate budget; full recompute keeps activations in check at TP4.
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
    # Off-policy RL occasionally produces a NaN/inf grad; let slime skip that
    # step instead of Megatron raising (from the 35B-A3B fully-async recipe).
    no_check_for_nan_in_loss_and_grad = True
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
    save_interval = 20  # every 20 rollouts ≈ 5h at ~15 min/step
    load = save

    # ── Algorithm ─────────────────────────────────────────────────────────────
    advantage_estimator = "grpo"
    use_kl_loss = False
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0
    eps_clip = 0.2
    eps_clip_high = 0.28  # asymmetric clip for stale samples
    # Off-policy handling from the 35B-A3B fully-async run: train on the rollout
    # (sglang) logprobs directly as the PPO old-policy (agentic_rl records
    # per-token rollout_log_probs). That run MEASURED rollout≈trainer (ois=1.0,
    # ppo_kl=0.0016), so TIS stays off; ppo_kl + pg_clipfrac (logged for free in
    # the GRPO loss) are the ongoing mismatch proxy. Incompatible with use_tis.
    use_rollout_logprobs = True

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
        """Pull the datasets, then filter train.jsonl to the prefilter id list."""
        for key in {_TRAIN, *(k for k, _ in _EVAL)}:
            pull(key)
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)

        ids_path = Path(_PREFILTER_IDS)
        if not ids_path.exists():
            raise FileNotFoundError(f"{ids_path}: run w_qwen3_6_27b_swe_rebench_v2_prefilter_eval first")
        ids = set(json.loads(ids_path.read_text())["instance_ids"])
        rows = Path(train_path(_TRAIN)).read_text().splitlines()
        kept = [r for r in rows if json.loads(r)["metadata"]["instance_id"] in ids]
        Path(_PREFILTER_TRAIN).write_text("\n".join(kept) + "\n")
        print(f"[prefilter] kept {len(kept)}/{len(rows)} train rows -> {_PREFILTER_TRAIN}")


slime = _Slime()
