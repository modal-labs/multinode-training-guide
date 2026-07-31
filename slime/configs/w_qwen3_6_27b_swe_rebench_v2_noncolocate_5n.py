"""Qwen3.6-27B (DENSE) SWE-rebench-V2 (Python) agentic RL — noncolocate, six nodes.

Dense-model sibling of ``w_qwen3_6_swe_rebench_v2_noncolocate_5n`` (which runs the
sparse Qwen3.6-35B-A3B MoE). Same data (SWE-rebench-V2 Python subset), algorithm,
optimizer, and agent env; rollout sizing differs (32×16 here vs the 35B-5n's
64×16), this config runs TWO training nodes, and the eval slice is in-distribution
rather than transfer (see below).

Self-contained: inherits only ``SlimeConfig`` and spells every arg out inline (no
inheritance from another experiment config), so the dense recipe can be tuned in
isolation.

Topology: 2 training nodes (16 GPU, Megatron, TP4 × CP2 -> DP2) + 4 rollout nodes
(32 GPU, 16× TP2 SGLang engines behind sgl-router) = 6 nodes / 48 GPU. The ``5n``
in the filename is inherited from the lineage; the actual footprint is 6 nodes.
Sync noncolocate (``async_mode=False``), matching the rebench lineage — set
``async_mode=True`` for rollout/train overlap (the SWE-noncolocate base default).

Dense (27B) deltas vs the 35B-A3B MoE config — see the per-section comments:
  * model script ``qwen3.5-27B.sh`` (dense GDN backbone: 64 layers, hidden 5120,
    num_query_groups=4, gated attention, 248k vocab) + HF ``Qwen/Qwen3.6-27B``.
  * training TP 2 -> 4 (num_query_groups=4 lifts the TP cap; the canonical dense
    27B recipe). TP4 × CP2 × PP1 over 2 train nodes (16 GPU) -> DP2.
  * MoE parallelism removed: EP 8 -> 1 and the MoE dispatch flags
    (``moe_token_dispatcher_type`` / ``moe_enable_deepep``) dropped — a dense model
    has no experts.
  * 2 training nodes (vs the 35B-5n's 1): DP2 halves the train step wall.

Training data is prefiltered to the mixed-outcome tasks recorded by
``w_qwen3_6_27b_swe_rebench_v2_prefilter_eval`` — run that config's workflow
before this config's ::download_data.

Eval: SWE-rebench-V2 ships no holdout (its ``eval.jsonl`` is empty), so the eval
slice is a fixed 500-task random draw from the prefiltered training pool. The eval
path never applies the dynamic sampling filter, so this is the unbiased reward
curve to read — ``rollout/average_last_reward`` only reflects the mixed groups DAPO
kept. ``skip_eval_before_train = False`` puts a base-model anchor at rollout 0.
Scoring a saved checkpoint offline: ``w_qwen3_6_27b_swe_rebench_v2_eval``.

SWE-bench Verified (both flags default OFF, and both must be set at LAUNCH time —
they are baked into the image env so in-container re-imports agree):

    SWEBENCH_EVAL=1     score the 500-task Verified set alongside the rebench slice
    SWEBENCH_TRAIN=1    also TRAIN on Verified (implies SWEBENCH_EVAL)

``SWEBENCH_EVAL`` is the cheap, safe one: Verified stays held out, so it remains a
valid independent benchmark for the base-vs-checkpoint comparison. ``SWEBENCH_TRAIN``
gives that up — a benchmark you train on no longer measures generalization — and has
three practical costs: Verified ships no train split (its ``eval.jsonl`` is the only
file, so training reads the same rows being scored), the prefilter ids do not cover
it so DAPO screens every group from scratch, and its tasks BUILD their sandbox from
a per-task Dockerfile instead of pulling a prebuilt image, so first-touch episodes
are much slower than rebench's.

Checkpointing + resume (ported from the ``_async`` sibling): Modal GPUs last at
most 24h, so a run MUST survive restarts to reach num_rollout. Checkpoints save
every ``save_interval`` rollouts under a per-launch save dir; a Modal auto-retry of
the same launch reuses the launch stamp and resumes from the latest checkpoint
automatically, while a new local launch starts fresh. To manually continue an
earlier (e.g. 24h-expired) run, relaunch with RESUME=<its run tag>:

    RESUME=qwen3.6-27b-swe-rebench-v2-noncolocate-5n-<stamp> \
    EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train

Optionally also set WANDB_GROUP=<tag> to keep the W&B group continuous.

Before the first run (one-time HF -> Megatron torch_dist conversion):

    EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n \
        modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint

Launch:

    EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

from configs.base import (
    ModalConfig,
    SlimeConfig,
    DATA_PATH,
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
)
from configs.datasets import eval_path, pull, subsample, train_path

# Launch stamp: minted ONCE on the local machine (client-side config import at
# `modal run`) and baked into the image env, so every in-container re-import —
# including Modal auto-retries after a preemption / 24h GPU expiry — reuses the
# SAME stamp. New local launch → new stamp → fresh run; auto-retry → same stamp
# → same save dir → resumes from the latest checkpoint automatically.
_LAUNCH_STAMP = os.environ.get("LAUNCH_STAMP") or f"{datetime.now():%Y%m%d-%H%M%S}"

# ── SWE-bench Verified switches (both default OFF; see the module docstring) ──
# SWEBENCH_EVAL=1 adds the 500-task Verified set as a SECOND eval dataset.
# SWEBENCH_TRAIN=1 additionally TRAINS on it (implies the eval), which trades the
# benchmark away — anything you train on stops being an independent measure.
_SWEBENCH = "swebench_verified"
_SWEBENCH_TRAIN = os.environ.get("SWEBENCH_TRAIN") == "1"
_SWEBENCH_EVAL = _SWEBENCH_TRAIN or os.environ.get("SWEBENCH_EVAL") == "1"

# W&B run name + launch stamp (stable across auto-retries, unlike run_tag()).
_RUN_TAG = (
    f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-swe-rebench-v2-noncolocate-5n'}"
    f"{'-swebench-train' if _SWEBENCH_TRAIN else ''}-{_LAUNCH_STAMP}"
)

# ── Resume switch ──────────────────────────────────────────────────────────────
# Default: every LOCAL launch is FRESH (new stamp → new save dir), but a Modal
# auto-retry of the same launch reuses the stamp and resumes from the latest
# checkpoint. RESUME=<run tag> (the W&B group name) manually continues an
# earlier run — see the resume instructions in the module docstring.
_RESUME = os.environ.get("RESUME")

# Datasets (one HF repo each; see configs/datasets.py). swe_rebench_v2 Python
# tasks are all training data (its eval.jsonl is empty — no published holdout),
# so eval is an IN-DISTRIBUTION slice carved out of the prefiltered train pool.
_TRAIN = "swe_rebench_v2"
_EVAL: list[tuple[str, int | None]] = []  # transfer eval (swegym_lite/swebench_verified) retired; [] to disable

# Train only on the mixed-outcome tasks (0 < solved < 16 base-model rollouts →
# nonzero GRPO advantage); ids come from w_qwen3_6_27b_swe_rebench_v2_prefilter_eval.
_PREFILTER_IDS = f"{DATA_PATH}/{_TRAIN}/prefilter_ids.json"
_PREFILTER_TRAIN = f"{DATA_PATH}/{_TRAIN}/train.prefilter.jsonl"

# Eval slice: a fixed random draw from the prefilter ids, sampled rather than
# taken as a prefix because instance_ids is stored alphabetically sorted — the
# first 500 would cover only 127 of the 446 repos, vs 237 for this draw (base
# difficulty is the same either way: mean solve count 8.17/16 vs 8.08/16).
# These tasks stay IN the training pool, so the curve measures in-distribution
# progress; drop them from _PREFILTER_TRAIN for a clean held-out read instead.
_EVAL_N, _EVAL_SEED = 500, 0
_EVAL_SLICE = f"{DATA_PATH}/{_TRAIN}/eval.prefilter{_EVAL_N}.jsonl"


def _eval_entry(name: str, path: str) -> dict:
    return {"name": name, "path": path, "metadata_overrides": {"eval_dataset": name}}


# The rebench slice is always scored; Verified rides alongside it when flagged, so
# the in-distribution curve and the benchmark move on the same x-axis.
_EVAL_ENTRIES = [_eval_entry(f"rebench_prefilter{_EVAL_N}", _EVAL_SLICE)]
if _SWEBENCH_EVAL:
    _EVAL_ENTRIES.append(_eval_entry(_SWEBENCH, eval_path(_SWEBENCH)))

# Datasets download_data must pull (Verified ships eval.jsonl only — no train split).
_PULL_KEYS = {_TRAIN, *(k for k, _ in _EVAL)} | ({_SWEBENCH} if _SWEBENCH_EVAL else set())

_IMAGE_ENV = {
    k: v
    for k in ("WANDB_PROJECT", "WANDB_GROUP", "RESUME", "SWEBENCH_EVAL", "SWEBENCH_TRAIN")
    if (v := os.environ.get(k)) is not None
} | {"LAUNCH_STAMP": _LAUNCH_STAMP}

modal = ModalConfig(
    gpu="H200",
    memory=(1024, int(2 * 1024 * 1024)),
    # Required now that save_interval is set: both prior runs of the _async
    # sibling died at the FIRST checkpoint save (~rollout 19, ~5-8h in) because
    # /tmp/ray hit the 512 GiB default container-disk quota (Ray object spilling
    # + engine logs accumulate; the ~112 GB save spike tips it over) → missed
    # heartbeats → node marked dead. 2 TiB gives ample headroom.
    ephemeral_disk=2 * 1024 * 1024,  # MiB → 2 TiB
    local_slime="/Users/shariqmobin/Documents/code/work/modal-projects/slime",
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
    # async_mode=False = sync (rollout step, then a train step; no overlap), as in
    # the rebench lineage. Flip to True for fully-overlapped async on these GPUs.
    async_mode = False
    colocate = False
    actor_num_nodes = 2           # 2 training nodes (16 GPU, Megatron) → DP2 at TP4×CP2
    actor_num_gpus_per_node = 8
    rollout_num_gpus = 32          # 4 rollout nodes → 32 // 2 = 16 TP2 engines
    update_weights_interval = 1    # resync weights every rollout step (fully on-policy in sync mode)
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
    # Harbor build of SWE-rebench-V2, pulled from HF into /data/swe_rebench_v2/
    # then PREFILTERED to the mixed-outcome subset (see download_data).
    # SWEBENCH_TRAIN=1 swaps in Verified's eval.jsonl (its only split — the repo
    # ships no train.jsonl), which the prefilter ids do not cover, so every group
    # goes through DAPO unscreened: expect a higher oversampling multiple than the
    # ~3x seen on the prefiltered pool. 500 tasks at rollout_batch_size=32 is also
    # only ~16 rollouts per epoch, so num_rollout=500 would be ~31 epochs.
    prompt_data = eval_path(_SWEBENCH) if _SWEBENCH_TRAIN else _PREFILTER_TRAIN
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
    dynamic_sampling_filter_path = "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    over_sampling_batch_size = 48  # > rollout_batch_size (32); start ~1.5× and watch rollout wall

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

    # ── Eval ──────────────────────────────────────────────────────────────────
    # Each pass blocks the train loop on the shared engines, so keep subsets small
    # (full sweeps: w_qwen3_swe_eval). Only eval_interval=None is "off".
    eval_interval = 5  # aligned with save_interval, so every checkpoint has an eval point
    skip_eval_before_train = False

    eval_max_response_len = 8192
    eval_config = {
        "defaults": {
            "n_samples_per_eval_prompt": 1,
            "temperature": 0.6,  # low-but-nonzero: Qwen3 degenerates at greedy
            "top_p": 1.0,
        },
        # Written by download_data; eval_datasets()/subsample() can't build the
        # rebench entry because they hardcode <key>/eval.jsonl and ours derives
        # from train. Verified is appended when SWEBENCH_EVAL/SWEBENCH_TRAIN is set.
        "datasets": _EVAL_ENTRIES,
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
    # ~3h of progress on the auto-retry resume (save itself is minutes). Note
    # slime has no checkpoint-retention knob, so iters accumulate at ~112 GB.
    save_interval = 5
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

    # ── Environment: PYTHONPATH + Modal sandbox knobs reach the Ray workers ──
    # In-process design: no adapter/tunnel/in-sandbox runner. Episode limits live
    # in custom_config_path; only sandbox/runtime knobs belong here.
    environment = {
        "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "MODAL_ENVIRONMENT": "shariq-dev",  # env the agent sandboxes boot in
        # harbor rows resolve relative task_path here; evalset.py uses the same root.
        "ASYNC_RL_TASK_ROOT": f"{DATA_PATH}",
        "SLIME_AGENT_SANDBOX_CPU": "2",
        "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
        "ASYNC_RL_REWARD_SHAPE": "binary",
    }

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = False
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_group = _RUN_TAG
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        """Pull the datasets, filter train.jsonl to the prefilter id list, then
        carve the fixed eval slice out of that filtered pool."""
        for key in _PULL_KEYS:
            pull(key)
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)

        ids_path = Path(_PREFILTER_IDS)
        if not ids_path.exists():
            raise FileNotFoundError(f"{ids_path}: run w_qwen3_6_27b_swe_rebench_v2_prefilter_eval first")
        id_list = json.loads(ids_path.read_text())["instance_ids"]
        ids = set(id_list)
        rows = Path(train_path(_TRAIN)).read_text().splitlines()
        kept = [r for r in rows if json.loads(r)["metadata"]["instance_id"] in ids]
        Path(_PREFILTER_TRAIN).write_text("\n".join(kept) + "\n")
        print(f"[prefilter] kept {len(kept)}/{len(rows)} train rows -> {_PREFILTER_TRAIN}")

        eval_ids = set(random.Random(_EVAL_SEED).sample(id_list, _EVAL_N))
        eval_rows = [r for r in kept if json.loads(r)["metadata"]["instance_id"] in eval_ids]
        Path(_EVAL_SLICE).write_text("\n".join(eval_rows) + "\n")
        print(f"[eval-slice] {len(eval_rows)} tasks (seed {_EVAL_SEED}) -> {_EVAL_SLICE}")


slime = _Slime()
