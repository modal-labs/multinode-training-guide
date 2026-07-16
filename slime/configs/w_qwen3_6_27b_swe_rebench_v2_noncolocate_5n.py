"""Qwen3.6-27B (DENSE) SWE-rebench-V2 (Python) agentic RL — noncolocate, six nodes.

Dense-model sibling of ``w_qwen3_6_swe_rebench_v2_noncolocate_5n`` (which runs the
sparse Qwen3.6-35B-A3B MoE). Same data (SWE-rebench-V2 Python subset), algorithm,
optimizer, agent env, and eval slice; rollout sizing differs (32×16 here vs the
35B-5n's 64×16), and this config runs TWO training nodes.

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

Before the first run (one-time HF -> Megatron torch_dist conversion):

    EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n \
        modal run slime/modal_train.py::convert_hf_to_megatron_checkpoint

Launch:

    EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

import json
import os
from pathlib import Path

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
_RUN_TAG = run_tag("qwen3.6-27b-swe-rebench-v2-noncolocate-5n")

# Datasets (one HF repo each; see configs/datasets.py). swe_rebench_v2 Python
# tasks are training (no in-distribution holdout), so eval is a transfer slice on
# the published swegym_lite + swebench_verified held-out sets.
_TRAIN = "swe_rebench_v2"
_EVAL = [("swegym_lite", None), ("swebench_verified", None)]  # transfer/generalization eval; [] to disable

# Train only on the mixed-outcome tasks (0 < solved < 16 base-model rollouts →
# nonzero GRPO advantage); ids come from w_qwen3_6_27b_swe_rebench_v2_prefilter_eval.
_PREFILTER_IDS = f"{DATA_PATH}/{_TRAIN}/prefilter_ids.json"
_PREFILTER_TRAIN = f"{DATA_PATH}/{_TRAIN}/train.prefilter.jsonl"

_WANDB_IMAGE_ENV = {
    k: v for k in ("WANDB_PROJECT", "WANDB_GROUP") if (v := os.environ.get(k)) is not None
}

modal = ModalConfig(
    gpu="H200",
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
