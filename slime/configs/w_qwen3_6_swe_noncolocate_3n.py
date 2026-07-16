"""Qwen3.6-35B-A3B SWE agentic RL — noncolocate, three nodes (1 train + 2 rollout).

Three-node sibling of ``w_qwen3_6_swe_noncolocate_2n`` that adopts the rollout-perf
study's recommended engine recipe (see
``slime/agentic_rl/profiles/swe_sglang_rollout_perf_profile``):

  * **dp-attention OFF** (pure tensor parallelism) — the single biggest lever; it
    unifies the KV pool (less re-prefill) and drops per-token all-gather overhead.
  * **small TP2 engines** behind sgl-router — 16 rollout GPUs ÷ TP2 = 8 engines,
    halving per-engine concurrency.
  * **2 rollout nodes** (16 GPU, vs the 2n config's 1 node / 8 GPU) for headroom.

Together these took the profiled per-turn engine latency from ~99s (the 2n
dp-attention TP8 baseline) to ~2.5s and roughly halved the rollout step wall
(933→603s), while raising step-0 reward. In fact 603s is the BEST rollout wall the
study measured at any node count. See ``distilled.md`` F4–F9.

  NB on colocate: a colocate config (``w_qwen3_6_swe_colocate_*``) MAY be more
  GPU-efficient at the same node budget — it reclaims this layout's train node,
  which sits idle ~77% of each async iteration. BUT that is NOT established for 3
  nodes: the profile measured the rollout step only (num_rollout=1), never the
  full rollout+serial-train cycle, and its one colocate-3n run hung on a straggler
  (no valid wall). On the rollout wall alone, this noncolocate-3n (603s) actually
  beat the clean colocate runs at 2n (686s) and 4n (692s). Treat colocate-vs-this
  as open until the full step_time is measured.

Topology: 1 training node (8 GPU, Megatron) + 2 rollout nodes (16 GPU, 8× TP2
SGLang engines) = 3 nodes / 24 GPU. Async: rollout and training overlap on
separate GPUs (weights resync every ``update_weights_interval`` rollout steps).

Self-contained (inherits only ``SlimeConfig``): the model / checkpoint / agent
env / algorithm / optimizer settings are spelled out inline rather than inherited
from ``w_qwen3_6_swe_colocate_1n`` — see that file's docstring for the model
rationale (qwen3.5 arch, TP2 cap, EAGLE, MoE dispatch, the one-time torch_dist
conversion). Training rows are the harbor build of SWE-Gym-Lite (``env/harbor.py``).

    EXPERIMENT_CONFIG=w_qwen3_6_swe_noncolocate_3n \
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
_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-gym-lite-noncolocate-3n")

# Datasets (one HF repo each; see configs/datasets.py). Train on swegym_lite;
# eval the full in-distribution held-out slice (30) + USACO transfer (50).
_TRAIN = "swegym_lite"
_EVAL = [("swegym_lite", None), ("usaco", 50)]

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
        "agentic_eval_timeout": 300,
        "agentic_exec_timeout": 120,  # per-command sandbox exec
        "router_policy": "consistent_hashing", 

    }
    metadata_key = "metadata"
    # Harbor build of SWE-Gym-Lite, pulled per-dataset from HF into /data/swe_gym_lite/.
    prompt_data = train_path(_TRAIN)
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

    # ── Rollout engines: 2 rollout nodes, 8× TP2, dp-attention OFF ────────────
    rollout_num_gpus = 16             # 2 rollout nodes (the +1 node vs the 2n config)
    rollout_num_gpus_per_engine = 2   # TP2 → 16 // 2 = 8 engines (sgl-router load-balanced)
    sglang_mem_fraction_static = 0.85
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
    # Subsets built with `python -m agentic_rl.evalset`. Each pass blocks
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
    # scale with the 64k rollout_max_context_len above (preserved from the prior
    # inherited config). Bump to 32768 if you want the budget to track the full
    # 64k context — at the cost of more activation memory per GPU.
    max_tokens_per_gpu = 16384*2
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
    }

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_group = _RUN_TAG
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        """Pull each dataset's HF repo into /data/<key>/ and subsample eval slices.

        Conversion is offline (environment/convert2slime → per-dataset HF repos);
        here we just download. In-distribution eval (swe_gym_lite) shares the train
        repo's tasks/ tree — same converter version keeps task ids aligned.
        Run on the slime-data volume: ``modal run slime/modal_train.py::download_data``.
        """
        for key in {_TRAIN, *(k for k, _ in _EVAL)}:
            pull(key)  # whole repo (train + eval + tasks) into /data/<key>/
        for key, n in _EVAL:
            if n is not None:
                subsample(key, n)


slime = _Slime()
