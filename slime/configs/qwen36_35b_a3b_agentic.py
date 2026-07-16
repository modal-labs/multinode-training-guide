"""Qwen3.6-35B-A3B (MoE + GDN) — fully-async agentic SWE RL on SWE-rebench-V2 (FULL dataset).

Base-model upgrade from GLM-4.7-Flash (59% SWE) → Qwen3.6-35B-A3B (73% SWE), same MoE-A3B class.
Reuses our agentic_rl rollout (mini-swe agent in Modal sandboxes + baseline-relative dense reward +
windowed-FIFO staleness), wired via slime's custom-* hooks. Model/parallelism mirror the upstream
examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh (qwen3.5 spec, router fp32, no MTP, no EAGLE).

Non-colocate fully-async on whfb H200: 4 trainer nodes (TP2/CP8 → DP2) + 8 rollout engines = 12 nodes.
Rollout concurrency maxed to the staleness pool (independent of the trainer); trainer DP2 for ~1.8× step.
"""

from configs.base import ModalConfig, SlimeConfig, DATA_PATH, CHECKPOINTS_PATH, HF_CACHE_PATH

modal = ModalConfig(
    gpu="H200",
    # cloud unset (was "whfb") — the whfb H200 pool degraded (GPU-less nodes, 3 launches in a row);
    # let Modal auto-pick a healthy cloud. EFA (AWS) is the documented RDMA alt and efa_enabled is already on.
    memory=(1024, int(2 * 1024 * 1024)),
    local_slime="/home/ec2-user/nan_wonderland/slime",
    # EAGLE/MTP spec-decode is OFF — MEASURED net SLOWDOWN on this workload, not just the crash. We fixed the
    # GDN q.shape crash two ways (#28096 patch + speculative_draft_attention_backend=fa3 OR =triton; fa3 also
    # needs the buggy flashinfer-draft avoided). With accept_len a healthy ~2.5, fa3-EAGLE was STILL ~17-43%
    # slower per-request than no-EAGLE at matched concurrency, because (a) the rollout is high-concurrency
    # throughput-bound (spec-decode helps latency, hurts throughput) and (b) the A3B+GDN main decode is already
    # cheap so the all-full-attention MTP draft+verify overhead outweighs acceptance. Re-enable only if decode
    # latency ever becomes the bottleneck. Patch kept at patches/sglang_eagle_gdn_28096.patch; fa3 was the right
    # draft backend (vs the flashinfer default that crashes / the triton draft that's even slower).
    image_run_commands=[
        f"rm -rf {HF_CACHE_PATH}",
        "apt-get update && apt-get install -y --no-install-recommends rdma-core libibverbs1 ibverbs-providers",
        "uv pip install --system modal mini-swe-agent datasets",
    ],
    image_env={"MSWEA_SILENT_STARTUP": "1"},
)


class _Slime(SlimeConfig):
    slime_model_script = "scripts/models/qwen3.5-35B-A3B.sh"  # Qwen3.6-35B-A3B reuses the qwen3.5 spec
    async_mode = True
    environment = {**SlimeConfig.environment, "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
                   "NVSHMEM_DISABLE_NCCL": "1"}

    # ── Model / checkpoint ──────────────────────────────────────────────────────
    # FP8 ROLLOUT was TRIED (2026-07-01) and REVERTED: the shared-expert block-quant blocker was fixed
    # (SGLANG_SHARED_EXPERT_TP1 patch), but the FP8+GDN decode then STALLED episodes (examined=0 for 26+ min,
    # requests hang) — worse than bf16. Needs sglang-level FP8+GDN debugging; not a quick win. Staying bf16.
    hf_checkpoint = "Qwen/Qwen3.6-35B-A3B"
    ref_load = f"{CHECKPOINTS_PATH}/Qwen3.6-35B-A3B_torch_dist_tp2pp2"  # torch_dist reshards TP2/PP2 → TP2/PP1/CP8 on load
    # NOTE: Qwen3.6-35B-A3B has NO MTP (the qwen3.5 spec defines none) — so no mtp_* args at all.

    # ── Infrastructure (non-colocate: 4 trainer nodes + 8 rollout engines = 12) ───────
    actor_num_nodes = 4          # 32 GPU at TP2×PP1×CP8 (=16) → DP2: ~1.8× faster train step (248→124 microbatch/
                                 # replica). Rollout is latency-floored after the concurrency bump, so spend the
                                 # extra hardware on the trainer (which is on the critical path), not more engines.
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 64        # 8 sglang engines × 8 GPU (KV ~8% at 4× concurrency → engines not the bottleneck)

    # ── Parallelism: TP2/PP1/CP8, EP8/ETP1 (upstream Qwen3.6 SWE recipe). TP×PP×CP = 2×1×8 = 16 = 2 nodes.
    tensor_model_parallel_size = 2
    pipeline_model_parallel_size = 1
    context_parallel_size = 8
    sequence_parallel = True
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1

    # ── Data: QWEN CURRICULUM — the 703 DEMONSTRATED-LEARNABLE prompts (partial solve-rate 0.12-0.88,
    # aggregated from 27k episodes across the flat runs). RL only learns where positive examples exist: 64% of
    # the full set was 0/8 (no positive → no gradient, wasted rollouts) and 16% was 8/8 (mastered, no gradient).
    # Concentrating on the 20% partial (positive+negative → real advantage) is the fix for the flat learning.
    prompt_data = f"{DATA_PATH}/swe_rebench_v2_python/train_curriculum_qwen.jsonl"
    input_key = "input"
    label_key = "label"
    apply_chat_template = False  # the agent harness builds its own conversation
    rm_type = "deepscaler"
    rollout_shuffle = True

    # ── Rollout / sglang ────────────────────────────────────────────────────────
    rollout_function_path = "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
    custom_generate_function_path = "agentic_rl.generate.generate"
    custom_rollout_log_function_path = "agentic_rl.metrics.log_rollout_data"
    rollout_num_gpus_per_engine = 8
    # ROOT CAUSE of slow decode (source-verified): each SGLang worker is ONE uvloop/tokenizer event loop that
    # accepts+tokenizes+streams for every connection → saturates ~120 conns → 8 engines cap ~960 concurrent →
    # decode batch ~14/rank (only 11% of the scheduler's max_running_requests=132/rank — so the batch is
    # CONNECTION-limited, not scheduler-limited). Push past ~960 → TCP accept RST → router "Failed to send"
    # storm → effective concurrency collapses. The intended widen-the-door knob (sglang_tokenizer_worker_num>1)
    # is BROKEN in Ray/Modal: uvicorn multiprocess workers die on `os.fdopen(stdin)` → [Errno 9] Bad file
    # descriptor (no stdin in a Ray worker). So the ~960 ceiling stands here. The clean speed lever is instead
    # FP8 rollout — halves param-read bandwidth → ~2× decode at the SAME batch, no need to break the ceiling.
    sglang_server_concurrency = 128    # =1024 threads ≈ the ~960 connection ceiling (proven storm-free)
    rollout_max_context_len = 131072   # 128k served context (user choice; upstream uses 96k)
    rollout_max_prompt_len = 16384
    rollout_max_response_len = 32768   # 32k per-turn gen (Qwen3.6 reasoning needs room; upstream value)
    rollout_temperature = 1.0
    rollout_top_p = 1.0
    sglang_mem_fraction_static = 0.7
    sglang_enable_dp_attention = True
    sglang_dp_size = 8
    sglang_ep_size = 8
    sglang_enable_dp_lm_head = True
    sglang_moe_dense_tp_size = 1
    sglang_max_running_requests = 2048  # engine admission cap (sglang caps to ~132/rank; never the binding limit)
    # NOTE: sglang_tokenizer_worker_num>1 (the widen-the-door fix) is BROKEN in Ray/Modal — uvicorn multiprocess
    # workers die on os.fdopen(stdin) [Errno 9]. If we ever want to break the ~960 ceiling, options are: patch
    # sglang to launch the http server with stdin=/dev/null, use Granian/http2, or run more (smaller) engines.
    sglang_cuda_graph_bs = [1, 2, 4, 8] + list(range(16, 257, 8))
    sglang_mamba_scheduler_strategy = "extra_buffer"  # GDN/linear-attention (mamba-family) scheduler
    use_fault_tolerance = True

    # ── MTP/EAGLE spec-decode: OFF (measured net slowdown — see ModalConfig note). To re-enable for a future
    # latency-bound config: sglang_speculative_algorithm="EAGLE", num_steps 2, eagle_topk 1, num_draft_tokens 3,
    # sglang_speculative_draft_attention_backend="fa3" (NOT flashinfer→crash, NOT triton→slow), + the #28096
    # patch in patch_files + SGLANG_ENABLE_SPEC_V2=1.
    #
    # HiCache: tier the radix prefix cache GPU↔host. Validated on GDN (prefix_cache_hit 0.60→0.65); helps
    # only the 10/40 full-attention layers (linear layers carry a non-cacheable recurrent state).
    sglang_enable_hierarchical_cache = True
    sglang_hicache_ratio = 1.0
    sglang_hicache_write_policy = "write_through"
    sglang_page_size = 64

    # Agent episode limits (read by agentic_rl.generate).
    custom_config_path = {
        # max_steps 64: turns rarely bind (measured step_cap trunc 1.1%) — episodes hit the 1800s TIME cap
        # first (~42% time-trunc). So raising turns wouldn't help; time_cap is the lever if we want fewer
        # cuts. (NOTE: the kept-collapse that earlier made me revert 100→64 was a cold-start PHANTOM, not the
        # turn count — kept recovers to ~42% by rollout completion. 64 kept for now; turns aren't binding.)
        "agentic_max_steps": 64,
        "agentic_episode_timeout": 3600,  # 30→60 min: at conc 128 episodes take ~25 min mean (slow decode), so
                                          # a 30-min cap TIME-truncated ~47% mid-trajectory — meaningless to train
                                          # on. 60 min lets episodes reach a real terminal state (submit or the
                                          # 64-turn cap) → complete trajectories. Cost: rollout ~2× slower. The
                                          # sandbox lifetime + hard_cap auto-scale from this (generate.py:183,318).
        "agentic_exec_timeout": 120,
        "agentic_grade_timeout": 600,
        "agentic_query_timeout": 1800,
        "agentic_max_boot_retries": 3,
        "agentic_max_format_errors": 64,  # consecutive tool-call misses before episode dies (Tmax uses 64)
        "agentic_ramp_window": 30.0,      # cold-start stagger (s): jitter episode starts so the ~1024-episode
                                          # herd @ conc 128 spreads over 30s instead of bursting the router.
        "router_policy": "consistent_hashing",
        "agentic_overlong_max": 122880,
        "agentic_overlong_cache": 40960,
        "agentic_overlong_correct_floor": 0.5,
    }

    # ── Training ────────────────────────────────────────────────────────────────
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 16384         # 128k / CP8 (one full sequence per rank)
    log_probs_chunk_size = 1024        # small chunk = memory headroom for GDN long-seq entropy backward (issue #1523)
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    moe_token_dispatcher_type = "flex"
    moe_enable_deepep = True
    # Off-policy RL occasionally produces a NaN/inf grad; let slime skip that step instead of Megatron raising.
    no_check_for_nan_in_loss_and_grad = True

    # ── Optimizer ───────────────────────────────────────────────────────────────
    optimizer = "adam"
    lr = 2e-6                          # the value the ONE prior curriculum climb used (0.42→0.50). With the
                                       # curriculum's dense learnable signal (vs sparse full-data), 2e-6 gives
                                       # real updates without overshoot risk. (full-data lr 1e-6→3e-6 was flat
                                       # because the SIGNAL was too sparse, not because the lr was too low.)
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True
    clip_grad = 1.0

    # ── Algorithm: GRPO. Train on the rollout (sglang) logprobs directly as the PPO reference (Tmax's
    # use_vllm_logprobs). MEASURED on the big run: ois=1.0, tis_abs=0.02, ppo_kl=0.0016 → rollout≈trainer.
    # So get_mismatch_metrics is OFF (it cost one extra ~10-min trainer fwd/step just to REPORT ois/tis);
    # ppo_kl + pg_clipfrac (logged for free in the base GRPO loss) are the ongoing mismatch proxy instead.
    advantage_estimator = "grpo"
    use_kl_loss = False
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_tis = False
    use_rollout_logprobs = True
    get_mismatch_metrics = False       # was True; ois=1.0 confirmed → drop the extra fwd. Watch ppo_kl/pg_clipfrac.
    custom_tis_function_path = "slime.backends.megatron_utils.loss.vanilla_tis_function"  # unused while both flags off
    tis_clip = 2.0
    tis_clip_low = 0.5
    balance_data = True
    update_weights_interval = 1
    rollout_pause_generation_mode = "in_place"
    rollout_max_staleness = 4          # back to 4 (was bumped to 6 only for the reverted 100-turn episodes).

    # ── Dynamic sampling (DAPO): discard zero-variance groups so every trained group carries gradient.
    # Async build in fully_async_rollout: filter zero-std groups from the pool, keep pulling to
    # `rollout_batch_size` passing groups. Over-generation is free (the windowed-FIFO already runs the pool
    # ahead of the trainer — no extra rounds). dynamic_sampling/raw_reward_all logs the UNBIASED pre-filter
    # mean reward (slime's rollout/raw_reward reads high once filtering is on, since it's the kept subset).
    dynamic_sampling_filter_path = "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"

    # ── Sizing — 8 samples/prompt × 32 prompts = 256 episodes/step ────────────
    # GROUP SIZE cut 16→8: smaller GRPO group → each group completes when its slowest-of-8 (not slowest-of-16)
    # finishes → less straggler gating → faster rollout. rollout_batch 32 (batch cut earlier for 2× fresher
    # updates). global_batch MUST = rollout_batch × n_samples // num_steps (asserted) = 32×8 = 256.
    num_rollout = 400
    rollout_batch_size = 32
    n_samples_per_prompt = 8           # GRPO group size (was 16) — smaller group, less straggler gating
    global_batch_size = 256

    # ── Checkpointing (resume across the 24h Modal cap + preserve the model) ─────
    save = f"{CHECKPOINTS_PATH}/qwen36_35b_a3b_agentic"
    save_interval = 10                 # every 10 rollouts (~resume granularity)
    # load = save  # set on RESUME runs (loads latest checkpoint if present, else ref_load)

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = "fully-async-rl-modal"  # same project as GLM-4.7-Flash agentic, for side-by-side comparison
    wandb_group = "qwen3.6-35b-a3b-swe-rebench-agentic-async"
    disable_wandb_random_suffix = True

    def download_data(self) -> None:
        """SWE-rebench-V2 → FULL prompt_data jsonl (python/pytest, >=1 FAIL_TO_PASS, good_instances filter).
        Same as the GLM config's download_data (no curriculum subset — difficulty is handled online by
        dynamic sampling)."""
        import json
        import os

        from datasets import load_dataset

        BAD = {"dhi__mikeio-227", "dhi__mikeio-442", "dhi__mikeio-690"}
        data_dir = f"{DATA_PATH}/swe_rebench_v2_python"
        out = f"{data_dir}/train.jsonl"
        good_path = f"{data_dir}/good_instances.json"
        good = set(json.load(open(good_path))) if os.path.exists(good_path) else None
        keys = ("instance_id", "repo", "base_commit", "image_name", "problem_statement",
                "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "install_config", "language")
        os.makedirs(data_dir, exist_ok=True)

        def keep(r) -> bool:
            if r["language"] != "python" or r["install_config"]["log_parser"] != "parse_log_pytest":
                return False
            if r["instance_id"] in BAD:
                return False
            if good is not None and r["instance_id"] not in good:
                return False
            return len(r["FAIL_TO_PASS"]) >= 1

        ds = load_dataset("nebius/SWE-rebench-V2", split="train").filter(keep)
        with open(out, "w") as f:
            for r in ds:
                row = {"input": r["problem_statement"], "label": r["instance_id"], "metadata": {k: r[k] for k in keys}}
                f.write(json.dumps(row) + "\n")
        print(f"download_data: wrote {len(ds)} filtered instances to {out}")


slime = _Slime()
