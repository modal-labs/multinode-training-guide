"""Eval-only run of Qwen3.6-27B (DENSE) on Frontier-CS algorithmic — no training.

Reuses ``w_qwen3_6_27b_frontier_cs_noncolocate_5n`` (already ``async_mode = False``),
so eval runs on that config's 2-train + 4-rollout topology (the swe_rebench_v2 27B
node setup: TP4×CP2→DP2 training, 16× TP2 / dp-off rollout engines at
mem_fraction 0.85) — the same footprint the frontier-cs training run will use, so
this eval doubles as an infra shakedown. ``num_rollout = 0`` routes through train.py's eval-only branch (eval
once, exit) — for baselining the base model or sweeping a checkpoint. Evals the full
held-out slice (``frontier_cs/eval.jsonl``). Inherits the parent's ``download_data``
(jsonl + harbor task trees + the 2.5 GB problems/) and verifier-server wiring.

Prereqs (see agentic_rl/environment/convert2slime/README.md): the converted data
published to the per-dataset HF repo (``frontier-cs``) and pulled by ``download_data``
onto slime-data (jsonl + tasks/ + problems/). The verifier server auto-boots per
worker (mounts slime-data) unless a pre-deployed FRONTIER_CS_JUDGE_URL is set.

    EXPERIMENT_CONFIG=w_qwen3_6_27b_frontier_cs_eval uv run --no-dev modal run slime/modal_train.py::download_data
    EXPERIMENT_CONFIG=w_qwen3_6_27b_frontier_cs_eval uv run --no-dev modal run -d slime/modal_train.py::train
"""

import os

from configs.base import CHECKPOINTS_PATH, run_tag
from configs.datasets import eval_datasets
from configs.w_qwen3_6_27b_frontier_cs_noncolocate_5n import _Slime, modal  # noqa: F401

_RUN_TAG = run_tag("qwen3.6-27b-frontier-cs-eval")

# Full held-out Frontier-CS eval slice (/data/frontier_cs/eval.jsonl); inherits the
# parent's download_data (pulls frontier_cs incl. problems/ + usaco).
_EVAL = [("frontier_cs", None)]


class _SlimeEval(_Slime):
    # async_mode=False already, so num_rollout=0 routes through the eval-only branch.
    num_rollout = 0
    eval_interval = 1  # any non-None value arms the eval-only branch
    sglang_server_concurrency = 32

    # slime derives train_iters from num_rollout; pin a dummy 1-iter schedule so
    # Megatron's `assert lr_decay_steps > 0` passes (no optimizer step runs).
    lr_decay_iters = 1


    # Competitive programming is reasoning-heavy: the per-turn cap guillotines
    # Qwen3.6's first <think> mid-thought before it emits any tool call
    # (finish=length with 0 tool calls → ContextLengthExceeded, reward 0). At 8192
    # that killed 32/38 problems; at 24576 still 22/38 (run 20260707-183540); at
    # 49152 still 12/38 (run 20260707-192513, all dead at exactly 49152 ~4 min in).
    eval_max_response_len = 65536  # per-turn generation cap (8192 → 24576 → 49152 → 65536)
    rollout_max_context_len = 65536 * 2  # total multi-turn budget (was 32768)

    # Longer per-turn thinks stretch episodes: the top scorer of the 24576 run was
    # already cut by the parent's 1800s episode budget (elapsed 1828s). Eval-only
    # relaxation; training keeps 1800s.
    #
    # Forced think closure (s1-style budget forcing, agentic_rl/model.py): a turn
    # that hits the per-turn cap with no tool call gets its dangling <think> closed
    # (injected, loss-masked ids) and continues the SAME turn under a 4096-token
    # budget, instead of rolling back into a ContextLengthExceeded null. Default ON
    # here for the A/B vs the 20260707-224556 baseline (identical config, flag off:
    # 11/38 null, reward 0.182). THINK_CLOSURE=0 relaunches the baseline behavior;
    # the parent bakes THINK_CLOSURE into the image env so the in-container config
    # re-import resolves the same value.
    custom_config_path = {
        **_Slime.custom_config_path,
        "agentic_episode_timeout": 3600,
        "agentic_close_think_on_length": os.environ.get("THINK_CLOSURE", "1") == "1",
        "agentic_max_think_closures": 2,
        "agentic_think_closure_budget": 4096,
    }

    # Point `load` at a Megatron dir to eval a trained checkpoint; else the base
    # hf_checkpoint weights are evaluated.
    # load = f"{CHECKPOINTS_PATH}/<run>/..."

    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": eval_datasets(_EVAL),
    }

    wandb_group = _RUN_TAG
    save_debug_rollout_data = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"


slime = _SlimeEval()
