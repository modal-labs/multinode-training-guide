"""Eval-only run of Qwen3.6-35B-A3B on Frontier-CS algorithmic — no training.

Reuses ``w_qwen3_6_frontier_cs_noncolocate_3n`` scaled to the 4-rollout-node
fleet of the swe_rebench_v2/frontier-cs 5n configs: 1 train node + 4 rollout
nodes (32 GPU → 16× TP2 / dp-off SGLang engines behind sgl-router) = 5 nodes.
(Previously this eval reused the colocate_2n config; it now matches the
noncolocate node setup the frontier-cs training runs use.) The parent is
``async_mode = True``, so this config flips it to False — ``num_rollout = 0``
then routes through train.py's eval-only branch (eval once, exit) — for
baselining the base model or sweeping a checkpoint. Evals the full 38-problem
held-out slice (``frontier_cs_eval.jsonl``). Inherits the parent's
``download_data`` (jsonl + harbor task trees) and verifier-server wiring.

Prereqs (see agentic_rl/environment/convert2slime/README.md): the converted
data published to the per-dataset HF repo (``frontier-cs``) and pulled by
``download_data`` onto slime-data (jsonl + tasks/ + problems/). The verifier server
auto-boots per worker (mounts slime-data) unless a pre-deployed FRONTIER_CS_JUDGE_URL is set.

    EXPERIMENT_CONFIG=w_qwen3_6_frontier_cs_eval uv run --no-dev modal run slime/modal_train.py::download_data
    EXPERIMENT_CONFIG=w_qwen3_6_frontier_cs_eval uv run --no-dev modal run -d slime/modal_train.py::train
"""

from configs.base import CHECKPOINTS_PATH, run_tag
from configs.datasets import eval_datasets
from configs.w_qwen3_6_frontier_cs_noncolocate_3n import _Slime, modal  # noqa: F401

_RUN_TAG = run_tag("qwen3.6-35b-a3b-frontier-cs-eval")

# Full held-out Frontier-CS eval slice (/data/frontier_cs/eval.jsonl); inherits the
# parent's download_data (pulls frontier_cs incl. problems/ + usaco).
_EVAL = [("frontier_cs", None)]


class _SlimeEval(_Slime):
    # The 3n parent is async; eval-only needs the sync train.py path so that
    # num_rollout=0 routes through the eval-only branch.
    async_mode = False
    num_rollout = 0
    eval_interval = 1  # any non-None value arms the eval-only branch
    sglang_server_concurrency = 32

    # Scale the parent's 2 rollout nodes to the 4-node fleet the 5n configs run:
    # 32 GPUs // 2 per engine = 16 TP2 engines (sgl-router load-balanced).
    rollout_num_gpus = 32

    # slime derives train_iters from num_rollout; pin a dummy 1-iter schedule so
    # Megatron's `assert lr_decay_steps > 0` passes (no optimizer step runs).
    lr_decay_iters = 1

    # Competitive programming is reasoning-heavy: the per-turn cap guillotines
    # Qwen3.6's first <think> mid-thought before it emits any tool call
    # (finish=length with 0 tool calls → ContextLengthExceeded, reward 0). The 27B
    # run 20260707-183540 still lost 22/38 problems at 24576; use the same 49152
    # cap here (see w_qwen3_6_27b_frontier_cs_eval).
    eval_max_response_len = 49152  # per-turn generation cap (8192 → 24576 → 49152)
    rollout_max_context_len = 65536 * 2  # total multi-turn budget (was 32768)

    # Longer per-turn thinks stretch episodes past the parent's 1800s budget (the
    # 27B run's top scorer was cut at 1828s). Eval-only relaxation; also pin the
    # router/exec knobs the 5n lineage uses (the 3n parent doesn't set them).
    custom_config_path = {
        **_Slime.custom_config_path,
        "agentic_episode_timeout": 3600,
        "agentic_exec_timeout": 120,
        "router_policy": "consistent_hashing",
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
