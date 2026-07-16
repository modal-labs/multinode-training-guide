"""Qwen3.6-35B-A3B on Frontier-CS algorithmic (competitive programming) — colocated, two nodes (2× H200:8).

Frontier-CS runs as **harbor tasks** (per-task Dockerfile + in-sandbox
``tests/evaluate.py``). The agent writes ``/app/solution.cpp`` and iterates with
``bash /app/submit.sh``, which POSTs to a **verifier server** (Node + go-judge)
booted once per worker by ``FrontierCsEnv`` (``environment/verifier_server/``). The
2.5 GB of testdata rides on slime-data (``frontier_cs/problems/``); the judge mounts
slime-data and reads it. Final grade = the judge's score on the final solution.cpp
(``rewards.py`` shapes it; default fractional).

Inherits the Qwen3.6 SWE 1n config (model / training parallelism / agent env /
optimizer / modal infra) and changes only: task data, eval, the verifier-server
env, download_data, the two-node colocate layout, and the SGLang rollout engine
recipe. (Inherits the SWE 1n base directly — NOT the frontier_cs 1n config — to
keep the inheritance shallow; the frontier-cs deltas are duplicated here on
purpose.) Rows carry ``task_type=frontier_cs`` (set by the converter), so
``FrontierCsEnv`` runs them — no env wiring needed here.

Rollout engine = the **TP2 / dp-off** recipe from ``rollout_profile/tp2.py`` (the
SGLang perf-study champion), scaled to two colocate nodes (cf.
``rollout_profile/colo2n_tp2.py``): ``rollout_num_gpus_per_engine = 2`` over the
16 colocate GPUs -> 8 TP2 engines (sgl-router load-balanced), with dp-attention
OFF => pure tensor parallelism (its ``dp_size`` / ``ep_size`` / ``dp_lm_head``
companions nulled). It deliberately does NOT use the TP4 + dp-attention layout
that ``w_qwen3_6_swe_colocate_2n`` ships.

Prereqs before launch (see agentic_rl/environment/convert2slime/README.md):
publish the frontier-cs dataset repo (jsonl + tasks/ + the 2.5 GB problems/) to HF;
``download_data`` pulls all of it onto slime-data.
"""

import os

from configs.base import CHECKPOINTS_PATH, run_tag
from configs.datasets import eval_datasets, pull, subsample, train_path
from configs.w_qwen3_6_swe_colocate_1n import _Slime as _SweSlime, modal  # noqa: F401

_RUN_TAG = run_tag("qwen3.6-35b-a3b-frontier-cs-colocate-2n")

# Train on frontier_cs; eval the full held-out frontier_cs slice + USACO transfer (50).
_TRAIN = "frontier_cs"
_EVAL = [("frontier_cs", None), ("usaco", 50)]


class _Slime(_SweSlime):
    # ── Task data: Frontier-CS algorithmic (harbor; task_type=frontier_cs) ────
    prompt_data = train_path(_TRAIN)

    # ── Colocate / sync: two nodes (DP doubles; TP/CP/EP/PP inherited from SWE 1n) ──
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8

    # ── SGLang rollout engine: TP2 / dp-off recipe (rollout_profile/tp2.py) ────
    # 16 colocate GPUs / 2-per-engine -> 8 TP2 engines (sgl-router load-balanced).
    # dp-attention off = pure tensor parallelism, so null its DP/EP companions
    # (the companion-safety rule in rollout_profile/_base.py). EAGLE stays on
    # (inherited). rollout_num_gpus is auto = actor_num_nodes * 8 under colocate.
    rollout_num_gpus_per_engine = 2
    sglang_enable_dp_attention = False
    sglang_dp_size = None
    sglang_ep_size = None
    sglang_enable_dp_lm_head = None
    # Colocate: engine KV coexists with Megatron weights/optimizer on the same
    # GPUs. 0.6 matches the TP2 colocate precedent (colo2n_tp2); nudge toward 0.65
    # for the 32k context if startup leaves headroom (down if it OOMs at capture).
    sglang_mem_fraction_static = 0.6

    # ── Eval ──────────────────────────────────────────────────────────────────
    eval_interval = 10
    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": eval_datasets(_EVAL),
    }

    # ── Environment: verifier-server wiring on top of SWE's ───────────────────
    # FrontierCsEnv boots the verifier server (vm_runtime Modal Sandbox) once per
    # worker and exports FRONTIER_CS_JUDGE_URL. Set FRONTIER_CS_JUDGE_URL here to
    # point at a pre-deployed judge instead (skips the per-worker boot).
    # ASYNC_RL_REWARD_SHAPE picks the central reward shape (fractional|binary|thresholded).
    environment = {
        **_SweSlime.environment,
        "FRONTIER_CS_JUDGE_URL": os.environ.get("FRONTIER_CS_JUDGE_URL", ""),
        "ASYNC_RL_REWARD_SHAPE": os.environ.get("ASYNC_RL_REWARD_SHAPE", "fractional"),
    }

    # ── WandB / dumps ──────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG
    save_debug_rollout_data = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"

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
