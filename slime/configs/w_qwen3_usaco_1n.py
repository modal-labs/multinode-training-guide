"""Qwen3-30B-A3B USACO (harbor) agentic RL — colocated, single node (1× H200:8).

Delta on ``w_qwen3_swe_colocate_1n``: same model/parallelism/optimizer/topology,
only the task changes. Harbor rows (``env/harbor.py``) boot the task's Dockerfile
sandbox (all USACO tasks share one ``python:3.13-slim`` image -> one cached
build), mini-swe-agent writes ``solution.py``, in-sandbox ``tests/test.sh``
grades, reward read from ``/logs/verifier/reward.txt``. ``download_data``
sparse-clones harbor-datasets, converts into ``/data/usaco``, re-roots
``task_path`` to ``usaco/tasks/<id>`` under ``ASYNC_RL_TASK_ROOT=/data``, then
pulls the usaco eval slice of the published HF bundle.

Sanity check before burning GPU-hours (expect reward=1.0)::

    uv run --with modal python -m async_rl_research.environment.harbor \
        <local-out>/usaco.jsonl --task-root <local-out> --limit 3
"""

from pathlib import Path

from configs.base import CHECKPOINTS_PATH, DATA_PATH, run_tag
from configs.w_qwen3_swe_eval import HF_EVAL_REPO

# Inherit the colocated single-node SWE config (model/parallelism/optimizer/
# engine + modal infra).
from configs.w_qwen3_swe_colocate_1n import (  # noqa: F401
    _Slime as _SweSlime,
    modal,
)

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3-30b-a3b-usaco-colocate-1n")

# Eval-set subsets from the published HF evalset; usaco_50's task dirs reuse the
# train conversion's.
_EVAL_DATASETS = [
    f"{DATA_PATH}/evalsets/v0/usaco_50.jsonl",
]


class _Slime(_SweSlime):
    # ── Task data: harbor-converted USACO instead of SWE-Gym-Lite ───────────
    prompt_data = f"{DATA_PATH}/usaco/usaco.jsonl"

    # ── Eval ─────────────────────────────────────────────────────────────────
    # Each pass blocks the train loop on the shared engines — keep small (full
    # sweeps: w_qwen3_swe_eval).
    eval_interval = 20
    # metadata_overrides keeps per-dataset attribution in the flattened dump.
    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": [
            {
                "name": Path(p).stem,
                "path": p,
                "metadata_overrides": {"eval_dataset": Path(p).stem},
            }
            for p in _EVAL_DATASETS
        ],
    }

    # Shorter than SWE: task.toml caps the agent leg at 600s; the budget mostly
    # covers sandbox/provisioning + the verifier's uv+pytest install.
    environment = {
        **_SweSlime.environment,
        "AGENT_TIME_BUDGET_SEC": "900",
        "AGENT_EVAL_TIMEOUT_SEC": "900",
        # Full cores so a correct-but-slow USACO solution isn't graded TLE.
        "SLIME_AGENT_SANDBOX_CPU": "2",
    }

    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG

    def download_data(self) -> None:
        """Stage the run's data: train set from GitHub, eval set from HF.

        Clone harbor-datasets (usaco subtree), convert into /data/usaco, then
        pull the usaco eval slice (its task dirs overlap the train output).
        """
        import json
        import subprocess
        import sys
        import tempfile

        sys.path.insert(0, "/root/slime")  # local_slime overlay → async_rl_research
        from async_rl_research.environment.convert2slime.harbor import main as convert_main

        clone = tempfile.mkdtemp(prefix="harbor-datasets-")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "https://github.com/laude-institute/harbor-datasets.git", clone],
            check=True,
        )
        subprocess.run(["git", "-C", clone, "sparse-checkout", "set", "datasets/usaco"], check=True)
        convert_main(
            ["--tasks-dir", f"{clone}/datasets/usaco", "--out-dir", f"{DATA_PATH}/usaco", "--name", "usaco"]
        )

        # Converter writes task_path as "tasks/<id>"; re-root to
        # "usaco/tasks/<id>" to resolve under ASYNC_RL_TASK_ROOT=/data.
        jsonl = Path(f"{DATA_PATH}/usaco/usaco.jsonl")
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        for row in rows:
            task_path = row["metadata"]["task_path"]
            if not task_path.startswith("usaco/"):
                row["metadata"]["task_path"] = f"usaco/{task_path}"
        jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"re-rooted {len(rows)} rows' task_path under usaco/ -> {jsonl}")

        # Eval set: usaco-only slice. Pulled after conversion so the eval-pinned
        # copies win if they diverge from the train dirs.
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=HF_EVAL_REPO,
            repo_type="dataset",
            local_dir=str(DATA_PATH),
            allow_patterns=["evalsets/v0/usaco_50.jsonl", "usaco/**"],
        )
        print(f"downloaded usaco eval slice of {HF_EVAL_REPO} -> {path}")


slime = _Slime()
