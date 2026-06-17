"""Qwen3.6-35B-A3B on OpenThoughts-Agent-v1-RL (harbor) — colocated, 2× H200:8.
"""

from pathlib import Path

from configs.base import CHECKPOINTS_PATH, DATA_PATH, run_tag

# Inherit the Qwen3.6-35B-A3B colocated single-node SWE config (model/
# parallelism/engine/optimizer + modal infra); HF_EVAL_REPO is the published
# eval bundle.
from configs.w_qwen3_6_swe_colocate_1n import (  # noqa: F401
    HF_EVAL_REPO,
    _Slime as _SweSlime,
    modal,
)

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-openthoughts-agent-colocate-1n")

_DATA_DIR = f"{DATA_PATH}/openthoughts_agent"

# Eval-set subsets from the published HF evalset; eval task dirs land in their
# own /data subtrees (no overlap with the train tree).
_EVAL_DATASETS = [
    f"{DATA_PATH}/evalsets/v0/usaco_50.jsonl",
    f"{DATA_PATH}/evalsets/v0/openthoughts_tblite.jsonl",
]


class _Slime(_SweSlime):
    # ── Task data: OpenThoughts-Agent (harbor) instead of SWE-Gym-Lite ───────
    prompt_data = f"{_DATA_DIR}/openthoughts_agent.jsonl"

    # ── Colocate / sync ──────────────────────────────────────────────────────
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    rollout_num_gpus_per_engine = 8
    sglang_mem_fraction_static = 0.85

    # ── Eval ─────────────────────────────────────────────────────────────────
    # Held-out transfer: usaco_50 + openthoughts_tblite. Each pass blocks the
    # train loop on the shared engines — keep small (full sweeps: w_qwen3_6_swe_eval).
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

    # Short episodes: task.toml caps the agent leg at 600s, so the budget mostly
    # covers sandbox boot + cold Dockerfile build; verifier is a quick cmp/pytest.
    environment = {
        **_SweSlime.environment,
        "AGENT_TIME_BUDGET_SEC": "1200",
        "AGENT_EVAL_TIMEOUT_SEC": "600",
        "SLIME_AGENT_SANDBOX_CPU": "2",
    }

    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_group = _RUN_TAG

    def download_data(self) -> None:
        """Stage the run's data: OpenThoughts-Agent train set + eval slices.

        Pull open-thoughts/OpenThoughts-Agent-v1-RL from HF, convert into
        /data/openthoughts_agent, re-root task_path under /data, then pull the
        usaco_50 + openthoughts_tblite eval slices.

        Run on the slime-data volume: ``modal run slime/modal_train.py::download_data``.
        """
        import json
        import sys

        sys.path.insert(0, "/root/slime")  # local_slime overlay → async_rl_research
        from async_rl_research.environment.convert2slime.openthoughts_agent import materialize

        converted, skipped = materialize(Path(_DATA_DIR))
        print(f"converted {converted} OpenThoughts-Agent tasks ({skipped} skipped) -> {_DATA_DIR}")

        # Converter writes task_path as "tasks/<id>"; re-root to
        # "openthoughts_agent/tasks/<id>" to resolve under ASYNC_RL_TASK_ROOT=/data.
        jsonl = Path(self.prompt_data)
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        for row in rows:
            task_path = row["metadata"]["task_path"]
            if not task_path.startswith("openthoughts_agent/"):
                row["metadata"]["task_path"] = f"openthoughts_agent/{task_path}"
        jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        print(f"re-rooted {len(rows)} rows' task_path under openthoughts_agent/ -> {jsonl}")

        # Eval slices: usaco_50 + openthoughts_tblite (repo layout mirrors /data).
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=HF_EVAL_REPO,
            repo_type="dataset",
            local_dir=str(DATA_PATH),
            allow_patterns=[
                "evalsets/v0/usaco_50.jsonl",
                "evalsets/v0/openthoughts_tblite.jsonl",
                "usaco/**",
                "openthoughts_tblite/**",
            ],
        )
        print(f"downloaded usaco_50 + openthoughts_tblite eval slices of {HF_EVAL_REPO} -> {path}")


slime = _Slime()
