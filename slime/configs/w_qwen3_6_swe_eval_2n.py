"""Eval-only run of the Qwen3.6-35B-A3B SWE agentic-RL setup — no training.
"""

import os
from pathlib import Path

from configs.base import CHECKPOINTS_PATH, DATA_PATH, run_tag
from configs.w_qwen3_6_swe_colocate_2n import _Slime, modal

HF_EVAL_REPO = "junlin-modal/agentic-rl-evalsets"

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-eval")

EVAL_VERSION = os.environ.get("EVAL_VERSION", "v1")
_EVAL_SUBSETS = {
    "v0": ["swe_gym_lite_100", "swebench_verified_100", "openthoughts_tblite", "usaco_50"],
    # "v1": ["swebench_verified", "swebenchpro", "swebench_multilingual", "terminal_bench"],
    "v1": ["terminal_bench"],
}[EVAL_VERSION]
# Comment names out above to eval fewer subsets.
_EVAL_DATASETS = [f"{DATA_PATH}/evalsets/{EVAL_VERSION}/{name}.jsonl" for name in _EVAL_SUBSETS]


class _SlimeEval(_Slime):
    num_rollout = 0
    eval_interval = 1  # any non-None value arms the eval-only branch
    sglang_server_concurrency=128

    lr_decay_iters = 1
    sglang_mem_fraction_static = 0.85
    rollout_max_context_len = 32768 * 2

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

    wandb_group = _RUN_TAG
    # Override the base's dump path so rollouts land under THIS run's tag (the
    # eval W&B name), not the inherited train-config tag.
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )

    def download_data(self) -> None:
        """Pull the published eval set from HF straight onto the data volume."""
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=HF_EVAL_REPO, repo_type="dataset", local_dir=str(DATA_PATH))
        print(f"downloaded {HF_EVAL_REPO} -> {path}")


slime = _SlimeEval()
