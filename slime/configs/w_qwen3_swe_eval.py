"""Eval-only run of the SWE agentic-RL setup — no training steps.

Reuses ``w_qwen3_swe_async``. ``num_rollout = 0`` + ``async_mode = False`` routes
through train.py's stock eval-only branch (eval once, exit) — for baselining the
base model or sweeping a checkpoint (train_async.py has no pre-training eval).

Data: the published eval set (https://huggingface.co/datasets/junlin-modal/agentic-rl-evalsets),
pulled onto the /data volume by ``download_data()``. Edit ``_EVAL_DATASETS`` to
select datasets.

    EXPERIMENT_CONFIG=w_qwen3_swe_eval uv run --no-dev modal run slime/modal_train.py::download_data
    EXPERIMENT_CONFIG=w_qwen3_swe_eval uv run --no-dev modal run -d slime/modal_train.py::train
"""

import os
from pathlib import Path

from configs.base import CHECKPOINTS_PATH, DATA_PATH, run_tag
from configs.w_qwen3_swe_async import _Slime, modal

HF_EVAL_REPO = "junlin-modal/agentic-rl-evalsets"

# W&B run name; run_tag() appends a launch timestamp so dumps don't collide.
_RUN_TAG = run_tag("qwen3-30b-a3b-swe-eval")

# Eval-set version, switched with EVAL_VERSION on launch (download_data pulls the
# whole HF repo, so both versions live on the volume; only paths change).
#   v0 = swe_gym_lite_100 (train distribution) / swebench_verified_100 /
#        openthoughts_tblite / usaco_50
#   v1 = swebench_verified / swebenchpro / swebench_multilingual / terminal_bench
EVAL_VERSION = os.environ.get("EVAL_VERSION", "v1")
_EVAL_SUBSETS = {
    "v0": ["swe_gym_lite_100", "swebench_verified_100", "openthoughts_tblite", "usaco_50"],
    "v1": ["swebench_verified", "swebenchpro", "swebench_multilingual", "terminal_bench"],
}[EVAL_VERSION]
# Comment names out above to eval fewer subsets.
_EVAL_DATASETS = [f"{DATA_PATH}/evalsets/{EVAL_VERSION}/{name}.jsonl" for name in _EVAL_SUBSETS]


class _SlimeEval(_Slime):
    # num_rollout=0 + async_mode=False routes through train.py's eval-only
    # branch (load weights -> push to sglang -> eval -> exit).
    async_mode = False
    num_rollout = 0
    eval_interval = 1  # any non-None value arms the eval-only branch

    # slime derives train_iters from num_rollout, so eval-only computes
    # lr_decay_steps == 0 and trips Megatron's `assert lr_decay_steps > 0`. Pin
    # a dummy 1-iter schedule; no optimizer step runs.
    lr_decay_iters = 1

    # Point `load` at a Megatron dir to eval a trained checkpoint; else the base
    # hf_checkpoint weights are evaluated.
    # load = f"{CHECKPOINTS_PATH}/<run>/..."

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
