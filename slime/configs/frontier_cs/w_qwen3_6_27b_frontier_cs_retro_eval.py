"""Full 38-problem eval for a Phase-2 retro checkpoint lineage.

    RETRO_EVAL_RUN_TAG=qwen3.6-27b-frontier-cs-retro-a-final-<stamp> \
    EXPERIMENT_CONFIG=frontier_cs.w_qwen3_6_27b_frontier_cs_retro_eval \
    MODAL_ENVIRONMENT=junlin-dev WANDB_PROJECT=Modal \
      uv run --no-dev modal run slime/modal_train.py::train

``load`` points at the run's save root; Slime resolves its latest checkpoint.
Use the same config/seed for Arm A and Arm B.
"""

import os

from configs.base import CHECKPOINTS_PATH
from configs.frontier_cs.w_qwen3_6_27b_frontier_cs_eval import (  # noqa: F401
    _SlimeEval,
    modal,
)

_SOURCE_RUN_TAG = os.environ.get("RETRO_EVAL_RUN_TAG", "")
_RUN_TAG = f"qwen3.6-27b-frontier-cs-retro-eval-{_SOURCE_RUN_TAG or 'MISSING_SOURCE'}"

modal.image_env = {
    **modal.image_env,
    "RETRO_EVAL_RUN_TAG": _SOURCE_RUN_TAG,
}


class _SlimeRetroEval(_SlimeEval):
    load = f"{CHECKPOINTS_PATH}/swe_ckpts/{_SOURCE_RUN_TAG or 'MISSING_SOURCE_RUN_TAG'}"
    wandb_group = _RUN_TAG
    disable_wandb_random_suffix = True
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _SlimeRetroEval()
