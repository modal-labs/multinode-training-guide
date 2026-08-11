"""Phase-2 Arm B: best-submission trajectory reward plus 25% retro groups.

This is matched to ``w_qwen3_6_27b_frontier_cs_retro_a_final`` in every
rollout/training/system setting.  The sole intended experimental difference is
``ASYNC_RL_OUTCOME_REWARD=best`` instead of ``final``.
"""

import os

from configs.base import CHECKPOINTS_PATH
from configs.frontier_cs.w_qwen3_6_27b_frontier_cs_retro_a_final import (  # noqa: F401
    _LAUNCH_STAMP,
    _SlimeRetroAFinal,
    modal,
)

_RUN_TAG = f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-frontier-cs-retro-b-best'}-{_LAUNCH_STAMP}"
_RESUME = os.environ.get("RESUME")
_STATE_TAG = _RESUME or _RUN_TAG
_MANIFEST_PATH = f"{CHECKPOINTS_PATH}/frontier_retro/{_STATE_TAG}/manifests.jsonl"


class _SlimeRetroBBest(_SlimeRetroAFinal):
    environment = {
        **_SlimeRetroAFinal.environment,
        "ASYNC_RL_OUTCOME_REWARD": "best",
        "ASYNC_RL_SOLVED_BONUS": "0",
        "ASYNC_RL_RETRO_RUN_TAG": _STATE_TAG,
        "ASYNC_RL_RETRO_MANIFEST_PATH": _MANIFEST_PATH,
    }

    wandb_group = _RUN_TAG
    save = f"{CHECKPOINTS_PATH}/swe_ckpts/{_STATE_TAG}"
    load = save
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_STATE_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _SlimeRetroBBest()
