"""Phase-1 branchability survey from manifests captured by the sibling config.

The survey restores each selected snapshot into eight independent sandboxes,
generates current-policy suffixes, saves the ordinary rollout dump, and exits
without training.  It requires the exact capture run tag:

    RETRO_SOURCE_RUN_TAG=qwen3.6-27b-frontier-cs-retro-capture-<stamp> \
    RETRO_SURVEY_GROUPS=1 \
    EXPERIMENT_CONFIG=frontier_cs.w_qwen3_6_27b_frontier_cs_retro_survey \
    MODAL_ENVIRONMENT=junlin-dev WANDB_PROJECT=Modal \
      uv run --no-dev modal run -d slime/modal_train.py::train

Analyze the resulting dump with:

    python -m agentic_rl.retro.phase1 /checkpoints/.../rollout_0.pt
"""

import os

from configs.base import CHECKPOINTS_PATH
from configs.frontier_cs.w_qwen3_6_27b_frontier_cs_retro_capture import (  # noqa: F401
    _LAUNCH_STAMP,
    _SlimeRetroCapture,
    modal,
)

_RUN_TAG = f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-frontier-cs-retro-survey'}-{_LAUNCH_STAMP}"
_SOURCE_RUN_TAG = os.environ.get("RETRO_SOURCE_RUN_TAG", "")
_GROUPS = max(1, int(os.environ.get("RETRO_SURVEY_GROUPS", "1")))
_MANIFEST_PATH = (
    os.environ.get("RETRO_MANIFEST_PATH")
    or f"{CHECKPOINTS_PATH}/frontier_retro/{_SOURCE_RUN_TAG or 'MISSING_SOURCE_RUN_TAG'}/manifests.jsonl"
)

# Preserve local launch choices across the config's in-container re-import.
modal.image_env = {
    **modal.image_env,
    "RETRO_SOURCE_RUN_TAG": _SOURCE_RUN_TAG,
    "RETRO_SURVEY_GROUPS": str(_GROUPS),
    "RETRO_MANIFEST_PATH": str(_MANIFEST_PATH),
}


class _SlimeRetroSurvey(_SlimeRetroCapture):
    rollout_function_path = "agentic_rl.retro.rollout.generate_retro_survey"
    debug_rollout_only = True
    num_rollout = 1
    rollout_batch_size = _GROUPS
    n_samples_per_prompt = 8
    global_batch_size = _GROUPS * 8

    environment = {
        **_SlimeRetroCapture.environment,
        "ASYNC_RL_RETRO_RUN_TAG": _RUN_TAG,
        "ASYNC_RL_RETRO_MANIFEST_PATH": _MANIFEST_PATH,
    }

    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _SlimeRetroSurvey()
