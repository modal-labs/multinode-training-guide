"""Phase 0/1 capture run for Frontier-CS retro replay.

This is an isolated debug-rollout config: it inherits the stable async 27B
recipe, swaps only the custom generate hook, captures at most one eligible
directory snapshot per episode, writes manifests to the checkpoint volume, and
never trains.

Start with one prompt group (8 episodes):

    EXPERIMENT_CONFIG=frontier_cs.w_qwen3_6_27b_frontier_cs_retro_capture \
    MODAL_ENVIRONMENT=junlin-dev WANDB_PROJECT=Modal \
      uv run --no-dev modal run -d slime/modal_train.py::train

After the one-group smoke succeeds, set RETRO_CAPTURE_GROUPS=32 to collect a
larger candidate pool.  The printed/W&B run tag is required by the survey config.
"""

import os

from configs.base import CHECKPOINTS_PATH
from configs.frontier_cs.w_qwen3_6_27b_frontier_cs_noncolocate_5n_async import (  # noqa: F401
    _LAUNCH_STAMP,
    _Slime,
    modal,
)

_RUN_TAG = f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-frontier-cs-retro-capture'}-{_LAUNCH_STAMP}"
_GROUPS = max(1, int(os.environ.get("RETRO_CAPTURE_GROUPS", "1")))
_MANIFEST_PATH = f"{CHECKPOINTS_PATH}/frontier_retro/{_RUN_TAG}/manifests.jsonl"

# These values are chosen by the local launch shell but the config is imported
# again inside Modal. Bake them into the image so both imports resolve the same
# capture recipe.
modal.image_env = {
    **modal.image_env,
    "RETRO_CAPTURE_GROUPS": str(_GROUPS),
    "RETRO_SNAPSHOT_KIND": os.environ.get("RETRO_SNAPSHOT_KIND", "directory"),
    "RETRO_SNAPSHOT_TTL": os.environ.get("RETRO_SNAPSHOT_TTL", str(48 * 60 * 60)),
    "RETRO_MIN_SCORE": os.environ.get("RETRO_MIN_SCORE", "0.1"),
    "RETRO_MAX_SCORE": os.environ.get("RETRO_MAX_SCORE", "0.95"),
    "RETRO_MIN_REMAINING": os.environ.get("RETRO_MIN_REMAINING", "0.0"),
    "RETRO_MIN_TURN": os.environ.get("RETRO_MIN_TURN", "2"),
    "RETRO_REGRESSION_DELTA": os.environ.get("RETRO_REGRESSION_DELTA", "0.1"),
    "RETRO_STAGNANT_SUBMISSIONS": os.environ.get("RETRO_STAGNANT_SUBMISSIONS", "2"),
    "RETRO_TARGET_TRAJECTORY_FRACTION": os.environ.get("RETRO_TARGET_TRAJECTORY_FRACTION", "0.5"),
    "RETRO_MAX_FRACTION_ERROR": os.environ.get("RETRO_MAX_FRACTION_ERROR", "0.4"),
    "RETRO_SELECTOR_ASSIGNMENT": os.environ.get("RETRO_SELECTOR_ASSIGNMENT", "hashed"),
    "RETRO_CAPTURE_PROMISING_RATIO": os.environ.get("RETRO_CAPTURE_PROMISING_RATIO", "0.5"),
    "RETRO_SELECTOR_SEED": os.environ.get("RETRO_SELECTOR_SEED", "20260802"),
    "RETRO_SELECTOR_FALLBACK": os.environ.get("RETRO_SELECTOR_FALLBACK", "0"),
}


class _SlimeRetroCapture(_Slime):
    custom_generate_function_path = "agentic_rl.retro.generate.generate"
    debug_rollout_only = True
    # Debug-only capture/survey does not need the production 2-train + 4-rollout
    # footprint. TP4×CP2 fits one 8-GPU actor node; four TP2 rollout engines fit
    # one 8-GPU rollout node. Modal still provisions whole 8-GPU nodes, so this
    # cuts the smoke from 48 to 16 H200s without changing model or tokenization.
    actor_num_nodes = 1
    rollout_num_gpus = 8
    sglang_server_concurrency = 16
    rollout_max_staleness = 1
    num_rollout = 1
    rollout_batch_size = _GROUPS
    n_samples_per_prompt = 8
    global_batch_size = _GROUPS * 8
    dynamic_sampling_filter_path = None

    environment = {
        **_Slime.environment,
        # Align the eventual matched Arm A/Arm B design with O1.  Phase-1
        # analysis still reports final and branch-local best separately.
        "ASYNC_RL_OUTCOME_REWARD": "best",
        "ASYNC_RL_SOLVED_BONUS": "0",
        "ASYNC_RL_RETRO_RUN_TAG": _RUN_TAG,
        "ASYNC_RL_RETRO_MANIFEST_PATH": _MANIFEST_PATH,
        "ASYNC_RL_RETRO_SNAPSHOT_KIND": os.environ.get("RETRO_SNAPSHOT_KIND", "directory"),
        "ASYNC_RL_RETRO_SNAPSHOT_PATH": "/app",
        "ASYNC_RL_RETRO_SNAPSHOT_TTL": os.environ.get("RETRO_SNAPSHOT_TTL", str(48 * 60 * 60)),
        # Standalone capture runs publish immediately. Mixed training overrides
        # this to tentative and activates only its accepted primary-fresh groups.
        "ASYNC_RL_RETRO_CAPTURE_STATUS": "available",
        "ASYNC_RL_RETRO_MIN_SCORE": os.environ.get("RETRO_MIN_SCORE", "0.1"),
        "ASYNC_RL_RETRO_MAX_SCORE": os.environ.get("RETRO_MAX_SCORE", "0.95"),
        "ASYNC_RL_RETRO_MIN_REMAINING": os.environ.get("RETRO_MIN_REMAINING", "0.0"),
        "ASYNC_RL_RETRO_MIN_TURN": os.environ.get("RETRO_MIN_TURN", "2"),
        "ASYNC_RL_RETRO_REGRESSION_DELTA": os.environ.get("RETRO_REGRESSION_DELTA", "0.1"),
        "ASYNC_RL_RETRO_STAGNANT_SUBMISSIONS": os.environ.get("RETRO_STAGNANT_SUBMISSIONS", "2"),
        "ASYNC_RL_RETRO_TARGET_TRAJECTORY_FRACTION": os.environ.get(
            "RETRO_TARGET_TRAJECTORY_FRACTION", "0.5"
        ),
        "ASYNC_RL_RETRO_MAX_FRACTION_ERROR": os.environ.get("RETRO_MAX_FRACTION_ERROR", "0.4"),
        "ASYNC_RL_RETRO_SELECTOR_ASSIGNMENT": os.environ.get("RETRO_SELECTOR_ASSIGNMENT", "hashed"),
        "ASYNC_RL_RETRO_CAPTURE_PROMISING_RATIO": os.environ.get(
            "RETRO_CAPTURE_PROMISING_RATIO", "0.5"
        ),
        "ASYNC_RL_RETRO_SELECTOR_SEED": os.environ.get("RETRO_SELECTOR_SEED", "20260802"),
        "ASYNC_RL_RETRO_SELECTOR_FALLBACK": os.environ.get("RETRO_SELECTOR_FALLBACK", "0"),
    }

    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _SlimeRetroCapture()
