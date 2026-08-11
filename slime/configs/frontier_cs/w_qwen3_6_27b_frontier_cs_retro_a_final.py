"""Phase-2 Arm A: final/last-solution reward plus 25% retro groups.

Safe default is a one-step paired smoke on two nodes (16 H200 total):

    EXPERIMENT_CONFIG=frontier_cs.w_qwen3_6_27b_frontier_cs_retro_a_final \
    MODAL_ENVIRONMENT=junlin-dev WANDB_PROJECT=Modal \
      uv run --no-dev modal run slime/modal_train.py::train

The rollout remains compute-matched: four groups total = three fresh groups and
one eight-sibling retro group.  For the later full pilot, set
RETRO_PHASE2_GROUPS=32 and RETRO_PHASE2_ROLLOUTS=20.

Continue a completed 20-step run to 85 with its original run tag:

    RESUME=qwen3.6-27b-frontier-cs-retro-a-final-<stamp> \
    RETRO_PHASE2_GROUPS=32 RETRO_PHASE2_ROLLOUTS=85 ...

Unified retro ablation surface (same flags apply to Arm B):

    # Where within each realized source trajectory to branch.
    RETRO_TARGET_TRAJECTORY_FRACTION=0.50  # try 0.50 or 0.75
    RETRO_MAX_FRACTION_ERROR=0.40          # nearest matching submission may be broad

    # How each fresh rollout is assigned a capture event class.
    RETRO_SELECTOR_ASSIGNMENT=hashed       # hashed|alternating|promising|recovery|any
    RETRO_CAPTURE_PROMISING_RATIO=0.50     # used only by hashed assignment
    RETRO_SELECTOR_SEED=20260802
    RETRO_SELECTOR_FALLBACK=0              # strict class matching by default

    # How eight replay groups are drawn from the eligible past-state pool.
    RETRO_POOL_PROMISING_RATIO=0.50        # 0.75 => six promising, two recovery
    RETRO_POOL_ORDER=newest                # newest|fifo
    RETRO_MIN_POLICY_AGE=0
    RETRO_MAX_POLICY_AGE=4
"""

import os

from configs.base import CHECKPOINTS_PATH
from configs.frontier_cs.w_qwen3_6_27b_frontier_cs_retro_capture import (  # noqa: F401
    _LAUNCH_STAMP,
    _SlimeRetroCapture,
    modal,
)

_RUN_TAG = f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-frontier-cs-retro-a-final'}-{_LAUNCH_STAMP}"
_GROUPS = max(2, int(os.environ.get("RETRO_PHASE2_GROUPS", "4")))
_ROLLOUTS = max(1, int(os.environ.get("RETRO_PHASE2_ROLLOUTS", "1")))
_RATIO = float(os.environ.get("RETRO_GROUP_RATIO", "0.25"))
_FULL_TOPOLOGY = _GROUPS >= 32
_RESUME = os.environ.get("RESUME")
_RESUME_CKPT_STEP = int(os.environ["RESUME_CKPT_STEP"]) if os.environ.get("RESUME_CKPT_STEP") else None
_STATE_TAG = _RESUME or _RUN_TAG
_MANIFEST_PATH = f"{CHECKPOINTS_PATH}/frontier_retro/{_STATE_TAG}/manifests.jsonl"

modal.image_env = {
    **modal.image_env,
    "RETRO_PHASE2_GROUPS": str(_GROUPS),
    "RETRO_PHASE2_ROLLOUTS": str(_ROLLOUTS),
    "RETRO_GROUP_RATIO": str(_RATIO),
    "RETRO_POOL_PROMISING_RATIO": os.environ.get("RETRO_POOL_PROMISING_RATIO", "0.5"),
    "RETRO_POOL_ORDER": os.environ.get("RETRO_POOL_ORDER", "newest"),
    "RETRO_MIN_POLICY_AGE": os.environ.get("RETRO_MIN_POLICY_AGE", "0"),
    "RETRO_MAX_POLICY_AGE": os.environ.get("RETRO_MAX_POLICY_AGE", "4"),
    **({"RESUME_CKPT_STEP": str(_RESUME_CKPT_STEP)} if _RESUME_CKPT_STEP is not None else {}),
}


class _SlimeRetroAFinal(_SlimeRetroCapture):
    debug_rollout_only = False
    rollout_function_path = "agentic_rl.retro.rollout.generate_retro_mixed"
    # The one-step pair is an engineering smoke: do not over-generate expensive
    # groups merely to satisfy DAPO. Full 32-group runs restore the production
    # nonzero-std filter.
    dynamic_sampling_filter_path = (
        "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
        if _FULL_TOPOLOGY
        else None
    )

    actor_num_nodes = 2 if _FULL_TOPOLOGY else 1
    rollout_num_gpus = 32 if _FULL_TOPOLOGY else 8
    sglang_server_concurrency = 64 if _FULL_TOPOLOGY else 16
    rollout_max_staleness = 4 if _FULL_TOPOLOGY else 1

    num_rollout = _ROLLOUTS
    # Extending a completed pilot changes Megatron's derived scheduler horizon
    # (20 -> 85 iterations). Keep the configured constant-LR schedule instead of
    # rejecting the checkpoint because its original total-step count differs.
    override_opt_param_scheduler = bool(_RESUME)
    ckpt_step = _RESUME_CKPT_STEP
    rollout_batch_size = _GROUPS
    n_samples_per_prompt = 8
    global_batch_size = _GROUPS * 8
    # Pair the two reward arms on the same prompt order and per-sibling sampling
    # seeds. They diverge only after their reward-dependent updates diverge.
    rollout_shuffle = False
    rollout_seed = 20260802
    sglang_enable_deterministic_inference = True

    environment = {
        **_SlimeRetroCapture.environment,
        "ASYNC_RL_OUTCOME_REWARD": "final",
        "ASYNC_RL_SOLVED_BONUS": "0",
        "ASYNC_RL_RETRO_RUN_TAG": _STATE_TAG,
        "ASYNC_RL_RETRO_MANIFEST_PATH": _MANIFEST_PATH,
        "ASYNC_RL_RETRO_CAPTURE_STATUS": "tentative",
        "ASYNC_RL_RETRO_GROUP_RATIO": str(_RATIO),
        "ASYNC_RL_RETRO_MAX_POLICY_AGE": os.environ.get("RETRO_MAX_POLICY_AGE", "4"),
        "ASYNC_RL_RETRO_MIN_POLICY_AGE": os.environ.get("RETRO_MIN_POLICY_AGE", "0"),
        "ASYNC_RL_RETRO_POOL_PROMISING_RATIO": os.environ.get(
            "RETRO_POOL_PROMISING_RATIO", "0.5"
        ),
        "ASYNC_RL_RETRO_POOL_ORDER": os.environ.get("RETRO_POOL_ORDER", "newest"),
        "ASYNC_RL_RETRO_MAX_ATTEMPTS": os.environ.get("RETRO_MAX_ATTEMPTS", "3"),
    }

    wandb_group = _RUN_TAG
    save = f"{CHECKPOINTS_PATH}/swe_ckpts/{_STATE_TAG}"
    load = save
    save_interval = 5
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_STATE_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _SlimeRetroAFinal()
