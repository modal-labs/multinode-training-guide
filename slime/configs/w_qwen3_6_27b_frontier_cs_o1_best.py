"""O1 — outcome reward = max(final, best submitted). Frontier-CS 27B async, 100 rollouts.

Arm O1 of the outcome-reward ablation (see rewards.py's episode-level outcome
interface), based on ``w_qwen3_6_27b_frontier_cs_noncolocate_5n_async`` — so it
inherits the fully-async rollout (staleness cap 4, fault tolerance, eval off)
and the DAPO nonzero-std group filter. The one substantive change is the
training scalar: ``ASYNC_RL_OUTCOME_REWARD=best`` is HARD-CODED into the
config's ``environment`` dict (not read from the launch shell), so the arm is
self-contained and reproducible — relaunching this config always runs O1.

Hypothesis (rollout_19 replay of the 20260708-023750 baseline): 37.7% of episodes
submit a better solution mid-episode than their final solution.cpp (mean uplift
0.32). Training on max(final, best submitted) stops punishing final-solution
regressions: replayed mean reward 0.175 -> 0.294, episodes with nonzero reward
76 -> 125 of 207, zero-std GRPO groups 7 -> 6 of 32 (which also lowers the DAPO
filter's drop/refill rate).

Watch: ``agentic/outcome/reward_trained`` vs ``reward_final`` (the gap is what
this arm trains on), and the submit-spam watchdog ``agentic/submissions/mean``
(sync baseline ~5.2, p90 12) — "best" mildly rewards submitting often, and the
async pool runs up to 1024 concurrent episodes against the per-worker judge
(which has crashed under load before), so flag if the mean crosses ~10 or
submit latency climbs.

Launch (one 6-node run at a time; O2 is ``w_qwen3_6_27b_frontier_cs_o2_bonus``):

    EXPERIMENT_CONFIG=w_qwen3_6_27b_frontier_cs_o1_best MODAL_ENVIRONMENT=junlin-dev \
    WANDB_PROJECT=Modal uv run --no-dev modal run -d slime/modal_train.py::train

Resume an expired run: RESUME=qwen3.6-27b-frontier-cs-o1-best-<stamp> + same command.
"""

import os

from configs.base import CHECKPOINTS_PATH
from configs.w_qwen3_6_27b_frontier_cs_noncolocate_5n_async import (  # noqa: F401
    _LAUNCH_STAMP,
    _RESUME,
    _Slime,
    modal,
)

# Same stamp mechanics as the async base (baked into the image env there, so
# Modal auto-retries resume); only the default group name differs per arm.
_RUN_TAG = f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-frontier-cs-o1-best'}-{_LAUNCH_STAMP}"


class _SlimeO1(_Slime):
    # The one substantive change: train on max(final, best mid-episode submission).
    # Hard-coded (not os.environ.get) so the in-container config re-import can't
    # silently fall back to the baseline if the launch shell forgets the var.
    environment = {
        **_Slime.environment,
        "ASYNC_RL_OUTCOME_REWARD": "best",
        "ASYNC_RL_SOLVED_BONUS": "0",
    }

    # Per-arm W&B group + checkpoint/dump dirs (the base computed these off its
    # own run tag at class-body time, so they must be re-derived here).
    wandb_group = _RUN_TAG
    save = f"{CHECKPOINTS_PATH}/swe_ckpts/{_RESUME or _RUN_TAG}"
    load = save
    save_debug_rollout_data = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"


slime = _SlimeO1()
