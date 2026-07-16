"""O2 — Kevin solved bonus: reward = final + 0.3*1(solved). Frontier-CS 27B async, 100 rollouts.

Arm O2 of the outcome-reward ablation (see rewards.py's episode-level outcome
interface), based on ``w_qwen3_6_27b_frontier_cs_noncolocate_5n_async`` — so it
inherits the fully-async rollout (staleness cap 4, fault tolerance, eval off)
and the DAPO nonzero-std group filter. The one substantive change is the
training scalar: ``ASYNC_RL_SOLVED_BONUS=0.3`` on top of the unchanged final
grade (``ASYNC_RL_OUTCOME_REWARD=final``), HARD-CODED into the ``environment``
dict so the arm is self-contained — relaunching this config always runs O2.

This is Kevin's (arXiv:2507.11948) correctness term S = 0.3*1{correct} + score,
ported to frontier-cs with "correct" = full marks from the verifier. Kevin's
ablations found weight 1.0 over-optimizes correctness and 0.0 plateaus; 0.3
balances. Isolates the bonus effect from the "best" effect (arm O1): expected to
be subtle — only ~4% of baseline episodes solve (9/207 at rollout 19), so the
bonus widens the solved-vs-0.9x gap inside few groups. Replayed mean reward
0.175 -> 0.188. Note the interaction with the DAPO filter: a group where all 8
samples solve is zero-std WITH or WITHOUT the bonus, so drop rates should track
the baseline closely — divergence there is itself signal.

Watch: ``agentic/outcome/bonus_frac`` (fraction of episodes getting the bonus;
tracks solve rate) and ``agentic/solved_frac`` vs the baseline — the arm bets on
pushing near-solves (score ~0.9+) over the full-marks line.

Launch (one 6-node run at a time; O1 is ``w_qwen3_6_27b_frontier_cs_o1_best``):

    EXPERIMENT_CONFIG=w_qwen3_6_27b_frontier_cs_o2_bonus MODAL_ENVIRONMENT=junlin-dev \
    WANDB_PROJECT=Modal uv run --no-dev modal run -d slime/modal_train.py::train

Resume an expired run: RESUME=qwen3.6-27b-frontier-cs-o2-bonus-<stamp> + same command.
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
_RUN_TAG = f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-frontier-cs-o2-bonus'}-{_LAUNCH_STAMP}"


class _SlimeO2(_Slime):
    # The one substantive change: additive 0.3 bonus on solved episodes, base
    # scalar unchanged ("final"). Hard-coded (not os.environ.get) so the
    # in-container config re-import can't silently fall back to the baseline.
    environment = {
        **_Slime.environment,
        "ASYNC_RL_OUTCOME_REWARD": "final",
        "ASYNC_RL_SOLVED_BONUS": "0.3",
    }

    # Per-arm W&B group + checkpoint/dump dirs (the base computed these off its
    # own run tag at class-body time, so they must be re-derived here).
    wandb_group = _RUN_TAG
    save = f"{CHECKPOINTS_PATH}/swe_ckpts/{_RESUME or _RUN_TAG}"
    load = save
    save_debug_rollout_data = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"


slime = _SlimeO2()
