"""GLM-4.7-355B-A32B GSPO on DAPO-Math-17k — 4-node actor + 4-node rollout.

Scaled-down non-colocated variant: TP=8, PP=4 on 4 actor nodes (32 GPUs),
no context parallelism needed. EP reduced from 16→8 to fit 8 GPUs per PP stage.
Uses the same TP=8,PP=4 checkpoint as the 8-node colocated config.
"""

from configs import glm47_355b_a32b as _base
from configs.base import ModalConfig

modal = ModalConfig(gpu="H200")


class _Slime(_base._Slime):
    # ── Infrastructure ────────────────────────────────────────────────────────
    colocate = False
    rollout_num_gpus = 64

    rollout_batch_size = 32
    n_samples_per_prompt = 4

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_group = "glm4.7-355b-a32b-dapo-math-noncolocate"


slime = _Slime()
