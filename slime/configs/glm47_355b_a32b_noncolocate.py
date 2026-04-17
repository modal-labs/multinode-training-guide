"""GLM-4.7-355B-A32B GSPO on DAPO-Math-17k — non-colocated: 4 actor + 4 rollout nodes.

Same model, TP=8, PP=4, and EP=16 as the base 8-node colocated config; this
variant splits actor and rollout across separate node pools (64 rollout GPUs)
instead of co-locating them on the same GPUs. Inherits the 2 TiB memory
reservation from the base config's ModalConfig.
"""

from configs import glm47_355b_a32b as _base

modal = _base.modal


class _Slime(_base._Slime):
    # ── Infrastructure ────────────────────────────────────────────────────────
    colocate = False
    rollout_num_gpus = 64

    rollout_batch_size = 32
    n_samples_per_prompt = 4

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_group = "glm4.7-355b-a32b-dapo-math-noncolocate"


slime = _Slime()
