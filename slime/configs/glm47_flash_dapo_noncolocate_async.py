"""GLM-4.7-Flash (30B-A3B MoE) GRPO on DAPO-Math-17k — non-colocated 2-node (2×8 H100 actor + 1×8 rollout).

TP=2, PP=2, CP=2, EP=8 → TP×EP=16, requires 3 nodes total.
Checkpoint: convert with nproc=4 (TP=2, PP=2, decoder_last=23) → GLM-4.7-Flash_torch_dist_tp2pp2
"""
from configs.glm47_flash_dapo_noncolocate import _Slime as _Glm47Slime, modal


class _Slime(_Glm47Slime):
    # ── Async ──────────────────────────────────────────────────────────────────
    async_mode = True

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_group = "glm4.7-flash-dapo-math-noncolocate-async"


slime = _Slime()
