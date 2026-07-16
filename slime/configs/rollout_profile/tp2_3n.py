"""Rollout-scaling: 3 nodes = 1 train + 2 rollout (rollout_num_gpus=16), tp2.

Noncolocate. Same `tp2` engine recipe, but the rollout node count doubles
(8->16 rollout GPU = 8xTP2 engines behind sgl-router), batch held at 256. Tests
LATENCY scaling at fixed work: since `tp2` was already work-bound, expect the
step wall to stay ~FLAT (more inference GPUs can't speed an agent-bound rollout).
Confirms that extra rollout nodes don't shrink the single-step wall.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime(
    "tp2_3n",
    rollout_num_gpus=16,
    rollout_num_gpus_per_engine=2,
    sglang_enable_dp_attention=False,
)
