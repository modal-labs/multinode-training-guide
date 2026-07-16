"""Rollout-scaling THROUGHPUT: 4 nodes, rollout_num_gpus=24, batch 3x -> 768.

Noncolocate. 3x rollout GPU AND 3x batch (rollout_batch_size 32->96 -> 768 samples,
global_batch_size 768) vs `tp2`. Throughput-scaling endpoint: does it stay flat
(linear scaling) at 4 nodes, or has per-engine load / sgl-router / cross-node
networking started to bite? Same training-dynamics caveat as `tp2_3n_b512`.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime(
    "tp2_4n_b768",
    rollout_num_gpus=24,
    rollout_num_gpus_per_engine=2,
    sglang_enable_dp_attention=False,
    rollout_batch_size=96,
    global_batch_size=768,
)
