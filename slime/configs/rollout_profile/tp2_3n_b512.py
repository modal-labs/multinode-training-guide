"""Rollout-scaling THROUGHPUT: 3 nodes, rollout_num_gpus=16, batch 2x -> 512.

Noncolocate. 2x rollout GPU AND 2x batch (rollout_batch_size 32->64 -> 512 samples,
global_batch_size 512) vs `tp2`. Tests THROUGHPUT scaling: if the wall stays ~flat
while doing 2x the samples, rollout throughput scales linearly with rollout nodes
(the real value of adding them in async — keeps the trainer fed, kills the 77%
idle). Where queue_s/e2e start climbing = the per-GPU saturation point.

NOTE: bigger rollout_batch_size changes the data/advantage statistics per step —
a training-dynamics change, not just perf. Here it's only for throughput profiling.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime(
    "tp2_3n_b512",
    rollout_num_gpus=16,
    rollout_num_gpus_per_engine=2,
    sglang_enable_dp_attention=False,
    rollout_batch_size=64,
    global_batch_size=512,
)
