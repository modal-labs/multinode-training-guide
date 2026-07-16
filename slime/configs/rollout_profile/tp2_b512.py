"""Rollout SATURATION probe: 2 nodes (rollout_num_gpus=8), batch 2x -> 512, tp2.

Noncolocate. Same 8-GPU rollout node as `tp2`, but 2x the batch (512 samples) on
the SAME engines (4xTP2, ~128 reqs/engine). The contrast to `tp2_3n_b512` (which
adds GPUs with the batch): here we OVERLOAD one node to see the saturation
signature — queue_s/e2e/uncached should climb as the engines go from work-bound
toward engine-bound. Marks where a single 8-GPU node tops out.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime(
    "tp2_b512",
    rollout_num_gpus=8,
    rollout_num_gpus_per_engine=2,
    sglang_enable_dp_attention=False,
    rollout_batch_size=64,
    global_batch_size=512,
)
