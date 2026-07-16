"""Rollout-scaling: 4 nodes = 1 train + 3 rollout (rollout_num_gpus=24), tp2.

Noncolocate, batch held at 256. Latency-scaling endpoint: like `tp2_3n` but 12xTP2
engines. Expect the step wall flat (work-bound). The point of the extra rollout
GPUs is throughput (see `tp2_4n_b768`), not single-step latency.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime(
    "tp2_4n",
    rollout_num_gpus=24,
    rollout_num_gpus_per_engine=2,
    sglang_enable_dp_attention=False,
)
