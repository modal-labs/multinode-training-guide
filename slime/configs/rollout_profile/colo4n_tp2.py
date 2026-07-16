"""COLOCATE variant: 4 nodes (32 GPU shared), 16xTP2 engines, dp-off, 64k ctx.

Colocate scaling endpoint. With 32 GPUs all time-sharing, train is fastest and
rollout has the most concurrency headroom — but the rollout is work-bound, so the
step is gated by the agent floor + the (now-small) train_time. Tells us the
ceiling of the colocate path before per-node coordination cost dominates.
"""

from configs.rollout_profile._base import make_colocate, modal  # noqa: F401

slime = make_colocate(
    "colo4n_tp2",
    actor_num_nodes=4,
    rollout_num_gpus=32,
    rollout_num_gpus_per_engine=2,
    sglang_enable_dp_attention=False,
    sglang_mem_fraction_static=0.6,
    rollout_max_context_len=65536,
)
