"""COLOCATE variant: 3 nodes (24 GPU shared), 12xTP2 engines, dp-off, 64k ctx.

Scales the colocate `colo2n_tp2` to 3 nodes. Since colocate runs train on ALL
GPUs, more nodes speed up BOTH the (work-bound) rollout's concurrency headroom AND
the training step (Megatron over 24 GPUs). Watch the step_time decomposition: does
train_time drop with more GPUs while the work-bound rollout floor holds?
"""

from configs.rollout_profile._base import make_colocate, modal  # noqa: F401

# NB: 24-GPU colocate trains with data-parallel size 6, so global_batch_size must
# be divisible by 6 (256 is NOT -> Megatron init asserts). 240 = 30*8 is divisible
# by 4/6/8, so it is safe at any colocate node count. Minor (~6%) batch delta vs the
# 256-sample variants; immaterial for per-turn perf metrics.
slime = make_colocate(
    "colo3n_tp2",
    actor_num_nodes=3,
    rollout_num_gpus=24,
    rollout_num_gpus_per_engine=2,
    sglang_enable_dp_attention=False,
    sglang_mem_fraction_static=0.6,
    rollout_max_context_len=65536,
    rollout_batch_size=30,
    global_batch_size=240,
)
