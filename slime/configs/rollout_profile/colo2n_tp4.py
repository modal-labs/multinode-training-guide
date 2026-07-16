"""COLOCATE variant: 2 nodes (16 GPU shared), 4xTP4 engines, dp-off, 64k ctx.

Colocate likely prefers a larger TP than the noncolocate champion: each GPU holds
both the SGLang KV pool AND (after offload) Megatron state, so the smaller per-GPU
model share of TP4 leaves more room for KV at 64k ctx than TP2. This is the
expected colocate sweet spot if `colo2n_tp2` OOMs. Compare to `colo2n_tp2` (engine
choice under colocate) and to the noncolocate `tp4`.
"""

from configs.rollout_profile._base import make_colocate, modal  # noqa: F401

slime = make_colocate(
    "colo2n_tp4",
    rollout_num_gpus=16,
    rollout_num_gpus_per_engine=4,
    sglang_enable_dp_attention=False,
    sglang_mem_fraction_static=0.6,
    rollout_max_context_len=65536,
)
