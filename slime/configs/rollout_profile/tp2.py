"""Variant: TP2 -> four engines over the 8-GPU rollout node (sgl-router LB).

``rollout_num_gpus`` stays 8; ``rollout_num_gpus_per_engine`` = 2 means
#engines = 8 // 2 = 4, each TP2. dp-attention off (factory nulls DP/EP
companions) -> clean pure-TP.

Follows up the study champion ``tp4`` (2xTP4, dp-off, e2e 11.2s, work-bound).
Question: does engine count keep scaling past 2 engines, or does TP2's smaller
per-request compute (only 2 GPUs of tensor parallelism per request) become the
floor? Watch: e2e/turn, itl_p50_ms (TP2 may raise per-token compute),
running_reqs, max_total_num_tokens (4 independent KV pools), step_wall / agent_p50
(is it still work-bound like tp4?).
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime(
    "tp2",
    rollout_num_gpus_per_engine=2,
    sglang_enable_dp_attention=False,
)
