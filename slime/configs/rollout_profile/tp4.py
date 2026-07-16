"""Variant: TP4 -> two engines over the 8-GPU rollout node (sgl-router LB).

``rollout_num_gpus`` stays 8 (the whole rollout node); ``rollout_num_gpus_per_engine``
= 4 means #engines = 8 // 4 = 2, each TP4. dp-attention is turned off so the
factory nulls the DP/EP companions and this is clean pure-TP.

This is the template for a TP sweep -- copy to tp2.py (per_engine=2 -> 4 engines)
etc. Trade-off to watch: smaller TP -> more engines -> more independent KV pools
& more concurrency headroom, but each engine has less aggregate compute/mem per
request. Watch: e2e/turn, itl_p50_ms, running_reqs, max_total_num_tokens,
step wall.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime(
    "tp4",
    rollout_num_gpus_per_engine=4,
    sglang_enable_dp_attention=False,
)
