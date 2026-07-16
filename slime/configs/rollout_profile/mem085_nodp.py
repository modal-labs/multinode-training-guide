"""Variant: mem_fraction 0.85 ON TOP of dp-attention OFF (1xTP8).

Batch-1 showed `mem085` (mem 0.85, dp_attn ON) did ~nothing: the KV pool is
partitioned across 8 DP ranks, so +0.15 mem_fraction barely moved uncached
(0.65->0.61). `nodp` (mem 0.7, dp OFF) unified the pool and dropped uncached to
0.38 on its own. This variant asks the clean H2 question: once the pool is
UNIFIED (dp off), does the static-fraction bump finally help?

``make_slime`` nulls the DP/EP companions (dp off). Watch: uncached_frac and
ttft_s vs `nodp` (same topology, mem 0.7) -> isolates the mem_fraction effect in
the unfragmented regime. max_total_num_tokens should rise above nodp's 4.74M.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime(
    "mem085_nodp",
    sglang_mem_fraction_static=0.85,
    sglang_enable_dp_attention=False,
)
