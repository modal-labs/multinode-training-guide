"""Variant: grow the static KV pool, 0.7 -> 0.85 mem_fraction_static.

Hypothesis (inherited from PERF.md): more KV cache -> fewer prefix evictions
between an agent's turns -> lower uncached_frac (re-prefill) -> fewer mixed
prefill+decode forwards -> faster decode. Watch: uncached_frac, cache_hit_rate,
max_total_num_tokens, ttft, e2e/turn.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime("mem085", sglang_mem_fraction_static=0.85)
