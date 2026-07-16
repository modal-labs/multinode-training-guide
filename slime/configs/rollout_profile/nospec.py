"""Variant: speculative decoding OFF (no EAGLE).

``make_slime`` auto-nulls the EAGLE companion flags (num_steps, eagle_topk,
num_draft_tokens) when the algorithm is None.

Hypothesis: EAGLE is a decode-latency win when accept-rate is high, but the MTP
draft head can go stale across weight updates and adds capture/overhead. At
step 0 (no weight update yet) this isolates EAGLE's raw decode contribution.
Watch: itl_p50_ms, gen tok/s, e2e/turn vs baseline.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime("nospec", sglang_speculative_algorithm=None)
