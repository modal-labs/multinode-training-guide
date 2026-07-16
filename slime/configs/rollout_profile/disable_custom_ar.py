"""Variant: disable SGLang's custom all-reduce (use NCCL all-reduce instead).

Baseline keeps custom all-reduce ON (sglang_disable_custom_all_reduce=False).
The custom kernel is usually faster intra-node, but can be flaky / slower on
some topologies. This flips it off to measure the all-reduce contribution to
per-token latency. Watch: itl_p50_ms, e2e/turn, and engine startup stability.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime("disable_custom_ar", sglang_disable_custom_all_reduce=True)
