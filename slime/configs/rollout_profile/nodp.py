"""Variant: DP-attention OFF (pure TP8).

``make_slime`` auto-nulls the DP/EP companions (dp_size, enable_dp_lm_head,
ep_size) when dp-attention is off, so this is a clean single-engine TP8 config.

Hypothesis (inherited from PERF.md): DP-attention makes every DP rank run the
same forward, so any in-flight prefill drops the whole engine onto the
graph-less mixed path -- decode loses cuda graphs. Decode batches here are small
(3-11 reqs/rank), where DP-attention's large-batch throughput win does not
apply. Watch: itl_p50_ms (decode speed), cuda-graph off-fraction (modal logs),
e2e/turn.
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime("nodp", sglang_enable_dp_attention=False)
