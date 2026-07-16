"""Variant: the study champion (tp4: 2xTP4, dp-off) PLUS EAGLE off.

Combines the two batch-1/2 wins: fix concurrency/queueing via 2 engines (tp4
already got queue to 5.7s and uncached to 0.05), AND lift EAGLE's hidden
``max_running_requests=48`` cap (nospec showed that cap drives baseline's queue;
removing it collapsed queue 81->0.45s). Candidate global optimum.

``make_slime`` nulls the DP/EP companions (dp-off) and the EAGLE companions
(spec None). Trade-off to watch: nospec doubled ITL (graphs went 80% off at high
concurrency). Does tp4's lower per-engine concurrency keep decode batches inside
the captured cuda-graph sizes (off-frac stays ~0) so we get nospec's queue/TTFT
collapse WITHOUT the ITL penalty? Watch: itl_p50_ms, cuda-graph off-frac (modal
logs), queue_s, e2e/turn, agent_p50/step_wall (work-bound?).
"""

from configs.rollout_profile._base import make_slime, modal  # noqa: F401

slime = make_slime(
    "tp4_nospec",
    rollout_num_gpus_per_engine=4,
    sglang_enable_dp_attention=False,
    sglang_speculative_algorithm=None,
)
