"""DEBUG fix-test: Qwen3.6 SWE 1n with EAGLE speculative decoding DISABLED.

Everything in the chain (converter, TP/EP gather, optimizer step) is proven
bit-correct, and the Megatron model is pristine after step 0 — yet step-1 rollout
collapses to token-salad after the post-train resync. The remaining suspect is the
sglang-side weight application, with EAGLE (MTP draft head across weight updates)
flagged by the base config docstring as the thing to disable first.

This run inherits the real 1n config but turns EAGLE off and runs just 2 steps:
step 0 = coherent baseline, step 1 = the test. If raw_reward stays up at step 1
(no entropy explosion), EAGLE is confirmed as the cause.
"""

from configs import w_qwen3_6_swe_colocate_1n as _base
from configs.base import CHECKPOINTS_PATH, run_tag

modal = _base.modal

_RUN_TAG = run_tag("qwen3.6-swe-noeagle-1n")


class _Slime(_base._Slime):
    # ── EAGLE OFF (the one delta vs the collapsing run) ───────────────────────
    sglang_speculative_algorithm = None
    sglang_speculative_num_steps = None
    sglang_speculative_eagle_topk = None
    sglang_speculative_num_draft_tokens = None

    # ── Just enough to see step 0 (coherent) -> step 1 (the test) ─────────────
    num_rollout = 2
    eval_interval = None

    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _Slime()
