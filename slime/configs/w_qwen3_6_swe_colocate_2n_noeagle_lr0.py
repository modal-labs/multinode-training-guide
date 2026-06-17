"""Two-rollout diagnostic: no EAGLE, zero learning rate.

Purpose: distinguish a repeated SGLang weight-update/export bug from a
post-optimizer/backup bug. If rollout 1 collapses even with lr=0 and
weight_decay=0, the second update path is corrupting serving weights despite no
intended actor change. If rollout 1 stays healthy, the corruption is introduced
by the optimizer/offload/backup state after a real step.
"""

from configs import w_qwen3_6_swe_colocate_1n as _base
from configs.base import CHECKPOINTS_PATH, run_tag

modal = _base.modal

_RUN_TAG = run_tag("qwen3.6-35b-a3b-swe-gym-lite-colocate-2n-noeagle-lr0")


class _Slime(_base._Slime):
    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    rollout_num_gpus_per_engine = 8

    num_rollout = 2
    eval_interval = None

    # No-EAGLE diagnostic. Null inherited sub-flags as well so SGLang sees a
    # clean non-speculative config.
    sglang_speculative_algorithm = None
    sglang_speculative_num_steps = None
    sglang_speculative_eagle_topk = None
    sglang_speculative_num_draft_tokens = None
    sglang_mem_fraction_static = 0.65

    # Make the train step a no-op at the parameter level.
    lr = 0.0
    weight_decay = 0.0

    wandb_group = _RUN_TAG
    save_debug_rollout_data = (
        f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    )


slime = _Slime()
