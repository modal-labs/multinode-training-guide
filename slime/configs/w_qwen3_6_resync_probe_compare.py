"""DEBUG pinpoint: name the sglang tensor(s) corrupted by the colocate
offload/resume/update cycle for qwen3_next.

check_weight_update_equal snapshots the (correct) weights at init + zeros them;
each update_weights should restore them. train.py SLIME_COMPARE_AND_EXIT runs a
`compare` AFTER the in-loop resync at rollout 0 (the update that precedes the
step-1 collapse, i.e. after an offload→resume cycle) and prints which tensors
mismatch the snapshot — gross mismatch = the corrupted tensor(s) (vs the legit
~1e-6 training delta). Then exits (no step-1 rollout needed). EAGLE off (collapse
reproduces without it; keeps the weight set clean).
"""

from configs import w_qwen3_6_swe_colocate_1n as _base

modal = _base.modal


class _Slime(_base._Slime):
    sglang_speculative_algorithm = None
    sglang_speculative_num_steps = None
    sglang_speculative_eagle_topk = None
    sglang_speculative_num_draft_tokens = None

    check_weight_update_equal = True
    num_rollout = 1
    eval_interval = None
    use_wandb = False
    environment = {**_base._Slime.environment, "SLIME_COMPARE_AND_EXIT": "1"}


slime = _Slime()
