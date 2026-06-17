"""DEBUG: does the step-0 optimizer/backward step corrupt the Megatron model?

Inherits the 1n training config, but feeds the COLLAPSING run's coherent step-0
rollout dump (load_debug_rollout_data → auto debug_train_only, no sglang), so the
optimizer takes the SAME first step that preceded the collapse (grad_norm≈0.08).
train.py SLIME_SAVE_HF_AFTER_TRAIN then trains exactly one step and dumps HF.

Diff that post-train HF vs origin (modal_train.py::diff_two_hf after pointing it at
iter_0 here): huge diff => the optimizer/backward corrupts the Megatron weights
(then bisect precision-aware-optimizer / cpu-offload); ~clean diff => the model is
fine and the collapse is sglang-side / EAGLE-MTP on the post-train resync.
"""

from configs import w_qwen3_6_swe_colocate_1n as _base
from configs.base import CHECKPOINTS_PATH, run_tag

modal = _base.modal

_RUN_TAG = run_tag("qwen3.6-resync-probe-posttrain")

# Coherent step-0 rollout from the collapsing 2n run (reward 0.42, real variance).
_DUMP = (
    f"{CHECKPOINTS_PATH}/swe_rollout_dumps/"
    f"qwen3.6-35b-a3b-swe-gym-lite-colocate-2n-20260616-081603/rollout_{{rollout_id}}.pt"
)


class _Slime(_base._Slime):
    load_debug_rollout_data = _DUMP
    num_rollout = 1
    save_hf = f"{CHECKPOINTS_PATH}/_faithful_resync_hf_posttrain/iter_{{rollout_id}}"
    use_wandb = False
    environment = {**_base._Slime.environment, "SLIME_SAVE_HF_AFTER_TRAIN": "1"}


slime = _Slime()
