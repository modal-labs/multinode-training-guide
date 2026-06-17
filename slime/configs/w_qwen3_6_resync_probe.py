"""DEBUG: faithful TP=2/EP=8 resync→HF probe for the qwen3.6 first-update collapse.

Inherits the 1n training config (same model, ref_load, TP=2/CP=2/EP=8 layout), but:
  - debug_train_only=True  → skip sglang entirely (no rollout engines).
  - save_hf=<path>         → on load, dump HF via the REAL resync converter
                             (HfWeightIteratorDirect = the live Megatron→SGLang
                             gather+convert path).
  - SLIME_SAVE_HF_AND_EXIT → train.py loads ref_load at the real EP=8 layout,
                             calls actor.save_hf(0), and exits before any rollout.

Then diff the dumped HF vs origin HF (modal_train.py::diff_two_hf): wrong params =>
the live TP/EP gather/offset in update_weight/common.py is the bug (and which params
are wrong says TP-gather vs EP-offset); clean => bug is sglang-side weight reception.
"""

from configs import w_qwen3_6_swe_colocate_1n as _base
from configs.base import CHECKPOINTS_PATH, run_tag

modal = _base.modal

_RUN_TAG = run_tag("qwen3.6-resync-probe")


class _Slime(_base._Slime):
    debug_train_only = True
    save_hf = f"{CHECKPOINTS_PATH}/_faithful_resync_hf/iter_{{rollout_id}}"
    use_wandb = False
    environment = {**_base._Slime.environment, "SLIME_SAVE_HF_AND_EXIT": "1"}


slime = _Slime()
