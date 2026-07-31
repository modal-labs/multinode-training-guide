"""Eval-only run of Qwen3.6-27B (DENSE) on SWE-rebench-V2 — no training.

Scores ONE set of weights and exits, for the base-vs-checkpoint comparison the
training run's inline eval curve only hints at. Reuses
``w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n``'s topology (2 train + 4 rollout
nodes, TP4xCP2 training, 16x TP2 rollout engines) and its eval slice, so the
numbers are directly comparable to that run's curve. ``num_rollout = 0`` routes
through train.py's eval-only branch (eval once, exit).

Which weights: ``EVAL_LOAD`` points at a Megatron checkpoint dir; leave it unset
to score the base model (slime falls back to ``ref_load``). Which tasks:
``EVAL_FULL=1`` swaps the 500-task slice for the whole 7,243-row train pool, and
``SWEBENCH_EVAL=1`` (the parent's flag) adds SWE-bench Verified alongside it —
the usual choice for a base-vs-checkpoint benchmark number.

    # baseline
    EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_eval \
        uv run --no-dev modal run -d slime/modal_train.py::train

    # a checkpoint from the training run
    EVAL_LOAD=/checkpoints/swe_ckpts/<run tag> \
    EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_eval \
        uv run --no-dev modal run -d slime/modal_train.py::train

Prereqs: the parent config's ::download_data (datasets + the eval slice) and the
one-time ::convert_hf_to_megatron_checkpoint.
"""

import copy
import os

from configs.base import CHECKPOINTS_PATH
from configs.datasets import eval_path, train_path
from configs.w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n import (  # noqa: F401
    _EVAL_N,
    _EVAL_SLICE,
    _LAUNCH_STAMP,
    _SWEBENCH,
    _SWEBENCH_EVAL,
    _TRAIN,
    _Slime,
    _eval_entry,
)
from configs.w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n import modal as _parent_modal

# Weights under test; unset → base model via ref_load.
_LOAD = os.environ.get("EVAL_LOAD") or None
# Full 7,243-task train pool instead of the fixed 500-task slice.
_FULL = os.environ.get("EVAL_FULL") == "1"

# SWEBENCH_EVAL=1 (inherited from the parent) adds the 500-task Verified set.
_DATASETS = [
    _eval_entry("rebench_full", train_path(_TRAIN))
    if _FULL
    else _eval_entry(f"rebench_prefilter{_EVAL_N}", _EVAL_SLICE)
]
if _SWEBENCH_EVAL:
    _DATASETS.append(_eval_entry(_SWEBENCH, eval_path(_SWEBENCH)))

_MODEL_TAG = _LOAD.rstrip("/").rsplit("/", 1)[-1] if _LOAD else "base"
_SET_TAG = ("full" if _FULL else f"prefilter{_EVAL_N}") + ("-swebench" if _SWEBENCH_EVAL else "")
_RUN_TAG = f"qwen3.6-27b-swe-rebench-v2-eval-{_SET_TAG}-{_MODEL_TAG}-{_LAUNCH_STAMP}"

# EVAL_LOAD/EVAL_FULL are read at config import, which happens INSIDE the
# container too, so they have to ride along in the image env like LAUNCH_STAMP
# (the parent already forwards SWEBENCH_EVAL/SWEBENCH_TRAIN).
modal = copy.copy(_parent_modal)
modal.image_env = {
    **_parent_modal.image_env,
    **{k: v for k in ("EVAL_LOAD", "EVAL_FULL") if (v := os.environ.get(k)) is not None},
}


class _SlimeEval(_Slime):
    # async_mode=False already, so num_rollout=0 routes through train.py's
    # eval-only branch: one eval at rollout 0, then dispose.
    num_rollout = 0
    eval_interval = 1  # any non-None value arms the eval-only branch
    sglang_server_concurrency = 32

    # slime derives train_iters from num_rollout; pin a dummy 1-iter schedule so
    # Megatron's `assert lr_decay_steps > 0` passes (no optimizer step runs).
    lr_decay_iters = 1

    # Nothing to checkpoint here, so drop the parent's save wiring (save_interval
    # asserts --save is set) and load exactly the weights we were handed.
    save = None
    save_interval = None
    load = _LOAD  # None → --load omitted → slime falls back to ref_load
    no_load_optim = True  # scoring only; skip reading the optimizer shards
    no_load_rng = True

    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": _DATASETS,
    }

    wandb_group = _RUN_TAG
    save_debug_rollout_data = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"


slime = _SlimeEval()
