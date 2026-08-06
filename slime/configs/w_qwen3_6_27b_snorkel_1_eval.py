"""Eval-only Qwen3.6-27B run on Snorkel dataset 1.

This inherits the model and runtime recipe from the self-contained Snorkel
training config, then applies only eval-specific overrides. It has no dependency
on a SWE-rebench experiment config.

By default it scores the 60-task holdout paired with the 240-task training set.
Set ``SNORKEL_TOTAL=full`` to score the complete split's holdout.
``EVAL_FULL=1`` scores all converted rows. ``EVAL_TRAIN=<rows>`` scores a
deterministic sample from the selected training pool; ``EVAL_TRAIN=all`` scores
that complete pool.

``EVAL_LOAD`` accepts either a checkpoint run directory or an ``iter_*`` path.
When an iteration path is supplied it is split into the run directory and
``ckpt_step`` so Megatron cannot silently fall back to the base model.

    EXPERIMENT_CONFIG=w_qwen3_6_27b_snorkel_1_eval \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

import copy
import os
import re

from configs.base import CHECKPOINTS_PATH
from configs.datasets import pull
from configs.snorkel_1_data import (
    KEY,
    ROOT,
    convert,
    eval_split_path,
    report,
    split,
    train_pool_path,
    train_slice,
    train_slice_path,
    unpack,
)
from configs.w_qwen3_6_27b_snorkel_1_noncolocate_5n import (
    _DATASET_TOTAL,
    _LAUNCH_STAMP,
    _SET_TAG,
    _Slime,
    _eval_entry,
)
from configs.w_qwen3_6_27b_snorkel_1_noncolocate_5n import modal as _parent_modal

_FULL = os.environ.get("EVAL_FULL") == "1"
_ON_TRAIN = os.environ.get("EVAL_TRAIN") or None
_TRAIN_ROWS = int(_ON_TRAIN) if _ON_TRAIN and _ON_TRAIN != "all" else None

if _ON_TRAIN:
    _SCORED = (
        train_slice_path(_TRAIN_ROWS, _DATASET_TOTAL)
        if _TRAIN_ROWS
        else train_pool_path(_DATASET_TOTAL)
    )
    _SCORED_TAG = f"{_SET_TAG}-train{_TRAIN_ROWS or 'all'}"
elif _FULL:
    _SCORED = str(ROOT / "eval.jsonl")
    _SCORED_TAG = "all"
else:
    _SCORED = eval_split_path(_DATASET_TOTAL)
    _SCORED_TAG = f"{_SET_TAG}-repo20"

_LOAD = os.environ.get("EVAL_LOAD") or None
_STEP = int(os.environ["EVAL_CKPT_STEP"]) if os.environ.get("EVAL_CKPT_STEP") else None
_ITER_DIR = re.fullmatch(r"(.+)/iter_(\d+)", _LOAD.rstrip("/")) if _LOAD else None
if _ITER_DIR:
    _LOAD, _STEP = _ITER_DIR[1], _STEP or int(_ITER_DIR[2])

_MODEL_TAG = _LOAD.rstrip("/").rsplit("/", 1)[-1] if _LOAD else "base"
_RUN_TAG = (
    f"qwen3.6-27b-snorkel-1-eval{_SCORED_TAG}-{_MODEL_TAG}"
    f"{f'-iter{_STEP}' if _STEP is not None else ''}-{_LAUNCH_STAMP}"
)

modal = copy.copy(_parent_modal)
modal.image_env = {
    **_parent_modal.image_env,
    **{
        key: value
        for key in (
            "EVAL_LOAD",
            "EVAL_CKPT_STEP",
            "EVAL_FULL",
            "EVAL_TRAIN",
        )
        if (value := os.environ.get(key)) is not None
    },
}


class _SlimeEval(_Slime):
    prompt_data = _SCORED

    # One TP4xCP2 actor node pushes weights to eight TP2 rollout engines.
    actor_num_nodes = 1
    rollout_num_gpus = 16
    sglang_server_concurrency = 32  # 32 x 8 engines = 256 episodes

    # num_rollout=0 enters slime's one-eval-and-exit branch.
    num_rollout = 0
    eval_interval = 1
    lr_decay_iters = 1
    save = None
    save_interval = None
    load = _LOAD
    ckpt_step = _STEP
    no_load_optim = True
    no_load_rng = True

    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": [_eval_entry(f"snorkel1_{_SCORED_TAG}", _SCORED)],
    }
    wandb_group = _RUN_TAG
    save_debug_rollout_data = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"

    def download_data(self) -> None:
        pull(KEY)
        unpack(ROOT)
        convert(ROOT)
        split(ROOT)
        if _TRAIN_ROWS:
            train_slice(_TRAIN_ROWS, total=_DATASET_TOTAL)
        report(ROOT)


slime = _SlimeEval()
