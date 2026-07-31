"""Qwen3.6-27B (DENSE) agentic RL on ``snorkel_private_dataset_1`` — noncolocate, six nodes.

Training sibling of ``w_qwen3_6_27b_snorkel_1_eval``, which measured the base
model on this dataset. Inherits the whole recipe (model, TP4xCP2 training, 16x
TP2 rollout engines, GRPO, optimizer, checkpoint/resume) from
``w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n`` and swaps only the data, so the
snorkel numbers sit on the same scale as the rebench work.

Data. The eval slice is 500 repo-disjoint tasks; this trains on the 3,329-row
complement written by the same ``_holdout`` call, so nothing trained on shares a
codebase, test layout, or build with anything scored. That makes the eval a true
holdout: it stays valid across checkpoints without a re-baseline.

Sandbox concurrency — the snorkel-specific risk. Every task BUILDS its sandbox
from a per-task Dockerfile (rebench pulls prebuilt images), so each episode costs
the Modal client an image build with a context upload, and all episodes run
through the ONE RolloutManager process's single client. The 500-task baseline
eval lost 30% of its episodes (``ImageUnusable``) at 512 in-flight — initially
misread as broken Dockerfiles, until ``modal_build_probe.py`` booted the whole
pool at low per-client load and found 0 of 3,289 graded tasks unbuildable. The
failures were client saturation, clustered by repo only because tasks in a repo
share a Dockerfile and hence a build time. Hence ``sglang_server_concurrency``
is set EXPLICITLY below (slime's default of 512 would mean a 2,048-episode
pool); if episodes still die as ImageUnusable, lower it further and resume.

Build filtering survives as a cheap safety net: ``download_data`` drops pool
rows the probe recorded as durably unbuildable (today: none) and keeps unprobed
ones. ``SNORKEL_SKIP_BUILD_FILTER=1`` skips it; refresh the record with:

    SNORKEL_EVAL_N=500 uv run --no-dev modal run -d slime/modal_build_probe.py::probe

Baseline to beat (base model, 500-task slice, 2026-07-30): 40.0% raw / 57.1%
over the 350 tasks that actually ran — the raw number carries the saturation
losses, the adjusted one is the capability estimate. Hard tasks 38.4%, java
28.9% — the two places with the most headroom.
``skip_eval_before_train=False`` puts a step-0 anchor on the same slice; with
the concurrency cap it should land near 57% raw, and that anchor doubles as the
saturation test for the cap itself.

Score a saved checkpoint offline with ``w_qwen3_6_27b_snorkel_1_eval``
(``EVAL_LOAD=<ckpt dir>``), then compare with ``modal_eval_report.py``.

Checkpoint/resume works exactly as in the rebench parent: a fresh local launch
mints a new stamp and starts clean; a Modal auto-retry reuses it and resumes
from the latest checkpoint. To continue an expired run:

    RESUME=qwen3.6-27b-snorkel-1-noncolocate-5n-<stamp> \
    SNORKEL_EVAL_N=500 EXPERIMENT_CONFIG=w_qwen3_6_27b_snorkel_1_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train

Prereqs: this config's ::download_data (pull, unpack, convert, split, filter) and
the one-time ::convert_hf_to_megatron_checkpoint shared with the rebench configs.

    SNORKEL_EVAL_N=500 EXPERIMENT_CONFIG=w_qwen3_6_27b_snorkel_1_noncolocate_5n \
        uv run --no-dev modal run -d slime/modal_train.py::train
"""

import copy
import json
import os
from pathlib import Path

from configs.base import CHECKPOINTS_PATH, DATA_PATH
from configs.datasets import pull
from configs.w_qwen3_6_27b_snorkel_1_eval import (
    _EVAL_N,
    _EVAL_SEED,
    _EVAL_SLICE,
    _KEY,
    _ROOT,
    _SET_TAG,
    _convert,
    _holdout,
    _report,
    _unpack,
    train_pool_path,
)
from configs.w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n import _LAUNCH_STAMP, _Slime, _eval_entry
from configs.w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n import modal as _parent_modal

# Repo-disjoint complement of the eval slice, and the buildable subset of it.
_TRAIN_POOL = train_pool_path(_EVAL_N, _EVAL_SEED)
_BUILDABLE_IDS = f"{DATA_PATH}/{_KEY}/buildable_ids.json"
_TRAIN_BUILDABLE = f"{DATA_PATH}/{_KEY}/train.holdout{_EVAL_N}.{_EVAL_SEED}.buildable.jsonl"

# Escape hatch: train on the unfiltered pool, skipping the probe's verdict file.
_SKIP_FILTER = os.environ.get("SNORKEL_SKIP_BUILD_FILTER") == "1"
_TRAIN_DATA = _TRAIN_POOL if _SKIP_FILTER else _TRAIN_BUILDABLE

_RESUME = os.environ.get("RESUME")
_RUN_TAG = (
    f"{os.environ.get('WANDB_GROUP') or 'qwen3.6-27b-snorkel-1-noncolocate-5n'}"
    f"{'-nofilter' if _SKIP_FILTER else ''}-{_LAUNCH_STAMP}"
)

# SNORKEL_* are read at config import, which happens inside the container too, so
# they ride in the image env alongside the parent's LAUNCH_STAMP/RESUME.
modal = copy.copy(_parent_modal)
modal.image_env = {
    **_parent_modal.image_env,
    **{
        k: v
        for k in ("SNORKEL_EVAL_N", "SNORKEL_BATCHES", "SNORKEL_UNPACK_WORKERS", "SNORKEL_SKIP_BUILD_FILTER")
        if (v := os.environ.get(k)) is not None
    },
}


def _filter_buildable() -> None:
    """Drop pool rows whose image the probe could not build.

    Tasks the probe never reached are KEPT: an unprobed task is unknown, not
    known-bad, and dropping it would silently shrink the pool if the probe was
    interrupted. The coverage line says how much of the pool the decision rests on.
    """
    ids_file = Path(_BUILDABLE_IDS)
    if not ids_file.is_file():
        raise FileNotFoundError(
            f"{ids_file}: run slime/modal_build_probe.py::probe first, "
            "or set SNORKEL_SKIP_BUILD_FILTER=1 to train on the unfiltered pool"
        )
    blob = json.loads(ids_file.read_text())
    buildable, unbuildable = set(blob["buildable"]), set(blob["unbuildable"])

    rows = [ln for ln in Path(_TRAIN_POOL).read_text().splitlines() if ln.strip()]
    kept, dropped, unprobed = [], 0, 0
    for line in rows:
        inst = json.loads(line)["metadata"]["instance_id"]
        if inst in unbuildable:
            dropped += 1
        else:
            kept.append(line)
            unprobed += inst not in buildable

    Path(_TRAIN_BUILDABLE).write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"[snorkel] buildable pool: kept {len(kept)}/{len(rows)}, dropped {dropped} unbuildable -> {_TRAIN_BUILDABLE}")
    print(f"[snorkel]   probe coverage: {len(rows) - unprobed}/{len(rows)} rows probed ({unprobed} unknown, kept)")


class _SlimeSnorkelTrain(_Slime):
    # ── Data ──────────────────────────────────────────────────────────────────
    prompt_data = _TRAIN_DATA

    # ── Episode concurrency: capped for Dockerfile-building sandboxes ─────────
    # Every episode's sandbox ops (App.lookup + Image.from_dockerfile with a
    # context upload + Sandbox.create) run through the ONE RolloutManager
    # process's single Modal client, and its event loop is what a bare
    # `ImageUnusable` failure actually is: at 512 in-flight the 500-task eval
    # lost 30% of episodes to client saturation, while the build probe proved
    # the same tasks ~100% buildable at low per-client load. The parent leaves
    # this unset, and slime's own default of 512 would mean 512 x 16 engines,
    # clamped to a 2,048-thread episode pool — 4x the load that already failed.
    # 16 x 16 engines = 256 concurrent episodes: half the failing level, and a
    # far gentler creation rate. The in-run eval shares the pool math, so the
    # step-0 anchor doubles as the saturation test — near 57% raw confirms the
    # cap works; ImageUnusable episodes in the logs mean it must drop further.
    # Rebench never needed this because its tasks pull PREBUILT images.
    sglang_server_concurrency = 16

    # Same slice and the same entry name the eval-only config scores, so the
    # in-run curve and the offline base-vs-checkpoint numbers are one series.
    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": [_eval_entry(f"snorkel1_{_SET_TAG}", _EVAL_SLICE)],
    }

    # ── Dynamic sampling ──────────────────────────────────────────────────────
    # The parent's pool was prefiltered to mixed-outcome tasks, so 1.5x
    # oversampling sufficed. This pool is not: it spans easy tasks the base model
    # solves 86% of the time and hard ones it solves 38% of, so a large share of
    # groups come back all-solved or all-failed and get dropped for zero
    # advantage. 2x gives the refill loop more to work with per round. Watch
    # rollout/dynamic_filter/drop_* and perf/rollout_time -- if the drop rate
    # stays low this is just wasted episodes and should come back down.
    over_sampling_batch_size = 64

    # ── Run identity: checkpoints, dumps and W&B all keyed to this launch ─────
    save = f"{CHECKPOINTS_PATH}/swe_ckpts/{_RESUME or _RUN_TAG}"
    load = save
    save_debug_rollout_data = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"
    wandb_group = _RUN_TAG

    def download_data(self) -> None:
        """Pull, unpack, convert, split, and filter the pool to buildable tasks.

        Everything persists on the slime-data volume and every step is
        idempotent, so re-runs only do the missing work. The split is
        deterministic, so this reproduces the exact eval slice the baseline was
        measured on.
        """
        pull(_KEY)
        _unpack(_ROOT)
        _convert(_ROOT)
        _holdout(_ROOT, _EVAL_N, _EVAL_SEED)
        if not _SKIP_FILTER:
            _filter_buildable()
        _report(_ROOT)


slime = _SlimeSnorkelTrain()
