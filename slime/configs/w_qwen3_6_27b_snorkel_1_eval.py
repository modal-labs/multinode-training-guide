"""Eval-only run of Qwen3.6-27B (DENSE) on ``snorkel_private_dataset_1`` — no training.

First contact with a new dataset: score one set of weights over a slice of it and
exit, so the data path (HF pull -> unpack -> convert -> harbor sandbox -> reward)
is proven end to end before any GPU-hours go into training. Same 6-node topology,
model, and agent env as ``w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n``, so the
numbers sit on the same scale as the rebench work and the one-time Megatron
conversion is already done.

The repo is GATED (manual approval) and is published in harbor's own layout, not
slime's: a ``tasks.csv`` index over 3,829 tasks plus 39 ``tasks/batch_*.zip``
bundles (~47 GB). ``download_data`` therefore does three things — pull, unpack the
bundles in place, and convert each task dir to a slime row — rather than the bare
``pull`` the other configs get. It is registered ``nested=False`` so the pull lands
under ``/data/<key>`` instead of scattering across ``/data``.

Every task ships its own source tree, so the 3,829 of them unpack to ~2.9M files —
this dataset is why ``slime-data`` is a Volumes v2 volume, v1 having capped a
volume at 500,000 inodes (creates past it fail with ENOSPC).

Unlike rebench, every task BUILDS its sandbox from a per-task Dockerfile (no
prebuilt image), so first-touch episodes pay a docker build. The set is also
multilingual — go, java, javascript, python, rust, typescript and more — where the
rebench pool was Python only.

Step 1 — pull, unpack, convert. Idempotent and incremental: the bundles and the
unpacked trees persist on the ``slime-data`` volume across runs. Start with a
couple of bundles to prove the path, then re-run without the limit for all 39:

    SNORKEL_BATCHES=2 EXPERIMENT_CONFIG=w_qwen3_6_27b_snorkel_1_eval \
        uv run --no-dev modal run slime/modal_train.py::download_data

Step 2 — score it. ``SNORKEL_EVAL_N`` sizes the repo-disjoint eval slice (default
100; the slice must have been materialized by a download_data run with the same
value, since step 1 is what writes it) and
``EVAL_FULL=1`` scores all 3,829 rows instead. ``EVAL_LOAD`` points at a Megatron
checkpoint dir; leave it unset to score the base model via ``ref_load``. All are
read at config import, in-container too, so they ride in the image env:

    SNORKEL_EVAL_N=500 EXPERIMENT_CONFIG=w_qwen3_6_27b_snorkel_1_eval \
        uv run --no-dev modal run -d slime/modal_train.py::train

Every task builds its sandbox from a per-task Dockerfile, and Modal caches those
builds by content hash — so the first pass over a given slice is much slower than
repeats, and a full pass warms the images for every later run.

Prereq: the one-time ``::convert_hf_to_megatron_checkpoint`` (shared with the
rebench configs — already done if you have run those).
"""

import copy
import csv
import json
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from configs.base import CHECKPOINTS_PATH, DATA_PATH
from configs.datasets import eval_path, pull
from configs.w_qwen3_6_27b_swe_rebench_v2_eval import _SlimeEval
from configs.w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n import _LAUNCH_STAMP, _eval_entry
from configs.w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n import modal as _parent_modal

_KEY = "snorkel_private_dataset_1"
_SLIME_ROOT = "/root/slime"  # the local_slime overlay; download_data doesn't add it to sys.path
_ROOT = DATA_PATH / _KEY

# Bundles to unpack (unset = all 39). A couple is enough to prove the path.
_BATCHES = int(os.environ["SNORKEL_BATCHES"]) if os.environ.get("SNORKEL_BATCHES") else None
# Volumes v1 topped out around 5 concurrent writers; v2 takes hundreds when they
# write distinct files, and bundles own disjoint task ids.
_UNPACK_WORKERS = int(os.environ.get("SNORKEL_UNPACK_WORKERS", "32"))
# Eval slice size, carved from the converted rows by datasets.subsample.
_EVAL_N, _EVAL_SEED = int(os.environ.get("SNORKEL_EVAL_N", "100")), 0
# EVAL_FULL=1 scores all 3,829 rows instead of the slice. A slice is the cheaper
# read: at a ~10% solve rate 500 tasks already pins it to about +/-1.3pp, while the
# full set costs ~8x the episodes — and every first-touch task pays a docker build.
_FULL = os.environ.get("EVAL_FULL") == "1"
_EVAL_SLICE = eval_path(_KEY) if _FULL else eval_path(_KEY, _EVAL_N, _EVAL_SEED)
_SET_TAG = "full" if _FULL else str(_EVAL_N)

_LOAD = os.environ.get("EVAL_LOAD") or None
_MODEL_TAG = _LOAD.rstrip("/").rsplit("/", 1)[-1] if _LOAD else "base"
_RUN_TAG = f"qwen3.6-27b-snorkel-1-eval{_SET_TAG}-{_MODEL_TAG}-{_LAUNCH_STAMP}"

modal = copy.copy(_parent_modal)
modal.image_env = {
    **_parent_modal.image_env,
    **{
        k: v
        for k in ("EVAL_LOAD", "EVAL_FULL", "SNORKEL_EVAL_N", "SNORKEL_BATCHES")
        if (v := os.environ.get(k)) is not None
    },
}


def _unpack(root: Path) -> None:
    """Extract the batch bundles in place. Each zip holds ``<task_id>/`` dirs, so
    they land as ``tasks/<task_id>/`` — already the layout task_path wants.

    Threaded over bundles: a bundle is ~1 GB of zip expanding to ~73k small files
    on a network-backed volume, so this is I/O bound and zlib drops the GIL while
    decompressing. Bundles own disjoint task ids, so they never race.
    """
    tasks = root / "tasks"
    every = sorted(tasks.glob("batch_*.zip"))
    bundles = every[:_BATCHES]
    print(f"[snorkel] unpacking {len(bundles)} of {len(every)} bundles, {_UNPACK_WORKERS} at a time")

    def one(bundle: Path) -> str:
        with zipfile.ZipFile(bundle) as zf:
            names = zf.namelist()
            tops = sorted({n.split("/")[0] for n in names if "/" in n})
            if all((tasks / t / "task.toml").is_file() for t in tops):
                return f"{bundle.name}: {len(tops)} task dirs already unpacked"
            zf.extractall(tasks)
        return f"{bundle.name}: {len(tops)} task dirs"

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=_UNPACK_WORKERS) as pool:
        for done, msg in enumerate(pool.map(one, bundles), start=1):
            print(f"[snorkel]   [{done}/{len(bundles)}] {msg} ({time.monotonic() - started:.0f}s elapsed)")


def _convert(root: Path) -> None:
    """Harbor task dirs -> slime rows in ``eval.jsonl``.

    Deliberately not ``convert2slime.harbor.convert``: that copytrees every task
    into a fresh tree, which would duplicate tens of GB on the volume when the
    bundles already unpack into their final location. We reuse its per-task
    translation and set ``task_path`` ourselves.
    """
    if _SLIME_ROOT not in sys.path:
        sys.path.insert(0, _SLIME_ROOT)
    from agentic_rl.environment.convert2slime.harbor import SkipTask, translate_task

    tasks = root / "tasks"
    # No task.toml probe here: translate_task already raises SkipTask for it, and
    # the probe costs a serial round trip per entry.
    task_dirs = sorted(p for p in tasks.iterdir() if p.is_dir())
    print(f"[snorkel] converting {len(task_dirs)} unpacked task dirs, {_UNPACK_WORKERS} at a time")

    # tasks.csv carries language/difficulty/category per task_id; stamp it onto
    # the row so eval dumps can be sliced by them later. Inert to the rollout.
    index: dict[str, dict[str, str]] = {}
    csv_path = root / "tasks.csv"
    if csv_path.is_file():
        with csv_path.open() as fh:
            index = {r["task_id"]: r for r in csv.DictReader(fh)}

    # translate_task is ~10 stat/read round trips per task against a network
    # volume, so serially this is ~30min of pure latency for 3.8k tasks. It only
    # reads and releases the GIL on every syscall, so fan it out. pool.map yields
    # in input order, keeping eval.jsonl row order -- and the holdout draw that
    # depends on it -- identical to the serial version.
    def _one(task_dir: Path) -> tuple[Path, dict | None, str | None]:
        try:
            return task_dir, translate_task(task_dir, dataset=_KEY), None
        except SkipTask as e:
            return task_dir, None, str(e)

    rows, skipped = [], Counter()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=_UNPACK_WORKERS) as pool:
        for done, (task_dir, row, reason) in enumerate(pool.map(_one, task_dirs), start=1):
            if done % 500 == 0 or done == len(task_dirs):
                print(f"[snorkel]   [{done}/{len(task_dirs)}] ({time.monotonic() - started:.0f}s elapsed)")
            if reason:
                skipped[reason] += 1
                continue
            # The trees were unpacked in place, so task_path follows the DIRECTORY
            # name; instance_id comes from task.toml and need not match it.
            row["metadata"]["task_path"] = f"{_KEY}/tasks/{task_dir.name}"
            if meta := index.get(task_dir.name):
                row["metadata"]["snorkel"] = {
                    k: meta[k] for k in ("language", "language_bucket", "category", "difficulty") if k in meta
                }
            rows.append(row)

    out = root / "eval.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"[snorkel] wrote {len(rows)} rows -> {out}")
    if skipped:
        print(f"[snorkel] skipped {sum(skipped.values())}: {dict(skipped)}")
    langs = Counter((r["metadata"].get("snorkel") or {}).get("language_bucket", "?") for r in rows)
    diffs = Counter((r["metadata"].get("snorkel") or {}).get("difficulty", "?") for r in rows)
    print(f"[snorkel] language buckets: {dict(langs)}")
    print(f"[snorkel] difficulty: {dict(diffs)}")


def train_pool_path(n: int, seed: int) -> str:
    """Complement of the ``n``-row eval slice: everything not held out. Named for
    the split that produced it, since ``train.jsonl`` would imply the full set."""
    return f"{DATA_PATH}/{_KEY}/train.holdout{n}.{seed}.jsonl"


def _holdout(root: Path, n: int, seed: int) -> Path:
    """Carve an eval slice of at most ``n`` rows, keeping each repo whole, and
    write the complement as the training pool.

    Task ids are ``<owner>_<repo>__<pr>``, so a row-level sample scatters a repo's
    issues across both sides of the split. Whole repos in or out keeps the
    complement usable as a training set: nothing trained on shares a codebase,
    test layout, or build with anything scored here, so the eval number stays a
    valid holdout instead of needing a re-baseline.
    """
    src = root / "eval.jsonl"
    rows = [json.loads(ln) for ln in src.read_text().splitlines() if ln.strip()]

    def repo_of(row: dict) -> str:
        return re.sub(r"__\d+$", "", Path(row["metadata"]["task_path"]).name)

    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(repo_of(row), []).append(i)

    # Whole repos, shuffled deterministically. Repos too big for the remaining
    # budget are skipped rather than split; the 600-odd single-task repos fill
    # the tail, so the slice lands exactly on n.
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    picked: list[int] = []
    chosen: list[str] = []
    for repo in order:
        if len(picked) == n:
            break
        if len(picked) + len(groups[repo]) <= n:
            picked += groups[repo]
            chosen.append(repo)
    picked.sort()

    if sum(len(groups[r]) for r in chosen) != len(picked):
        raise RuntimeError("repo split leaked: a chosen repo did not contribute all its tasks")

    out = Path(eval_path(_KEY, n, seed))
    out.write_text("".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in picked), encoding="utf-8")
    print(f"[snorkel] holdout: {len(picked)} rows from {len(chosen)} repos -> {out}")

    held = set(picked)
    rest = [i for i in range(len(rows)) if i not in held]
    pool = Path(train_pool_path(n, seed))
    pool.write_text("".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in rest), encoding="utf-8")
    print(f"[snorkel]   train pool: {len(rest)} rows from {len(groups) - len(chosen)} repos, repo-disjoint -> {pool}")
    for label, key in (("language", "language"), ("difficulty", "difficulty")):
        full = Counter((r["metadata"].get("snorkel") or {}).get(key, "?") for r in rows)
        cut = Counter((rows[i]["metadata"].get("snorkel") or {}).get(key, "?") for i in picked)
        mix = ", ".join(
            f"{k} {100 * cut[k] / max(len(picked), 1):.0f}%/{100 * v / max(len(rows), 1):.0f}%"
            for k, v in full.most_common(6)
        )
        print(f"[snorkel]   {label} (slice/full): {mix}")
    return out


def _report(root: Path) -> None:
    """Print the dataset's shape. Structure and counts only — task content stays
    out of the logs."""
    if not root.is_dir():
        print(f"[snorkel] {root} missing after pull")
        return
    print(f"[snorkel] {root} top level: {sorted(p.name + ('/' if p.is_dir() else '') for p in root.iterdir())}")

    unpacked = root / "tasks"
    if unpacked.is_dir():
        dirs = sum(1 for p in unpacked.iterdir() if p.is_dir())
        zips = len(list(unpacked.glob("*.zip")))
        print(f"[snorkel] tasks/: {dirs} unpacked task dirs, {zips} bundles")

    for jsonl in sorted(root.glob("*.jsonl")):
        rows = [ln for ln in jsonl.read_text().splitlines() if ln.strip()]
        print(f"[snorkel] {jsonl.name}: {len(rows)} rows")
        if not rows:
            continue
        parsed = [json.loads(r) for r in rows]
        md = parsed[0].get("metadata") or {}
        print(f"[snorkel]   row keys: {sorted(parsed[0])} | metadata keys: {sorted(md)}")
        srcs = Counter(
            "docker_image"
            if (p.get("metadata") or {}).get("docker_image")
            else "dockerfile"
            if (p.get("metadata") or {}).get("dockerfile")
            else "NEITHER"
            for p in parsed
        )
        print(f"[snorkel]   image source: {dict(srcs)}")
        missing = sum(
            1 for p in parsed if not (DATA_PATH / str((p.get("metadata") or {}).get("task_path", ""))).is_dir()
        )
        print(f"[snorkel]   rows whose task dir is missing under {DATA_PATH}: {missing}/{len(parsed)}")


class _SlimeSnorkelEval(_SlimeEval):
    # Unused at num_rollout=0, but slime still builds the train dataset from it.
    prompt_data = _EVAL_SLICE

    # ── Topology: 3 nodes instead of the parent's 6 ────────────────────────────
    # train.py still builds the actors in the eval-only branch (they read the
    # checkpoint and push weights to sglang), but scoring loads no optimizer
    # state, grads, or activations, so one node covers TP4xCP2 at DP1.
    actor_num_nodes = 1
    rollout_num_gpus = 16  # 2 nodes → 16 // 2 = 8 TP2 engines
    # In-flight episodes = sglang_server_concurrency x engines, and the parent's
    # 32x16 = 512 is what lets all 500 tasks run at once. Half the engines, so
    # double the per-engine cap to hold that window. An episode is mostly sandbox
    # time (image builds alone measured 60-320s), not decode, so the engines see
    # far fewer concurrent requests than the admitted episode count suggests.
    sglang_server_concurrency = 64

    eval_config = {
        "defaults": {"n_samples_per_eval_prompt": 1, "temperature": 0.6, "top_p": 1.0},
        "datasets": [_eval_entry(f"snorkel1_{_SET_TAG}", _EVAL_SLICE)],
    }

    wandb_group = _RUN_TAG
    save_debug_rollout_data = f"{CHECKPOINTS_PATH}/swe_rollout_dumps/{_RUN_TAG}/rollout_{{rollout_id}}.pt"

    def download_data(self) -> None:
        """Pull, unpack the batch bundles, convert them to slime rows, and carve
        the eval slice. Everything persists on the slime-data volume, so re-runs
        only do the missing work."""
        pull(_KEY)
        _unpack(_ROOT)
        _convert(_ROOT)
        _holdout(_ROOT, _EVAL_N, _EVAL_SEED)
        _report(_ROOT)


slime = _SlimeSnorkelEval()
