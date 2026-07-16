"""Prefilter eval: Qwen3.6-27B x 16 rollouts on EVERY SWE-rebench-V2 TRAIN task.

Purpose: prefilter the training benchmark. GRPO gets zero advantage from a
prompt group whose rollouts are uniformly all-solved or all-failed, so this run
measures, per train task, how many of 16 independent BASE-model rollouts solve
it, then records the ids of the mixed-outcome tasks (0 < solved < 16). The
training config (``w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n``) trains on
that subset only: its ``download_data`` filters ``train.jsonl`` by the id list
this config produces.

Reuses the training config's topology and engines (6 nodes, ``async_mode =
False``); ``num_rollout = 0`` routes through train.py's eval-only branch (eval
once, exit). Sampling matches TRAINING rollouts (temperature 1.0), not the
usual 0.6 eval default — the prefilter keys off group variance under the
policy training will actually sample.

Scale + sharding: train.jsonl has 7,243 tasks -> ~116k episodes at 16 samples
each. One run would flirt with the 24h Modal function timeout, so the set is
split into PREFILTER_NUM_SHARDS round-robin shards (default 4, ~29k episodes
each) and one ::train launch evals one shard (PREFILTER_SHARD=0..N-1). Shards
can run sequentially or in parallel (each launch is its own Modal app). Every
shard dumps its eval samples to a FIXED (untimestamped) dir on the checkpoints
volume; ::post_process_data aggregates ALL shard dumps and writes the id list
to the data volume at ``/data/swe_rebench_v2/prefilter_ids.json``.

Workflow (export PREFILTER_NUM_SHARDS for every step if not using the default 4):

    export EXPERIMENT_CONFIG=w_qwen3_6_27b_swe_rebench_v2_prefilter_eval

    # 1. pull the repo + materialize the shard files (once)
    uv run --no-dev modal run slime/modal_train.py::download_data

    # 2. eval each shard (repeat for PREFILTER_SHARD=0..3)
    PREFILTER_SHARD=0 uv run --no-dev modal run -d slime/modal_train.py::train

    # 3. aggregate all shard dumps -> /data/swe_rebench_v2/prefilter_ids.json
    uv run --no-dev modal run slime/modal_train.py::post_process_data

Failure accounting: an episode that never produced a gradeable trajectory
(sandbox/image boot failure, context blowup, ...) counts as UNSOLVED — so a
task whose image never boots grades all-failed and is excluded, which is the
same (zero) signal training would see on it.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

from configs.base import CHECKPOINTS_PATH, DATA_PATH, ModalConfig, run_tag
from configs.datasets import pull, train_path

from configs.w_qwen3_6_27b_swe_rebench_v2_noncolocate_5n import (
    _Slime,
    modal as _train_modal,
)

_TRAIN = "swe_rebench_v2"
_N_SAMPLES = 16  # rollouts per train task; the mixed-outcome criterion is 0 < solved < 16

_NUM_SHARDS = int(os.environ.get("PREFILTER_NUM_SHARDS", "4"))
_SHARD = int(os.environ.get("PREFILTER_SHARD", "0"))
assert 0 <= _SHARD < _NUM_SHARDS, f"PREFILTER_SHARD={_SHARD} out of range for {_NUM_SHARDS} shards"

# FIXED (untimestamped) dump root: ::post_process_data must find every shard's
# eval dump after its app exits, and a relaunched shard overwrites its own dump
# instead of forking a new timestamped dir.
_DUMP_ROOT = Path(f"{CHECKPOINTS_PATH}/swe_rollout_dumps/qwen3.6-27b-swe-rebench-v2-prefilter")
_PREFILTER_IDS = Path(f"{DATA_PATH}/{_TRAIN}/prefilter_ids.json")

_RUN_TAG = run_tag(f"qwen3.6-27b-swe-rebench-v2-prefilter-s{_SHARD}of{_NUM_SHARDS}")


def _shard_path(shard: int, num_shards: int) -> str:
    return f"{DATA_PATH}/{_TRAIN}/train.shard{shard}of{num_shards}.jsonl"


# ::train re-imports this config inside the remote containers, so the shard
# selection must survive into the image env alongside the parent's vars.
modal = ModalConfig(**vars(_train_modal))
modal.image_env = {
    **modal.image_env,
    "PREFILTER_SHARD": str(_SHARD),
    "PREFILTER_NUM_SHARDS": str(_NUM_SHARDS),
}


class _SlimeEval(_Slime):
    # async_mode=False already, so num_rollout=0 routes through train.py's
    # eval-only branch (eval once, exit).
    num_rollout = 0
    eval_interval = 1  # any non-None value arms the eval-only branch

    # slime derives train_iters from num_rollout; pin a dummy 1-iter schedule so
    # Megatron's `assert lr_decay_steps > 0` passes (no optimizer step runs).
    lr_decay_iters = 1

    # The parent's prompt_data points at the PREFILTERED train jsonl, which does
    # not exist until this config has produced prefilter_ids.json — and the train
    # data source still loads at startup even in eval-only mode, so point it back
    # at the raw file.
    prompt_data = train_path(_TRAIN)

    eval_config = {
        "defaults": {
            "n_samples_per_eval_prompt": _N_SAMPLES,
            # TRAINING sampling, not the 0.6 eval default: the prefilter keys off
            # group variance under the policy training will sample from.
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "datasets": [
            {
                "name": f"{_TRAIN}_train_s{_SHARD}of{_NUM_SHARDS}",
                "path": _shard_path(_SHARD, _NUM_SHARDS),
                "metadata_overrides": {"eval_dataset": f"{_TRAIN}_train_s{_SHARD}of{_NUM_SHARDS}"},
            }
        ],
    }

    wandb_group = _RUN_TAG
    save_debug_rollout_data = f"{_DUMP_ROOT}/shard{_SHARD}of{_NUM_SHARDS}/rollout_{{rollout_id}}.pt"

    def download_data(self) -> None:
        """Pull the repo, then materialize ALL round-robin shard files (run once;
        every shard's ::train launch reads its own file)."""
        pull(_TRAIN)
        rows = [line for line in Path(train_path(_TRAIN)).read_text().splitlines() if line.strip()]
        for shard in range(_NUM_SHARDS):
            out = Path(_shard_path(shard, _NUM_SHARDS))
            chunk = rows[shard::_NUM_SHARDS]
            out.write_text("\n".join(chunk) + "\n")
            print(f"[prefilter] shard {shard}/{_NUM_SHARDS}: {len(chunk)} rows -> {out}")

    def post_process_data(self) -> None:
        """Aggregate every shard's eval dump into the mixed-outcome id list.

        One episode = one (instance_id, sample.index) pair — an episode's sibling
        chains share `index`, and any chain carries the episode's `is_solved`.
        Writes /data/swe_rebench_v2/prefilter_ids.json with the ids where
        0 < solved < 16, plus per-instance solve counts for the record.
        """
        import gc

        import torch

        dumps = sorted(_DUMP_ROOT.glob("shard*of*/rollout_eval_0.pt"))
        if not dumps:
            raise RuntimeError(f"no eval dumps under {_DUMP_ROOT}; run ::train for each shard first")
        layout: dict[int, set[int]] = {}
        for dump in dumps:
            if m := re.fullmatch(r"shard(\d+)of(\d+)", dump.parent.name):
                layout.setdefault(int(m.group(2)), set()).add(int(m.group(1)))
        if len(layout) != 1:
            raise RuntimeError(f"mixed shard layouts under {_DUMP_ROOT}: {sorted(layout)}; clean up stale dirs")
        ((n_shards, present),) = layout.items()
        if missing := set(range(n_shards)) - present:
            raise RuntimeError(f"missing shard dump(s) {sorted(missing)} of {n_shards}; eval those shards first")

        per_instance: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # iid -> [solved, episodes]
        for dump in dumps:
            samples = torch.load(dump, weights_only=False)["samples"]
            episodes: dict[tuple, bool] = {}
            for s in samples:
                md = s.get("metadata") or {}
                iid = md.get("instance_id") or s.get("label")
                solved = bool((md.get("agentic") or {}).get("is_solved"))
                key = (iid, s.get("index"))
                episodes[key] = episodes.get(key, False) or solved
            for (iid, _), solved in episodes.items():
                per_instance[iid][0] += int(solved)
                per_instance[iid][1] += 1
            print(f"[prefilter] {dump.parent.name}: {len(samples)} samples, {len(episodes)} episodes")
            del samples, episodes
            gc.collect()

        short = {iid: tuple(c) for iid, c in per_instance.items() if c[1] != _N_SAMPLES}
        if short:
            print(f"[prefilter] WARNING: {len(short)} instances with != {_N_SAMPLES} episodes, e.g. {list(short.items())[:5]}")

        mixed = sorted(iid for iid, (w, n) in per_instance.items() if 0 < w < n)
        n_all_solved = sum(1 for w, n in per_instance.values() if w == n)
        n_all_failed = sum(1 for w, _ in per_instance.values() if w == 0)

        payload = {
            "model": _SlimeEval.hf_checkpoint,
            "n_samples": _N_SAMPLES,
            "temperature": 1.0,
            "criterion": f"0 < solved < n_episodes over {_N_SAMPLES} rollouts (binary is_solved)",
            "num_instances": len(per_instance),
            "num_mixed": len(mixed),
            "num_all_solved": n_all_solved,
            "num_all_failed": n_all_failed,
            "instance_ids": mixed,
            "solve_counts": {iid: per_instance[iid][0] for iid in sorted(per_instance)},
        }
        _PREFILTER_IDS.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"[prefilter] {len(mixed)}/{len(per_instance)} mixed-outcome tasks -> {_PREFILTER_IDS} "
            f"(all-solved={n_all_solved}, all-failed={n_all_failed})"
        )


slime = _SlimeEval()
