"""Dataset registry: one HF repo per dataset, one ``/data/<key>/`` subdir.

Single source of truth tying together, from one ``key``:
  * the HF dataset repo,
  * the ``/data/<key>/`` subdir (where ``download_data`` lands it),
  * the ``task_path`` prefix the converter bakes (``<key>/tasks/<id>``),
  * the train/eval jsonl paths.

Each repo stores its content nested under ``<key>/`` (built by
``environment/convert2slime``; published with ``path_in_repo=<key>``), so a single
``snapshot_download(repo, local_dir=/data)`` puts it at ``/data/<key>/`` and
``task_path=<key>/tasks/<id>`` is self-consistent on HF and on the volume.

Per-repo layout::

    <key>/train.jsonl          # train datasets only
    <key>/eval.jsonl           # held-out; eval-only datasets ship just this
    <key>/tasks/<id>/...        # ONE shared task tree, indexed by both jsonls
    <key>/problems/<pid>/...    # frontier_cs only: judge testdata (read by the verifier server)

Slices are NOT committed — ``download_data`` subsamples ``eval.jsonl`` in-config
(``subsample``), and ``eval_path(key, n)`` returns the deterministic subsample path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from configs.base import DATA_PATH


@dataclass(frozen=True)
class Dataset:
    repo: str  # HF dataset repo id (content nested under the registry key)
    has_train: bool = False  # ships <key>/train.jsonl
    revision: str | None = None  # pin a commit/tag for reproducibility


# key (== /data subdir == task_path prefix) -> dataset. Add a line per dataset.
# Keys match the existing on-disk/HF dir names so publishing is a clean repackage.
DATASETS: dict[str, Dataset] = {
    "swegym_lite": Dataset("junlin-modal/swegym-lite", has_train=True),
    "frontier_cs": Dataset("junlin-modal/frontier-cs", has_train=True),
    "openthoughts_agent": Dataset("junlin-modal/openthoughts-agent", has_train=True),
    "usaco": Dataset("junlin-modal/usaco", has_train=True),  # train (usaco config) + transfer-eval
    "terminal_bench_2_1": Dataset("junlin-modal/terminal-bench-2.1"),
    "swebench_verified": Dataset("junlin-modal/swebench-verified"),
    "swebench_multilingual": Dataset("junlin-modal/swebench-multilingual"),
    "openthoughts_tblite": Dataset("junlin-modal/openthoughts-tblite"),
    "swebenchpro": Dataset("junlin-modal/swebenchpro"),
    "swe_rebench_v2": Dataset("junlin-modal/swe-rebench-v2", has_train=True),  # nebius/SWE-rebench-V2, python subset
}


def train_path(key: str) -> str:
    return f"{DATA_PATH}/{key}/train.jsonl"


def eval_path(key: str, n: int | None = None, seed: int = 0) -> str:
    """Eval jsonl path for ``key``; the deterministic subsample file when ``n`` is
    set (materialized by ``subsample`` in ``download_data``). Safe at config import
    (returns the path; reads nothing)."""
    if n is None:
        return f"{DATA_PATH}/{key}/eval.jsonl"
    return f"{DATA_PATH}/{key}/eval.{n}.{seed}.jsonl"


def eval_datasets(specs: list[tuple[str, int | None]]) -> list[dict]:
    """Build ``eval_config['datasets']`` from ``(key, n|None)`` specs."""
    out = []
    for key, n in specs:
        name = f"{key}_{n}" if n is not None else key
        out.append({"name": name, "path": eval_path(key, n), "metadata_overrides": {"eval_dataset": name}})
    return out


def pull(key: str, *, allow_patterns: list[str] | None = None) -> str:
    """``snapshot_download`` the dataset repo into ``/data`` (nests under ``<key>/``)."""
    from huggingface_hub import snapshot_download

    ds = DATASETS[key]
    path = snapshot_download(
        ds.repo, repo_type="dataset", local_dir=str(DATA_PATH), revision=ds.revision, allow_patterns=allow_patterns
    )
    print(f"[datasets] pulled {key} ({ds.repo}) -> {DATA_PATH}/{key}")
    return path


def subsample(key: str, n: int, seed: int = 0) -> str:
    """Deterministically subsample ``<key>/eval.jsonl`` -> ``<key>/eval.<n>.<seed>.jsonl``.

    Run in ``download_data`` after ``pull``. Passthrough (copy) when ``n >= rows``.
    """
    src = Path(f"{DATA_PATH}/{key}/eval.jsonl")
    rows = [line for line in src.read_text().splitlines() if line.strip()]
    if n < len(rows):
        idx = sorted(random.Random(seed).sample(range(len(rows)), n))
        rows = [rows[i] for i in idx]
    out = Path(eval_path(key, n, seed))
    out.write_text("\n".join(rows) + "\n")
    print(f"[datasets] subsampled {key} eval {len(rows)} rows -> {out}")
    return str(out)
