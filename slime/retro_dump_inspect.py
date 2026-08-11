"""Inspect rollout/group/index invariants in a saved retro debug dump.

    MODAL_ENVIRONMENT=junlin-dev uv run modal run slime/retro_dump_inspect.py \
      --path /checkpoints/swe_rollout_dumps/<run>/rollout_0.pt
"""

from __future__ import annotations

import collections
import json

import modal

app = modal.App("frontier-cs-retro-dump-inspect")
image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260529a")
    .entrypoint([])
    .add_local_dir(
        "/Users/junlin/Documents/Research/async-rl/slime",
        remote_path="/root/slime",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "agentic_rl/profiles/**"],
    )
)
checkpoints = modal.Volume.from_name("slime-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoints},
    cpu=2,
    memory=8 * 1024,
    timeout=20 * 60,
)
def inspect_dump(path: str):
    import sys

    import torch

    sys.path.insert(0, "/root/slime")
    from slime.utils.types import Sample

    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = [Sample.from_dict(value) for value in payload.get("samples", [])]
    rollout_counts = collections.Counter(sample.rollout_id for sample in samples)
    index_counts = collections.Counter(sample.index for sample in samples)
    groups: dict[int | None, list[Sample]] = collections.defaultdict(list)
    for sample in samples:
        groups[sample.group_index].append(sample)

    duplicates = []
    for rollout_id, count in rollout_counts.items():
        if count <= 1:
            continue
        duplicates.append(
            {
                "rollout_id": rollout_id,
                "count": count,
                "samples": [
                    {
                        "index": sample.index,
                        "group_index": sample.group_index,
                        "retro": bool((sample.metadata or {}).get("agentic", {}).get("retro_branch")),
                        "instance_id": (sample.metadata or {}).get("instance_id"),
                    }
                    for sample in samples
                    if sample.rollout_id == rollout_id
                ],
            }
        )

    result = {
        "path": path,
        "rollout_id": payload.get("rollout_id"),
        "samples": len(samples),
        "unique_rollout_ids": len(rollout_counts),
        "unique_indices": len(index_counts),
        "groups": len(groups),
        "group_sizes": collections.Counter(len(group) for group in groups.values()),
        "duplicate_rollout_ids": duplicates,
        "duplicate_indices": {
            str(index): count for index, count in index_counts.items() if count > 1
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main(path: str):
    inspect_dump.remote(path)
