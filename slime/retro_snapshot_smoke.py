"""Low-cost Modal Sandbox snapshot fidelity/latency smoke.

Run from the multinode-training-guide root:

    MODAL_ENVIRONMENT=junlin-dev uv run modal run slime/retro_snapshot_smoke.py

This uses CPU-only Debian sandboxes.  It tests directory and filesystem
snapshots, eight isolated writable directory restores, and explicit cleanup.
It does not provision the 48-H200 training cluster.
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import time
import uuid

import modal
import modal.experimental

app = modal.App("frontier-cs-retro-snapshot-smoke")
image = modal.Image.debian_slim()


def _sandbox(*, source_image=None):
    return modal.Sandbox.create(
        "sleep",
        "infinity",
        app=app,
        image=source_image or image,
        cpu=0.25,
        memory=512,
        timeout=15 * 60,
    )


def _exec(sb, command: str) -> tuple[int, str, str]:
    process = sb.exec("bash", "-lc", command, text=True, timeout=120)
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    return process.wait(), stdout, stderr


def _check(sb, command: str) -> str:
    returncode, stdout, stderr = _exec(sb, command)
    if returncode != 0:
        raise RuntimeError(f"smoke command (rc={returncode}): {(stderr or stdout)[-500:]}")
    return stdout


def _branch(directory_snapshot_id: str, branch: int) -> dict:
    sb = _sandbox()
    started = time.perf_counter()
    try:
        sb.mount_image("/mnt/retro", modal.Image.from_id(directory_snapshot_id))
        _check(sb, "mkdir -p /app && cp -a /mnt/retro/. /app/")
        _check(sb, f"printf '\\n// branch {branch}\\n' >> /app/solution.cpp")
        digest = _check(sb, "cksum /app/solution.cpp").strip()
        marker = _check(sb, "cat /app/marker.txt").strip()
        return {
            "branch": branch,
            "restore_seconds": time.perf_counter() - started,
            "marker": marker,
            "digest": digest,
        }
    finally:
        sb.terminate()


def _snapshot_directory(sb, path: str, ttl_seconds: int):
    try:
        return sb.snapshot_directory(path, ttl=ttl_seconds), True
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return sb.snapshot_directory(path), False


def _snapshot_filesystem(sb, ttl_seconds: int):
    try:
        return sb.snapshot_filesystem(ttl=ttl_seconds), True
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return sb.snapshot_filesystem(), False


@app.local_entrypoint()
def main(branches: int = 8, ttl_seconds: int = 60 * 60):
    if branches < 1:
        raise ValueError("branches must be positive")

    marker = f"retro-{uuid.uuid4().hex}"
    parent = _sandbox()
    directory_snapshot = None
    filesystem_snapshot = None
    result: dict = {"branches": branches, "ttl_seconds": ttl_seconds, "marker": marker}
    try:
        _check(
            parent,
            "mkdir -p /app && "
            "printf '#include <iostream>\\nint main(){std::cout << 1;}\\n' > /app/solution.cpp && "
            f"printf '%s' '{marker}' > /app/marker.txt",
        )

        started = time.perf_counter()
        directory_snapshot, directory_ttl_enforced = _snapshot_directory(parent, "/app", ttl_seconds)
        result["directory_snapshot_seconds"] = time.perf_counter() - started
        result["directory_snapshot_id"] = directory_snapshot.object_id
        result["directory_ttl_enforced_by_sdk"] = directory_ttl_enforced

        started = time.perf_counter()
        filesystem_snapshot, filesystem_ttl_enforced = _snapshot_filesystem(parent, ttl_seconds)
        result["filesystem_snapshot_seconds"] = time.perf_counter() - started
        result["filesystem_snapshot_id"] = filesystem_snapshot.object_id
        result["filesystem_ttl_enforced_by_sdk"] = filesystem_ttl_enforced
    finally:
        parent.terminate()

    direct = _sandbox()
    try:
        direct.mount_image("/direct", modal.Image.from_id(directory_snapshot.object_id))
        returncode, _, stderr = _exec(direct, "printf '\\nwrite-test\\n' >> /direct/marker.txt")
        result["directory_direct_write"] = {
            "supported": returncode == 0,
            "stderr": stderr[-300:],
        }
    finally:
        direct.terminate()

    fs_restore = _sandbox(source_image=modal.Image.from_id(filesystem_snapshot.object_id))
    fs_started = time.perf_counter()
    try:
        result["filesystem_restore_marker"] = _check(fs_restore, "cat /app/marker.txt").strip()
        _check(fs_restore, "printf '\\nfs-write\\n' >> /app/solution.cpp")
        result["filesystem_restore_seconds"] = time.perf_counter() - fs_started
    finally:
        fs_restore.terminate()

    with concurrent.futures.ThreadPoolExecutor(max_workers=branches) as pool:
        branch_results = list(pool.map(lambda branch: _branch(directory_snapshot.object_id, branch), range(branches)))
    result["branch_results"] = branch_results
    result["branch_restore_p50_seconds"] = statistics.median(row["restore_seconds"] for row in branch_results)
    result["branch_restore_max_seconds"] = max(row["restore_seconds"] for row in branch_results)
    result["isolated"] = (
        all(row["marker"] == marker for row in branch_results)
        and len({row["digest"] for row in branch_results}) == branches
    )

    for snapshot in (directory_snapshot, filesystem_snapshot):
        if snapshot is not None:
            modal.experimental.image_delete(snapshot.object_id)
    result["snapshots_deleted"] = True

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["filesystem_restore_marker"] != marker or not result["isolated"]:
        raise RuntimeError("retro snapshot smoke failed fidelity/isolation checks")
