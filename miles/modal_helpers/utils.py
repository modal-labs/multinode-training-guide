"""Helper functions for Modal multi-node training infrastructure."""

import os
import shlex
import subprocess
import time
from os import PathLike

# (attr_name_on_miles_cfg, cli_flag) — optional per-rank conversion args
_CONVERSION_EXTRA_ARGS = [
    ("decoder_first_pipeline_num_layers", "decoder-first-pipeline-num-layers"),
    ("decoder_last_pipeline_num_layers", "decoder-last-pipeline-num-layers"),
    ("mtp_num_layers", "mtp-num-layers"),
    ("make_vocab_size_divisible_by", "make-vocab-size-divisible-by"),
]


def is_local_checkpoint_ref(ref: str | PathLike) -> bool:
    """Return True when a checkpoint ref is already a mounted local path."""
    return str(ref).startswith("/")


def resolve_checkpoint_ref(
    ref: str | PathLike,
    *,
    local_files_only: bool = True,
) -> str:
    """Resolve a checkpoint ref to a local path.

    Absolute paths are returned unchanged. Other values are treated as
    Hugging Face repo IDs and resolved through the local HF cache by default.
    """
    ref_str = str(ref)
    if is_local_checkpoint_ref(ref_str):
        return ref_str

    from huggingface_hub import snapshot_download

    return snapshot_download(ref_str, local_files_only=local_files_only)


def get_checkpoint_conversion_policy(miles_cfg) -> tuple[int, int, list[str]]:
    """Return (num_nodes, nproc_per_node, extra_args) for checkpoint conversion."""
    gpus_per_node = getattr(miles_cfg, "actor_num_gpus_per_node", 8)
    actor_nodes = getattr(miles_cfg, "actor_num_nodes", 1)
    tp = getattr(miles_cfg, "tensor_model_parallel_size", 1)
    pp = getattr(miles_cfg, "pipeline_model_parallel_size", 1)

    world_size = tp * pp if (tp > 1 or pp > 1) else gpus_per_node
    max_world_size = actor_nodes * gpus_per_node
    if world_size > max_world_size:
        raise ValueError(
            f"checkpoint conversion world_size={world_size} exceeds actor cluster capacity "
            f"{actor_nodes}x{gpus_per_node}={max_world_size}"
        )

    for num_nodes in range(1, actor_nodes + 1):
        if world_size % num_nodes != 0:
            continue
        nproc_per_node = world_size // num_nodes
        if nproc_per_node > gpus_per_node:
            continue

        extra_args: list[str] = []
        if tp > 1 or pp > 1:
            extra_args += [
                f"--tensor-model-parallel-size {tp}",
                f"--pipeline-model-parallel-size {pp}",
            ]
        for attr, flag in _CONVERSION_EXTRA_ARGS:
            if x := getattr(miles_cfg, attr, None):
                extra_args.append(f"--{flag} {x}")

        return num_nodes, nproc_per_node, extra_args

    raise ValueError(
        f"cannot find checkpoint conversion layout for world_size={world_size} "
        f"with actor_num_nodes={actor_nodes}, actor_num_gpus_per_node={gpus_per_node}"
    )


def get_modal_cluster_context(n_nodes: int) -> tuple[int, str, str, int]:
    """Return (rank, master_addr, my_ip, n_nodes) for the current Modal cluster."""
    if n_nodes == 1:
        return 0, "127.0.0.1", "127.0.0.1", 1

    import modal.experimental

    info = modal.experimental.get_cluster_info()
    actual_nodes = len(info.container_ipv4_ips)
    if actual_nodes != n_nodes:
        raise RuntimeError(
            f"cluster size mismatch: expected {n_nodes} node(s), got {actual_nodes}"
        )
    return (
        info.rank,
        info.container_ipv4_ips[0],
        info.container_ipv4_ips[info.rank],
        actual_nodes,
    )


def start_ray_head(my_ip: str, n_nodes: int) -> None:
    """Start Ray head node and wait for all workers to join."""
    import ray

    subprocess.Popen(
        [
            "ray",
            "start",
            "--head",
            f"--node-ip-address={my_ip}",
            "--dashboard-host=0.0.0.0",
        ]
    )
    for _ in range(30):
        try:
            ray.init(address="auto")
            break
        except ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("Ray head node failed to start")

    for _ in range(60):
        alive = [n for n in ray.nodes() if n["Alive"]]
        print(f"Waiting for workers: {len(alive)}/{n_nodes} alive")
        if len(alive) == n_nodes:
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Timed out waiting for all {n_nodes} Ray nodes to join")


def prepare_miles_config(miles_cfg, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    import yaml

    from configs.base import YAML_CONFIG_FIELDS

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if val := getattr(miles_cfg, attr, None):
            setattr(miles_cfg, attr, resolve_checkpoint_ref(val))

    for field in YAML_CONFIG_FIELDS:
        if isinstance(val := getattr(miles_cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            print(f"Materialized {field} → {path}")
            setattr(miles_cfg, field, path)


def build_train_cmd(miles_cfg, miles_root: str) -> str:
    """Build the Ray job entrypoint, sourcing model arch args if needed."""
    train_script = (
        f"{miles_root}/{'train_async.py' if miles_cfg.async_mode else 'train.py'}"
    )
    if miles_cfg.miles_model_script:
        inner = (
            f"source {miles_root}/{miles_cfg.miles_model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(miles_cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(miles_cfg.cli_args())}"
