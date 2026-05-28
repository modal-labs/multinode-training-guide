"""Helper functions for Modal multi-node training infrastructure."""

import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

# (attr_name_on_slime_cfg, cli_flag) — optional per-rank conversion args
_CONVERSION_EXTRA_ARGS = [
    ("decoder_first_pipeline_num_layers", "decoder-first-pipeline-num-layers"),
    ("decoder_last_pipeline_num_layers", "decoder-last-pipeline-num-layers"),
    ("mtp_num_layers", "mtp-num-layers"),
    ("make_vocab_size_divisible_by", "make-vocab-size-divisible-by"),
]


def get_checkpoint_conversion_policy(slime_cfg) -> tuple[int, int, list[str]]:
    """Return (num_nodes, nproc_per_node, extra_args) for checkpoint conversion."""
    gpus_per_node = getattr(slime_cfg, "actor_num_gpus_per_node", 8)
    actor_nodes = getattr(slime_cfg, "actor_num_nodes", 1)
    tp = getattr(slime_cfg, "tensor_model_parallel_size", 1)
    pp = getattr(slime_cfg, "pipeline_model_parallel_size", 1)

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
            if x := getattr(slime_cfg, attr, None):
                extra_args.append(f"--{flag} {x}")

        return num_nodes, nproc_per_node, extra_args

    raise ValueError(
        f"cannot find checkpoint conversion layout for world_size={world_size} "
        f"with actor_num_nodes={actor_nodes}, actor_num_gpus_per_node={gpus_per_node}"
    )


def get_modal_cluster_context(n_nodes: int) -> tuple[int, str, str, int]:
    """Return (rank, master_addr, my_ip, n_nodes) for the current Modal cluster."""
    import modal.experimental

    try:
        info = modal.experimental.get_cluster_info()
    except Exception:  # noqa: BLE001
        if n_nodes == 1:
            ip = _get_local_ip()
            return 0, ip, ip, 1
        raise
    actual_nodes = len(info.container_ipv4_ips)
    if actual_nodes == 0 and n_nodes == 1:
        ip = _get_local_ip()
        return 0, ip, ip, 1
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


def _get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def start_ray_head(my_ip: str, n_nodes: int, *, ray_port: int = 6379, include_dashboard: bool = True) -> None:
    """Start Ray head node and wait for all workers to join."""
    import ray

    start_timeout = int(os.getenv("RAY_START_TIMEOUT_SECONDS", "240"))
    worker_timeout = int(os.getenv("RAY_WORKER_JOIN_TIMEOUT_SECONDS", "180"))
    start_cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={my_ip}",
        f"--port={ray_port}",
        "--disable-usage-stats",
    ]
    if include_dashboard:
        start_cmd.append("--dashboard-host=0.0.0.0")
    else:
        start_cmd.append("--include-dashboard=false")
    try:
        subprocess.run(start_cmd, check=True, timeout=start_timeout)
    except subprocess.TimeoutExpired as exc:
        _print_ray_logs()
        raise RuntimeError(f"Ray head node did not finish startup within {start_timeout}s") from exc
    except subprocess.CalledProcessError as exc:
        _print_ray_logs()
        raise RuntimeError(f"Ray head node failed to start with exit code {exc.returncode}") from exc

    last_error = ""
    for _ in range(start_timeout):
        try:
            ray.init(address=f"{my_ip}:{ray_port}")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
    else:
        _print_ray_logs()
        raise RuntimeError(f"Ray head node failed to start before timeout: {last_error}")

    for _ in range(worker_timeout):
        alive = [n for n in ray.nodes() if n["Alive"]]
        print(f"Waiting for workers: {len(alive)}/{n_nodes} alive")
        if len(alive) == n_nodes:
            break
        time.sleep(1)
    else:
        _print_ray_logs()
        raise RuntimeError(f"Timed out waiting for all {n_nodes} Ray nodes to join")


def _print_ray_logs() -> None:
    log_dir = Path("/tmp/ray/session_latest/logs")
    for name in (
        "dashboard.log",
        "dashboard.err",
        "gcs_server.out",
        "gcs_server.err",
        "raylet.out",
        "raylet.err",
        "monitor.out",
        "monitor.err",
    ):
        path = log_dir / name
        if not path.exists():
            continue
        print(f"===== {path} =====")
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            print(f"could not read {path}: {exc}")
            continue
        for line in lines[-80:]:
            print(line)


def prepare_slime_config(slime_cfg, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    from huggingface_hub import snapshot_download
    import yaml

    from configs.base import YAML_CONFIG_FIELDS

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(slime_cfg, attr, None)) and not str(val).startswith("/"):
            setattr(slime_cfg, attr, snapshot_download(val, local_files_only=True))

    for field in YAML_CONFIG_FIELDS:
        if isinstance(val := getattr(slime_cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            print(f"Materialized {field} → {path}")
            setattr(slime_cfg, field, path)


def build_train_cmd(slime_cfg, slime_root: str) -> str:
    """Build the Ray job entrypoint, sourcing model arch args if needed."""
    train_script = (
        f"{slime_root}/{'train_async.py' if slime_cfg.async_mode else 'train.py'}"
    )
    if slime_cfg.slime_model_script:
        inner = (
            f"source {slime_root}/{slime_cfg.slime_model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(slime_cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(slime_cfg.cli_args())}"
