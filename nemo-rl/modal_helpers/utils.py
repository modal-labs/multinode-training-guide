"""
General helpers for Ray multinode 
"""
import json
import shlex
import subprocess
import time

# Glob covering both lib/ and lib64/ python site-packages in the NeMo-RL venv.
_NSIGHT_GLOB = (
    "/opt/nemo_rl_venv/lib*/python*/site-packages/ray/_private/runtime_env/nsight.py"
)

# Keep Ray worker ports below the OS ephemeral range (see NeMo-RL ray.sub).
_MIN_WORKER_PORT = 10002
_MAX_WORKER_PORT = 11000


def _ray_node_resources(gpus_per_node: int, node_rank: int | None = None) -> str:
    """Custom Ray resources registered by NeMo-RL's ray.sub on every node."""
    resources: dict[str, int] = {
        "worker_units": gpus_per_node,
        "slurm_managed_ray_cluster": 1,
    }
    if node_rank is not None:
        resources[f"modal_node_{node_rank}"] = 1
    return json.dumps(resources, separators=(",", ":"))


def cluster_driver_env(head_ip: str, cluster_ips: list[str] | None = None) -> dict[str, str]:
    """Environment for the NeMo-RL driver and Ray actors on a multi-node cluster."""
    no_proxy_hosts = ",".join(dict.fromkeys(["127.0.0.1", head_ip, *(cluster_ips or [])]))
    return {
        # NeMo-RL uses `uv run` itself; disable Ray's per-task uv runtime env (ray.sub).
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "TRAIN_ENABLE_SHARE_CUDA_VISIBLE_DEVICES": "0",
        "MASTER_ADDR": head_ip,
        "no_proxy": no_proxy_hosts,
        "NCCL_NVLS_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }


def get_modal_cluster_context(n_nodes: int) -> tuple[int, str, str, int, list[str]]:
    """Return (rank, head_ip, my_ip, n_nodes, cluster_ips) for the current Modal cluster."""
    if n_nodes == 1:
        return 0, "127.0.0.1", "127.0.0.1", 1, ["127.0.0.1"]

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
        list(info.container_ipv4_ips),
    )

# TODO: probably unnecessary and can be removed
def _patch_nsight() -> None:
    """Mirror ray.sub's nsight patch so Ray honors NeMo-RL's py_executable.

    NeMo-RL launches workers via ``uv run``; Ray's nsight runtime-env plugin
    otherwise hardcodes ``python`` and breaks profiling. Best-effort.
    """
    import glob

    sed = (
        r's/context\.py_executable = " "\.join(self\.nsight_cmd) + " python"/'
        r'context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"/g'
    )
    for path in glob.glob(_NSIGHT_GLOB):
        subprocess.run(["sed", "-i", sed, path], check=False)


def _wait_for_ray_cluster(
    ray, n_nodes: int, gpus_per_node: int, timeout_s: int = 240
) -> None:
    """Block until each Ray node has registered its GPUs (not just cluster-wide total)."""
    expected_gpus = n_nodes * gpus_per_node
    for _ in range(timeout_s // 2):
        alive = [n for n in ray.nodes() if n["Alive"]]
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        per_node = sorted(
            (
                n.get("NodeManagerAddress"),
                int(n.get("Resources", {}).get("GPU", 0)),
            )
            for n in alive
        )
        ready_nodes = sum(1 for _, g in per_node if g >= gpus_per_node)
        print(
            f"Waiting for cluster: {len(alive)}/{n_nodes} nodes, "
            f"{total_gpus}/{expected_gpus} GPUs, "
            f"{ready_nodes}/{n_nodes} nodes with >={gpus_per_node} GPU(s)"
        )
        print(f"  per-node: {per_node}")
        if (
            len(alive) >= n_nodes
            and total_gpus >= expected_gpus
            and ready_nodes >= n_nodes
        ):
            return
        time.sleep(2)
    raise RuntimeError(
        f"Timed out waiting for {n_nodes} nodes each with {gpus_per_node} GPUs "
        f"(cluster total {expected_gpus})"
    )

# Nemo-RL launches training through a SLURM bash script (https://github.com/NVIDIA-NeMo/RL/blob/main/ray.sub) getting nodes from a list SLURM_JOBS_NODELIST
# instead we break this up into start_ray_head and start_ray_worker
def start_ray_head(
    head_ip: str, port: int, n_nodes: int, gpus_per_node: int, node_rank: int = 0
) -> None:
    """Start the Ray head and block until every node and GPU has joined."""
    import ray

    _patch_nsight()
    subprocess.Popen(
        [
            "ray",
            "start",
            "--head",
            "--disable-usage-stats",
            f"--num-gpus={gpus_per_node}",
            f"--resources={_ray_node_resources(gpus_per_node, node_rank)}",
            f"--node-ip-address={head_ip}",
            f"--port={port}",
            "--dashboard-host=0.0.0.0",
            "--include-dashboard=True",
            "--block",
        ]
    )

    for _ in range(60):
        try:
            ray.init(address="auto")
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("Ray head node failed to start")

    _wait_for_ray_cluster(ray, n_nodes, gpus_per_node)
    # Detach the driver-side ray handle; the NeMo-RL driver reconnects itself.
    ray.shutdown()


def start_ray_worker(
    head_ip: str, port: int, my_ip: str, gpus_per_node: int, node_rank: int
) -> None:
    """Start a Ray worker that joins the head and blocks forever."""
    _patch_nsight()
    subprocess.Popen(
        [
            "ray",
            "start",
            f"--node-ip-address={my_ip}",
            "--address",
            f"{head_ip}:{port}",
            "--disable-usage-stats",
            f"--num-gpus={gpus_per_node}",
            f"--resources={_ray_node_resources(gpus_per_node, node_rank)}",
            f"--min-worker-port={_MIN_WORKER_PORT}",
            f"--max-worker-port={_MAX_WORKER_PORT}",
            "--block",
        ]
    )



def build_train_cmd(nemo_rl_cfg, nemo_rl_root: str, experiment: str | None = None) -> str:
    """Build the driver command run on the Ray head node."""
    import importlib.util

    args = shlex.join(nemo_rl_cfg.cli_args(experiment))
    if nemo_rl_cfg.num_nodes > 1:
        
        spec = importlib.util.find_spec("modal_helpers.run_grpo_multinode")
        if spec is None or spec.origin is None:
            raise RuntimeError("modal_helpers.run_grpo_multinode not found in image")
        entrypoint = spec.origin
    else:
        entrypoint = nemo_rl_cfg.entrypoint
    return f"cd {shlex.quote(nemo_rl_root)} && uv run python {shlex.quote(entrypoint)} {args}"
