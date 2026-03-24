"""Unified Miles + Harbor training script for Modal."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import modal
import modal.experimental

from configs.base import USACO_GIT_COMMIT, USACO_GIT_URL


here = Path(__file__).parent
MILES_GIT_COMMIT = "6e9151cc4fc02dfbf3b2271e5cd070c3e9c8ac55"
MILES_SRC_PATH = Path("/root/miles-src")
MILES_OVERRIDES_PATH = Path("/root/miles_overrides")

image = (
    modal.Image.from_registry("radixark/miles:latest")
    .run_commands(
        "python -m ensurepip || true",
        "python -m pip install --upgrade pip",
        "uv pip install --system git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps",
        "uv pip install --system 'nvidia-modelopt[torch]>=0.37.0' --no-build-isolation",
        "uv pip install --system git+https://github.com/yushengsu-thu/Megatron-Bridge.git@merged-megatron-0.16.0rc0-miles --no-deps --no-build-isolation",
        f"git clone https://github.com/radixark/miles.git {MILES_SRC_PATH}",
        f"cd {MILES_SRC_PATH} && git checkout {MILES_GIT_COMMIT}",
        f"cd {MILES_SRC_PATH} && uv pip install --system -e .",
        "uv pip install --system git+https://github.com/BerriAI/litellm.git git+https://github.com/laude-institute/harbor.git",
    )
    .entrypoint([])
    .add_local_dir(here / "configs", remote_path="/root/configs", copy=True)
    .add_local_file(here / "generate.py", "/root/generate.py", copy=True)
    .add_local_file(here / "harbor_agent.py", "/root/harbor_agent.py", copy=True)
    .add_local_file(here / "harbor_agent_function.py", "/root/harbor_agent_function.py", copy=True)
    .add_local_dir(here / "tasks", remote_path="/root/harbor_tasks", copy=True)
    .add_local_dir(here / "overrides", remote_path="/root/miles_overrides", copy=True)
)

with image.imports():
    import ray
    from harbor.models.task.id import GitTaskId
    from harbor.tasks.client import TaskClient
    from huggingface_hub import snapshot_download
    from ray.job_submission import JobSubmissionClient


DATA_PATH = Path("/data")
HF_CACHE_PATH = Path("/root/.cache/huggingface")
CHECKPOINTS_PATH = Path("/checkpoints")

data_volume = modal.Volume.from_name("miles-harbor-example-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("miles-harbor-checkpoints", create_if_missing=True)

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
RAY_CLIENT_SERVER_PORT = 10001
RAY_METRICS_EXPORT_PORT = 20000
RAY_MIN_WORKER_PORT = 20001
RAY_MAX_WORKER_PORT = 29999
SINGLE_NODE_MASTER_ADDR = "127.0.0.1"
MULTI_NODE_COUNT = 2

APP_NAME = os.environ.get("MILES_APP_NAME", "miles-harbor")
app = modal.App(APP_NAME)


def _init_ray(rank: int, main_node_addr: str, node_ip_addr: str, n_nodes: int):
    os.environ["MILES_HOST_IP"] = node_ip_addr

    if rank == 0:
        print(f"Starting Ray head node at {node_ip_addr}")
        subprocess.Popen(
            [
                "ray",
                "start",
                "--head",
                f"--node-ip-address={node_ip_addr}",
                "--dashboard-host=0.0.0.0",
                f"--dashboard-port={RAY_DASHBOARD_PORT}",
                f"--port={RAY_PORT}",
                f"--ray-client-server-port={RAY_CLIENT_SERVER_PORT}",
                f"--metrics-export-port={RAY_METRICS_EXPORT_PORT}",
                f"--min-worker-port={RAY_MIN_WORKER_PORT}",
                f"--max-worker-port={RAY_MAX_WORKER_PORT}",
                "--disable-usage-stats",
            ]
        )

        for _ in range(60):
            try:
                ray.init(address="auto")
                break
            except ConnectionError:
                time.sleep(1)
        else:
            raise RuntimeError("Failed to connect to Ray head node")

        for _ in range(120):
            alive_nodes = [node for node in ray.nodes() if node["Alive"]]
            print(f"Alive nodes: {len(alive_nodes)}/{n_nodes}")
            if len(alive_nodes) == n_nodes:
                return
            time.sleep(1)
        raise RuntimeError("Timed out waiting for Ray workers")

    print(f"Starting Ray worker node at {node_ip_addr}, connecting to {main_node_addr}")
    subprocess.Popen(
        [
            "ray",
            "start",
            f"--node-ip-address={node_ip_addr}",
            "--address",
            f"{main_node_addr}:{RAY_PORT}",
            f"--metrics-export-port={RAY_METRICS_EXPORT_PORT}",
            f"--min-worker-port={RAY_MIN_WORKER_PORT}",
            f"--max-worker-port={RAY_MAX_WORKER_PORT}",
            "--disable-usage-stats",
        ]
    )


def _resolve_train_script(script_name: str) -> str:
    candidates = [
        f"{MILES_SRC_PATH}/{script_name}",
        f"/root/{script_name}",
        f"/root/miles/{script_name}",
        f"/workspace/miles/{script_name}",
        f"/workspace/{script_name}",
        f"/opt/miles/{script_name}",
        script_name,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    try:
        discovered = subprocess.check_output(
            ["sh", "-lc", f"find / -maxdepth 5 -name {script_name} 2>/dev/null | head -n 1"],
            text=True,
        ).strip()
        if discovered:
            return discovered
    except Exception:
        pass
    return script_name


def _resolve_convert_script() -> str:
    candidates = [
        f"{MILES_SRC_PATH}/tools/convert_hf_to_torch_dist.py",
        "/root/miles/tools/convert_hf_to_torch_dist.py",
        "/root/tools/convert_hf_to_torch_dist.py",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Could not locate convert_hf_to_torch_dist.py in the Miles image")


def _ensure_bootstrap_checkpoint(cfg, hf_model_path: str):
    bootstrap_path = cfg.bootstrap_checkpoint_path(CHECKPOINTS_PATH)
    tracker_path = bootstrap_path / "latest_checkpointed_iteration.txt"
    if tracker_path.exists():
        print(f"Using existing Megatron checkpoint at {bootstrap_path}")
        return

    if bootstrap_path.exists():
        shutil.rmtree(bootstrap_path)
    bootstrap_path.mkdir(parents=True, exist_ok=True)
    convert_script = _resolve_convert_script()
    cmd = [
        "python3",
        convert_script,
        "--hf-checkpoint",
        hf_model_path,
        "--save",
        bootstrap_path.as_posix(),
        *cfg._clean_args(cfg.model_args).split(),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"/root/miles_overrides:{MILES_SRC_PATH}:/root:/root/Megatron-LM" + (
        f":{existing_pythonpath}" if existing_pythonpath else ""
    )
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = env.get("MASTER_PORT", "12355")
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["DEPRECATED_MEGATRON_COMPATIBLE"] = "1"
    print(f"Converting {cfg.model_id} into Megatron checkpoint at {bootstrap_path}")
    subprocess.run(cmd, check=True, env=env)
    checkpoints_volume.commit()


def _install_runtime_overrides():
    overrides = [
        (
            MILES_OVERRIDES_PATH / "miles" / "router" / "session" / "sessions.py",
            MILES_SRC_PATH / "miles" / "router" / "session" / "sessions.py",
        )
    ]
    for source, target in overrides:
        if not source.exists():
            raise FileNotFoundError(f"Missing override source: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def generate_miles_cmd(config, master_addr: str) -> tuple[str, dict]:
    hf_model_path = snapshot_download(repo_id=config.model_id, local_files_only=True)
    train_args = config.generate_train_args(hf_model_path, DATA_PATH, CHECKPOINTS_PATH)
    train_script = _resolve_train_script(config.train_script)

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        train_args += (
            f" --use-wandb --wandb-project {config.wandb_project}"
            f" --wandb-group {config.wandb_run_name_prefix or config.model_name.lower()}"
            f" --wandb-key '{wandb_key}'"
        )

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = f"/root/miles_overrides:{MILES_SRC_PATH}:/root:/root/Megatron-LM"
    if existing_pythonpath:
        pythonpath = f"{pythonpath}:{existing_pythonpath}"

    runtime_env = {
        "env_vars": {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "DEPRECATED_MEGATRON_COMPATIBLE": "1",
            "MASTER_ADDR": master_addr,
            "PYTHONPATH": pythonpath,
            "NCCL_NVLS_ENABLE": "1",
            "AGENT_MODEL_NAME": "model",
            "MODAL_ENVIRONMENT": os.environ.get("MODAL_ENVIRONMENT", ""),
            "MODAL_TOKEN_ID": os.environ.get("MODAL_TOKEN_ID", ""),
            "MODAL_TOKEN_SECRET": os.environ.get("MODAL_TOKEN_SECRET", ""),
            "HF_HOME": HF_CACHE_PATH.as_posix(),
            "HARBOR_USE_LOCAL_TASKS": "1",
            "no_proxy": master_addr,
        }
    }

    return f"python3 {train_script} {train_args}", runtime_env


@app.function(
    image=image,
    timeout=60 * 60,
)
def inspect_miles_runtime():
    script = _resolve_train_script("train_async.py")
    print(f"Miles source: {MILES_SRC_PATH}")
    print(f"Resolved train_async.py: {script}")
    print(subprocess.check_output(["git", "-C", MILES_SRC_PATH.as_posix(), "rev-parse", "HEAD"], text=True).strip())
    cmd = [
        "python3",
        "-c",
        (
            "import inspect; "
            "from miles.utils.arguments import parse_args_train_backend; "
            "from miles.rollout.generate_hub.agentic_tool_call import generate; "
            "print('train_backend', parse_args_train_backend()); "
            "print('generate_has_add_arguments', callable(getattr(generate, 'add_arguments', None)))"
        ),
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        print(exc.output)
        raise
    print(output.strip())


async def run_training(config, n_nodes: int, master_addr: str):
    _install_runtime_overrides()
    client = JobSubmissionClient("http://127.0.0.1:8265")
    miles_cmd, runtime_env = generate_miles_cmd(config, master_addr)

    print(f"Submitting Miles job for {config.model_name}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Harbor task mode: {config.harbor_task_mode}")
    print(f"  Async: {not config.sync}")

    job_id = client.submit_job(entrypoint=miles_cmd, runtime_env=runtime_env)
    print(f"Job submitted with ID: {job_id}")

    async for line in client.tail_job_logs(job_id):
        print(line, end="", flush=True)


def _write_jsonl(records: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def _prepare_hello_dataset(cfg):
    task_path = "/root/harbor_tasks/hello-world"
    records = [
        {
            "prompt": 'Create a file called hello.txt with "Hello, world!" as the content.',
            "metadata": {
                "harbor_task_path": task_path,
                "harbor_task_name": f"hello-world-{idx}",
                "harbor_task_mode": "hello",
            },
        }
        for idx in range(cfg.harbor_task_limit)
    ]
    output_path = cfg.dataset_path(DATA_PATH)
    _write_jsonl(records, output_path)
    print(f"Wrote Harbor hello-world dataset to {output_path}")


def _prepare_usaco_dataset(cfg):
    task_root = DATA_PATH / "harbor" / "usaco" / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)

    task_ids = [
        GitTaskId(
            git_url=USACO_GIT_URL,
            git_commit_id=USACO_GIT_COMMIT,
            path=Path(f"datasets/usaco/{task_id}"),
        )
        for task_id in cfg.harbor_task_ids[: cfg.harbor_task_limit]
    ]
    downloaded = TaskClient().download_tasks(task_ids, overwrite=True, output_dir=task_root)

    records = []
    for path in downloaded:
        instruction = (path / "instruction.md").read_text()
        records.append(
            {
                "prompt": instruction,
                "metadata": {
                    "harbor_task_path": path.as_posix(),
                    "harbor_task_name": path.name,
                    "harbor_task_mode": "usaco",
                },
            }
        )

    output_path = cfg.dataset_path(DATA_PATH)
    _write_jsonl(records, output_path)
    print(f"Wrote Harbor USACO dataset to {output_path}")


@app.function(
    image=image,
    gpu="H100:1",
    volumes={
        HF_CACHE_PATH.as_posix(): hf_cache_volume,
        CHECKPOINTS_PATH.as_posix(): checkpoints_volume,
    },
    timeout=24 * 60 * 60,
)
def download_model(config: str = "hello-qwen-0-6b", revision: Optional[str] = None):
    from configs import get_config

    cfg = get_config(config)
    checkpoints_volume.reload()
    path = snapshot_download(repo_id=cfg.model_id, revision=revision)
    _ensure_bootstrap_checkpoint(cfg, path)
    print(f"Model downloaded to {path}")
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={DATA_PATH.as_posix(): data_volume},
    timeout=24 * 60 * 60,
)
def prepare_dataset(config: str = "hello-qwen-0-6b"):
    from configs import get_config

    cfg = get_config(config)
    data_volume.reload()

    if cfg.harbor_task_mode == "usaco":
        _prepare_usaco_dataset(cfg)
    else:
        _prepare_hello_dataset(cfg)

    data_volume.commit()


@app.local_entrypoint()
def list_available_configs():
    from configs import list_configs

    print("Available configs:")
    for name in list_configs():
        print(f"  - {name}")


@app.function(
    image=image,
    gpu="H100:8",
    volumes={
        HF_CACHE_PATH.as_posix(): hf_cache_volume,
        CHECKPOINTS_PATH.as_posix(): checkpoints_volume,
        DATA_PATH.as_posix(): data_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * 60,
)
async def train_single_node(config: str = "hello-qwen-0-6b", sync: bool = False):
    from configs import get_config

    cfg = get_config(config, sync)

    await hf_cache_volume.reload.aio()
    await data_volume.reload.aio()
    _init_ray(0, SINGLE_NODE_MASTER_ADDR, SINGLE_NODE_MASTER_ADDR, 1)

    with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Dashboard URL: {tunnel.url}")
        print(f"Config: {config}")
        await run_training(cfg, 1, SINGLE_NODE_MASTER_ADDR)


@app.function(
    image=image,
    gpu="H100:8",
    volumes={
        HF_CACHE_PATH.as_posix(): hf_cache_volume,
        CHECKPOINTS_PATH.as_posix(): checkpoints_volume,
        DATA_PATH.as_posix(): data_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(MULTI_NODE_COUNT, rdma=True)
async def train_multi_node(config: str = "usaco-qwen-0-6b", sync: bool = False):
    from configs import get_config

    cfg = get_config(config, sync)
    if cfg.n_nodes != MULTI_NODE_COUNT:
        raise ValueError(f"Config {config} expects {cfg.n_nodes} nodes, but train_multi_node is fixed to {MULTI_NODE_COUNT}.")

    await hf_cache_volume.reload.aio()
    await data_volume.reload.aio()

    cluster_info = modal.experimental.get_cluster_info()
    print(f"Rank: {cluster_info.rank}, task id: {os.environ['MODAL_TASK_ID']}")
    print(f"Config: {config}")
    print(f"Container IPv4 IPs: {cluster_info.container_ipv4_ips}")

    ray_main_node_addr = cluster_info.container_ipv4_ips[0]
    my_ip_addr = cluster_info.container_ipv4_ips[cluster_info.rank]
    n_nodes = len(cluster_info.container_ipv4_ips)

    _init_ray(cluster_info.rank, ray_main_node_addr, my_ip_addr, n_nodes)

    if cluster_info.rank == 0:
        with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
            print(f"Dashboard URL: {tunnel.url}")
            await run_training(cfg, n_nodes, ray_main_node_addr)
    else:
        while True:
            time.sleep(10)
