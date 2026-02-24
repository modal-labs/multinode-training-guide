from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import modal
import modal.experimental
import requests

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from configs.base import RLConfig

EXAMPLE_REMOTE_ROOT = "/root/code_golf_harbor_modal"

image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260126a")
    .run_commands(
        "uv pip install --system git+https://github.com/huggingface/transformers.git@eebf856",
        "uv pip install --system harbor==0.1.44 pandas pyarrow huggingface_hub requests",
        r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/qwen/qwen3_bridge.py""",
    )
    .entrypoint([])
    .add_local_dir(
        str(CURRENT_DIR),
        remote_path=EXAMPLE_REMOTE_ROOT,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
    )
)

with image.imports():
    import ray
    from ray.job_submission import JobSubmissionClient

DATA_PATH = Path("/data")
HF_CACHE_PATH = Path("/root/.cache/huggingface")
CHECKPOINTS_PATH = Path("/checkpoints")

DATA_VOLUME_NAME = "slime-code-golf-harbor-data"
CHECKPOINT_VOLUME_NAME = "slime-code-golf-harbor-checkpoints"

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265

APP_NAME = os.environ.get("SLIME_APP_NAME", "slime-code-golf")
app = modal.App(APP_NAME)


def _init_ray(rank: int, main_node_addr: str, node_ip_addr: str, n_nodes: int) -> None:
    os.environ["SLIME_HOST_IP"] = node_ip_addr

    if rank == 0:
        subprocess.Popen(
            [
                "ray",
                "start",
                "--head",
                f"--node-ip-address={node_ip_addr}",
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
            raise RuntimeError("Unable to initialize ray head node.")

        for _ in range(120):
            alive_nodes = [node for node in ray.nodes() if node["Alive"]]
            if len(alive_nodes) == n_nodes:
                return
            time.sleep(1)
        raise RuntimeError("Not all ray worker nodes joined in time.")

    subprocess.Popen(
        [
            "ray",
            "start",
            f"--node-ip-address={node_ip_addr}",
            "--address",
            f"{main_node_addr}:{RAY_PORT}",
        ]
    )


def _build_runtime_env(config: RLConfig, master_addr: str) -> dict:
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    extra_paths = ["/root/Megatron-LM/", EXAMPLE_REMOTE_ROOT]
    pythonpath = ":".join(extra_paths + ([existing_pythonpath] if existing_pythonpath else []))

    return {
        "env_vars": {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "no_proxy": master_addr,
            "MASTER_ADDR": master_addr,
            "PYTHONPATH": pythonpath,
            "HARBOR_TASKS_ROOT": config.harbor_task_root,
            "HARBOR_DATA_VOLUME_NAME": config.harbor_data_volume_name,
            "HARBOR_RM_MODAL_APP": config.harbor_rm_modal_app,
            "HARBOR_RM_MAX_CONCURRENCY": str(config.harbor_rm_max_concurrency),
            "HARBOR_RM_TIMEOUT_SEC": str(config.harbor_rm_timeout_sec),
            "HARBOR_LENGTH_BONUS_WEIGHT": str(config.harbor_length_bonus_weight),
        }
    }


def _generate_slime_cmd(config: RLConfig, master_addr: str) -> tuple[str, dict]:
    from huggingface_hub import snapshot_download

    hf_model_path = snapshot_download(repo_id=config.model_id, local_files_only=True)
    train_args = config.generate_train_args(
        hf_model_path=hf_model_path,
        checkpoints_path=CHECKPOINTS_PATH,
        data_path=DATA_PATH,
    )

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        train_args += (
            f" --use-wandb --wandb-project {config.wandb_project}"
            f" --wandb-group {config.wandb_run_name_prefix}"
            f" --wandb-key '{wandb_key}' --disable-wandb-random-suffix"
        )

    train_script = config.train_script
    return f"python3 {train_script} {train_args}", _build_runtime_env(config, master_addr)


async def _run_training(config: RLConfig, n_nodes: int, master_addr: str) -> None:
    client = JobSubmissionClient("http://127.0.0.1:8265")
    slime_cmd, runtime_env = _generate_slime_cmd(config, master_addr)
    job_id = client.submit_job(entrypoint=slime_cmd, runtime_env=runtime_env)
    print(f"Submitted SLIME training job: {job_id}")
    async for line in client.tail_job_logs(job_id):
        print(line, end="", flush=True)


def _find_latest_checkpoint(search_root: Path) -> Path:
    candidates: list[Path] = []
    for config_path in search_root.rglob("config.json"):
        parent = config_path.parent
        if any(parent.glob("*.safetensors")) or any(parent.glob("*.bin")):
            candidates.append(parent)

    if not candidates:
        for candidate in search_root.rglob("*"):
            if candidate.is_dir() and candidate.name.startswith(("global_step", "step", "iter")):
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {search_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _wait_for_openai_server(base_url: str, timeout_sec: int = 300) -> None:
    models_url = f"{base_url.rstrip('/')}/v1/models"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            response = requests.get(models_url, timeout=2)
            if response.ok:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for server readiness: {models_url}")


def _run_harbor_eval_subprocess(
    server_base_url: str,
    model_name: str,
    n_concurrent: int,
    n_tasks: Optional[int],
    tasks_dir: Path,
    jobs_dir: Path,
) -> None:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "harbor",
        "run",
        "-p",
        str(tasks_dir),
        "--agent-import-path",
        "harbor_litellm_agent.SingleShotCodeAgent",
        "-m",
        model_name,
        "--ak",
        f"api_base={server_base_url.rstrip('/')}/v1",
        "--ak",
        "temperature=0.0",
        "--ak",
        "max_tokens=1024",
        "-e",
        "modal",
        "-n",
        str(n_concurrent),
        "--jobs-dir",
        str(jobs_dir),
    ]
    if n_tasks is not None:
        cmd.extend(["-l", str(n_tasks)])

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{EXAMPLE_REMOTE_ROOT}:{existing}" if existing else EXAMPLE_REMOTE_ROOT

    print("Running Harbor eval command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


@app.function(
    image=image,
    volumes={HF_CACHE_PATH.as_posix(): hf_cache_volume},
    timeout=24 * 60 * 60,
)
def download_model(config: str = "qwen-8b-multi", revision: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    from configs import get_config

    cfg = get_config(config)
    path = snapshot_download(repo_id=cfg.model_id, revision=revision)
    print(f"Model downloaded to: {path}")
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={DATA_PATH.as_posix(): data_volume},
    timeout=24 * 60 * 60,
)
def prepare_dataset(train_size: int = 900, limit: Optional[int] = None) -> None:
    from dataset_to_harbor import convert_mbpp_to_harbor_and_slime

    data_volume.reload()
    output_root = DATA_PATH / "mbpp_harbor"
    summary = convert_mbpp_to_harbor_and_slime(
        output_root=output_root,
        train_size=train_size,
        limit=limit,
    )
    data_volume.commit()
    print(summary)


@app.local_entrypoint()
def list_available_configs() -> None:
    from configs import list_configs

    print("Available configs:")
    for config_name in list_configs():
        print(f"  - {config_name}")


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
@modal.experimental.clustered(4, rdma=True)
async def train_multi_node(config: str = "qwen-8b-multi") -> None:
    from configs import get_config

    cfg = get_config(config)

    hf_cache_volume.reload()
    data_volume.reload()

    cluster_info = modal.experimental.get_cluster_info()
    ray_main_node_addr = cluster_info.container_ipv4_ips[0]
    my_ip_addr = cluster_info.container_ipv4_ips[cluster_info.rank]
    n_nodes = len(cluster_info.container_ipv4_ips)

    _init_ray(cluster_info.rank, ray_main_node_addr, my_ip_addr, n_nodes)

    if cluster_info.rank == 0:
        with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
            print(f"Ray dashboard: {tunnel.url}")
            await _run_training(cfg, n_nodes, ray_main_node_addr)
    else:
        while True:
            time.sleep(30)


@app.function(
    image=image,
    gpu="H100:1",
    volumes={CHECKPOINTS_PATH.as_posix(): checkpoints_volume},
    timeout=24 * 60 * 60,
)
@modal.web_server(8000, startup_timeout=30 * 60)
def serve_latest_checkpoint(
    checkpoint_subdir: str = "qwen8b_code_golf",
    tensor_parallel_size: int = 1,
) -> None:
    checkpoints_volume.reload()
    search_root = CHECKPOINTS_PATH / checkpoint_subdir
    checkpoint_path = _find_latest_checkpoint(search_root)
    print(f"Serving checkpoint: {checkpoint_path}")
    subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(checkpoint_path),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--tp",
            str(tensor_parallel_size),
        ]
    )


@app.function(
    image=image,
    volumes={DATA_PATH.as_posix(): data_volume},
    timeout=24 * 60 * 60,
)
def run_harbor_eval(
    server_base_url: str,
    model_name: str = "qwen-code-golf",
    n_concurrent: int = 128,
    n_tasks: Optional[int] = 200,
) -> None:
    data_volume.reload()
    tasks_dir = DATA_PATH / "mbpp_harbor" / "tasks"
    jobs_dir = DATA_PATH / "mbpp_harbor" / "harbor_jobs"
    _run_harbor_eval_subprocess(
        server_base_url=server_base_url,
        model_name=model_name,
        n_concurrent=n_concurrent,
        n_tasks=n_tasks,
        tasks_dir=tasks_dir,
        jobs_dir=jobs_dir,
    )
    data_volume.commit()


@app.function(
    image=image,
    gpu="H100:1",
    volumes={
        CHECKPOINTS_PATH.as_posix(): checkpoints_volume,
        DATA_PATH.as_posix(): data_volume,
    },
    timeout=24 * 60 * 60,
)
def eval_latest_checkpoint(
    checkpoint_subdir: str = "qwen8b_code_golf",
    model_name: str = "qwen-code-golf",
    n_concurrent: int = 128,
    n_tasks: Optional[int] = 200,
) -> None:
    checkpoints_volume.reload()
    data_volume.reload()

    checkpoint_path = _find_latest_checkpoint(CHECKPOINTS_PATH / checkpoint_subdir)
    print(f"Using checkpoint for eval: {checkpoint_path}")

    server_proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(checkpoint_path),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--tp",
            "1",
        ]
    )
    try:
        _wait_for_openai_server("http://127.0.0.1:8000", timeout_sec=300)
        _run_harbor_eval_subprocess(
            server_base_url="http://127.0.0.1:8000",
            model_name=model_name,
            n_concurrent=n_concurrent,
            n_tasks=n_tasks,
            tasks_dir=DATA_PATH / "mbpp_harbor" / "tasks",
            jobs_dir=DATA_PATH / "mbpp_harbor" / "harbor_jobs",
        )
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=15)

    data_volume.commit()
