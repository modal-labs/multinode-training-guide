from __future__ import annotations

# pyright: reportMissingImports=false, reportCallIssue=false, reportOptionalCall=false

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import modal
import modal.experimental


SKYRL_COMMIT = "830f005b454e9002233c3a7f96370a68d3602ca7"
SKYRL_ROOT = Path("/root/SkyRL")
EXAMPLE_ROOT = Path("/root/skyrl_tx_example")
MODEL_ID = "Qwen/Qwen3-8B"
HF_CACHE = "/root/.cache/huggingface"
CHECKPOINTS_DIR = "/checkpoints"
TINKER_PORT = 8000
COORDINATOR_PORT = 7777

N_NODES = int(os.environ.get("SKYRL_TX_N_NODES", "2"))
GPUS_PER_NODE = int(os.environ.get("SKYRL_TX_GPUS_PER_NODE", "4"))
GPU_TYPE = os.environ.get("SKYRL_TX_GPU_TYPE", "H100")

app = modal.App("example-skyrl-tx-qwen")

hf_cache_volume = modal.Volume.from_name("skyrl-tx-hf-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("skyrl-tx-checkpoints", create_if_missing=True)
run_state = modal.Dict.from_name("skyrl-tx-run-state", create_if_missing=True)

example_dir = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "git",
        "libnuma1",
        "numactl",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}",
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SKYRL_REPO_ROOT": str(SKYRL_ROOT),
            "TINKER_API_KEY": "tml-dummy",
            "UV_LINK_MODE": "copy",
            "UV_PROJECT_ENVIRONMENT": f"{SKYRL_ROOT}/.venv",
        }
    )
    .run_commands(
        f"git clone https://github.com/NovaSky-AI/SkyRL {SKYRL_ROOT}",
        f"cd {SKYRL_ROOT} && git checkout {SKYRL_COMMIT}",
    )
    .workdir(str(SKYRL_ROOT))
    .run_commands(
        "uv sync --extra tinker --extra gpu --extra jax",
        gpu="any",
    )
    .add_local_dir(str(example_dir), str(EXAMPLE_ROOT), copy=True)
    .entrypoint([])
)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub==1.2.3", "hf_transfer==0.1.9")
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _tail(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-max_lines:])


def _terminate_process_group(process: subprocess.Popen[bytes], timeout: int = 30) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=timeout)


def _wait_for_server(process: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + 45 * 60
    url = f"http://localhost:{TINKER_PORT}/healthz"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"SkyRL-TX API exited with {process.returncode}\n{_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {url}\n{_tail(log_path)}")


def _backend_config(master_addr: str, n_nodes: int, train_micro_batch_size: int) -> dict[str, object]:
    return {
        "max_lora_adapters": 2,
        "max_lora_rank": 8,
        "tensor_parallel_size": GPUS_PER_NODE,
        "fully_sharded_data_parallel_size": n_nodes,
        "train_micro_batch_size": train_micro_batch_size,
        "sample_max_num_sequences": 64,
        "gradient_checkpointing": True,
        "coordinator_address": f"{master_addr}:{COORDINATOR_PORT}",
        "num_processes": n_nodes,
    }


def _run_worker(master_addr: str, n_nodes: int, rank: int, run_key: str) -> None:
    log_path = Path(f"/tmp/skyrl_tx_worker_{rank}.log")
    cmd = [
        "uv",
        "run",
        "--extra",
        "gpu",
        "--extra",
        "tinker",
        "--extra",
        "jax",
        "-m",
        "skyrl.backends.jax",
        "--coordinator-address",
        f"{master_addr}:{COORDINATOR_PORT}",
        "--num-processes",
        str(n_nodes),
        "--process-id",
        str(rank),
    ]
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=SKYRL_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        while process.poll() is None:
            if run_state.get(run_key) == "done":
                _terminate_process_group(process)
                print(f"worker_rank={rank} stopped after coordinator completed", flush=True)
                print(_tail(log_path), flush=True)
                return
            time.sleep(5)
        if run_state.get(run_key) != "done":
            raise RuntimeError(f"worker_rank={rank} exited early\n{_tail(log_path)}")
    finally:
        if process.poll() is None:
            _terminate_process_group(process)


def _run_client(client_script: str, model_id: str, extra_args: list[str]) -> None:
    cmd = [
        "uv",
        "run",
        "--extra",
        "tinker",
        "--with",
        "numpy",
        "python",
        str(EXAMPLE_ROOT / client_script),
        "--base-url",
        f"http://localhost:{TINKER_PORT}",
        "--model-name",
        model_id,
        *extra_args,
    ]
    subprocess.run(cmd, cwd=SKYRL_ROOT, check=True)


def _run_tinker_job(
    mode: str,
    client_script: str,
    model_id: str,
    client_args: list[str],
    train_micro_batch_size: int,
) -> None:
    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else N_NODES
    master_addr = cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    run_key = f"{mode}:{model_id}:{master_addr}:{COORDINATOR_PORT}"

    if rank > 0:
        _run_worker(master_addr, n_nodes, rank, run_key)
        return

    run_state[run_key] = "running"
    log_path = Path(f"/tmp/skyrl_tx_{mode}_api.log")
    backend_config = _backend_config(master_addr, n_nodes, train_micro_batch_size)
    cmd = [
        "uv",
        "run",
        "--extra",
        "gpu",
        "--extra",
        "tinker",
        "--extra",
        "jax",
        "-m",
        "skyrl.tinker.api",
        "--base-model",
        model_id,
        "--backend",
        "jax",
        "--port",
        str(TINKER_PORT),
        "--checkpoints-base",
        str(Path(CHECKPOINTS_DIR) / mode),
        "--backend-config",
        json.dumps(backend_config),
    ]
    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=SKYRL_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _wait_for_server(process, log_path)
        _run_client(client_script, model_id, client_args)
    finally:
        run_state[run_key] = "done"
        if process is not None:
            _terminate_process_group(process)
        print(_tail(log_path), flush=True)


@app.function(
    image=download_image,
    volumes={HF_CACHE: hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=4 * 60 * 60,
)
def download_model(model_id: str = MODEL_ID, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(model_id, revision=revision)
    hf_cache_volume.commit()


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    volumes={HF_CACHE: hf_cache_volume, CHECKPOINTS_DIR: checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=24 * 60 * 60,
    memory=262144,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def run_sft(
    model_id: str = MODEL_ID,
    steps: int = 8,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
) -> None:
    _run_tinker_job(
        mode="sft",
        client_script="sft_client.py",
        model_id=model_id,
        client_args=[
            "--steps",
            str(steps),
            "--lora-rank",
            str(lora_rank),
            "--learning-rate",
            str(learning_rate),
        ],
        train_micro_batch_size=1,
    )


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    volumes={HF_CACHE: hf_cache_volume, CHECKPOINTS_DIR: checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=24 * 60 * 60,
    memory=262144,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def run_rl(
    model_id: str = MODEL_ID,
    steps: int = 4,
    samples_per_prompt: int = 2,
    lora_rank: int = 8,
    learning_rate: float = 5e-5,
) -> None:
    _run_tinker_job(
        mode="rl",
        client_script="rl_client.py",
        model_id=model_id,
        client_args=[
            "--steps",
            str(steps),
            "--samples-per-prompt",
            str(samples_per_prompt),
            "--lora-rank",
            str(lora_rank),
            "--learning-rate",
            str(learning_rate),
        ],
        train_micro_batch_size=max(1, samples_per_prompt),
    )
