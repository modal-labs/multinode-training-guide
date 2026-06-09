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
GPUS_PER_NODE = 8
GPU_TYPE = "H100"

app = modal.App("example-skyrl-tx-qwen")

hf_cache_volume = modal.Volume.from_name("skyrl-tx-hf-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "skyrl-tx-checkpoints", create_if_missing=True
)

example_dir = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "build-essential",
        "ca-certificates",
        "git",
        "libnuma1",
        "numactl",
    )
    .pip_install("uv==0.11.19")
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


def _terminate_process_group(
    process: subprocess.Popen[bytes], timeout: int = 30
) -> None:
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
    url = f"http://localhost:{TINKER_PORT}/api/v1/healthz"
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


def _wait_for_coordinator_launch(run_state: modal.Dict, run_key: str) -> None:
    deadline = time.monotonic() + 10 * 60
    while time.monotonic() < deadline:
        status = run_state.get(run_key)
        if status == "running":
            return
        if status == "done":
            raise RuntimeError("Coordinator completed before worker startup")
        time.sleep(2)
    raise TimeoutError("Timed out waiting for coordinator startup")


def _patch_skyrl_checkpointing() -> None:
    jax_backend_path = SKYRL_ROOT / "skyrl" / "backends" / "jax.py"
    source = jax_backend_path.read_text()
    if "primary_host=None" in source:
        return

    source = source.replace(
        "from flax.training import checkpoints\n",
        "import orbax.checkpoint as ocp\nfrom flax.training import checkpoints\n",
    )
    source = source.replace(
        """    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        \"\"\"Save training checkpoint using Flax checkpoints.\"\"\"
        checkpoint_data = self._extract_checkpoint_data(model_id)
        checkpoints.save_checkpoint_multiprocess(
            target=checkpoint_data,
            ckpt_dir=output_path,
            step=0,
            prefix="checkpoint_",
            overwrite=True,
        )
        logger.info(f"Saved training checkpoint to {output_path}")
""",
        """    def save_checkpoint(self, output_path: AnyPath, model_id: str) -> None:
        \"\"\"Save training checkpoint using Flax checkpoints.\"\"\"
        checkpoint_data = self._extract_checkpoint_data(model_id)
        checkpointer = ocp.Checkpointer(
            ocp.PyTreeCheckpointHandler(
                multiprocessing_options=ocp.options.MultiprocessingOptions(primary_host=None)
            )
        )
        checkpoints.save_checkpoint_multiprocess(
            target=checkpoint_data,
            ckpt_dir=output_path,
            step=0,
            prefix="checkpoint_",
            overwrite=True,
            orbax_checkpointer=checkpointer,
        )
        logger.info(f"Saved training checkpoint to {output_path}")
""",
    )
    source = source.replace(
        """    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        \"\"\"Load training checkpoint using Flax checkpoints.\"\"\"
        checkpoint = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=self._extract_checkpoint_data(model_id),
            prefix="checkpoint_",
        )
""",
        """    def load_checkpoint(self, checkpoint_path: AnyPath, model_id: str) -> None:
        \"\"\"Load training checkpoint using Flax checkpoints.\"\"\"
        checkpointer = ocp.Checkpointer(
            ocp.PyTreeCheckpointHandler(
                multiprocessing_options=ocp.options.MultiprocessingOptions(primary_host=None)
            )
        )
        checkpoint = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=self._extract_checkpoint_data(model_id),
            prefix="checkpoint_",
            orbax_checkpointer=checkpointer,
        )
""",
    )
    if source.count("orbax_checkpointer=checkpointer") != 2:
        raise RuntimeError("Failed to patch SkyRL JAX checkpointing")
    jax_backend_path.write_text(source)


def _backend_config(
    master_addr: str,
    n_nodes: int,
    gpus_per_node: int,
    train_micro_batch_size: int,
    lora_rank: int,
) -> dict[str, object]:
    return {
        "max_lora_adapters": 2,
        "max_lora_rank": max(1, lora_rank),
        "tensor_parallel_size": gpus_per_node,
        "fully_sharded_data_parallel_size": n_nodes,
        "train_micro_batch_size": train_micro_batch_size,
        "sample_max_num_sequences": 64,
        "gradient_checkpointing": True,
        "coordinator_address": f"{master_addr}:{COORDINATOR_PORT}",
        "num_processes": n_nodes,
    }


def _run_worker(
    run_state: modal.Dict,
    master_addr: str,
    n_nodes: int,
    rank: int,
    run_key: str,
) -> None:
    _patch_skyrl_checkpointing()
    _wait_for_coordinator_launch(run_state, run_key)
    log_path = Path(f"/tmp/skyrl_tx_worker_{rank}.log")
    cmd = [
        "uv",
        "run",
        "--no-sync",
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
                checkpoint_volume.commit()
                print(
                    f"worker_rank={rank} stopped after coordinator completed",
                    flush=True,
                )
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
        "--no-sync",
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


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _commit_and_print_checkpoints(mode: str) -> None:
    checkpoint_root = Path(CHECKPOINTS_DIR) / mode
    checkpoint_files = sorted(checkpoint_root.rglob("*.tar.gz"))
    if not checkpoint_files:
        raise RuntimeError(
            f"No {mode} checkpoints were written under {checkpoint_root}"
        )
    for checkpoint_file in checkpoint_files:
        relative_path = checkpoint_file.relative_to(checkpoint_root)
        print(
            f"{mode}_checkpoint_file={relative_path} bytes={_path_size(checkpoint_file)}",
            flush=True,
        )
    checkpoint_volume.commit()
    print(f"{mode}_checkpoint_volume_committed={len(checkpoint_files)}", flush=True)


def _run_tinker_job(
    run_state: modal.Dict,
    mode: str,
    client_script: str,
    model_id: str,
    client_args: list[str],
    gpus_per_node: int,
    train_micro_batch_size: int,
    lora_rank: int,
) -> None:
    _patch_skyrl_checkpointing()
    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank
    if not cluster_info.container_ipv4_ips:
        raise RuntimeError(
            "Clustered run did not receive container IPv4 addresses from Modal"
        )
    n_nodes = len(cluster_info.container_ipv4_ips)
    master_addr = cluster_info.container_ipv4_ips[0]
    run_key = f"{cluster_info.cluster_id}:{mode}:{model_id}:{COORDINATOR_PORT}"
    print(
        f"{mode}_cluster rank={rank} master_addr={master_addr} "
        f"n_nodes={n_nodes} gpu={GPU_TYPE}:{gpus_per_node}",
        flush=True,
    )

    if rank > 0:
        _run_worker(run_state, master_addr, n_nodes, rank, run_key)
        return

    log_path = Path(f"/tmp/skyrl_tx_{mode}_api.log")
    backend_config = _backend_config(
        master_addr,
        n_nodes,
        gpus_per_node,
        train_micro_batch_size,
        lora_rank,
    )
    cmd = [
        "uv",
        "run",
        "--no-sync",
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
        run_state[run_key] = "running"
        _wait_for_server(process, log_path)
        _run_client(client_script, model_id, client_args)
        _commit_and_print_checkpoints(mode)
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
def run_sft_cluster(
    run_state: modal.Dict,
    model_id: str = MODEL_ID,
    steps: int = 8,
    lora_rank: int = 8,
    learning_rate: float = 1e-6,
    gpus_per_node: int = GPUS_PER_NODE,
) -> None:
    _run_tinker_job(
        run_state=run_state,
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
        gpus_per_node=gpus_per_node,
        train_micro_batch_size=1,
        lora_rank=lora_rank,
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
def run_rl_cluster(
    run_state: modal.Dict,
    model_id: str = MODEL_ID,
    steps: int = 4,
    samples_per_prompt: int = 2,
    lora_rank: int = 8,
    learning_rate: float = 1e-6,
    gpus_per_node: int = GPUS_PER_NODE,
) -> None:
    _run_tinker_job(
        run_state=run_state,
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
        gpus_per_node=gpus_per_node,
        train_micro_batch_size=max(1, samples_per_prompt),
        lora_rank=lora_rank,
    )


@app.local_entrypoint()
def run_sft(
    model_id: str = MODEL_ID,
    steps: int = 8,
    lora_rank: int = 8,
    learning_rate: float = 1e-6,
) -> None:
    with modal.Dict.ephemeral() as run_state:
        run_sft_cluster.remote(
            run_state, model_id, steps, lora_rank, learning_rate, GPUS_PER_NODE
        )


@app.local_entrypoint()
def run_rl(
    model_id: str = MODEL_ID,
    steps: int = 4,
    samples_per_prompt: int = 2,
    lora_rank: int = 8,
    learning_rate: float = 1e-6,
) -> None:
    with modal.Dict.ephemeral() as run_state:
        run_rl_cluster.remote(
            run_state,
            model_id,
            steps,
            samples_per_prompt,
            lora_rank,
            learning_rate,
            GPUS_PER_NODE,
        )
