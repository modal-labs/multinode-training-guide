"""
Miles GRPO training on Modal — multi-node and single-node.

Run commands in the following order:
    modal run miles_modal_grpo.py::download_model
    modal run miles_modal_grpo.py::prepare_dataset
    modal run miles_modal_grpo.py
"""

import os
import subprocess
import time
from pathlib import Path

import modal
import modal.experimental

# Configuration
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_NAME = "qwen3"

N_NODES = 2
GPU_TYPE = "H100:8"
GPUS_PER_NODE = 8

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265

# Set up Modal Volumes

HF_CACHE_PATH = Path("/hf-cache")
CHECKPOINTS_PATH = Path("/checkpoints")
DATA_PATH = Path("/data")

hf_cache_volume = modal.Volume.from_name("miles-hf-cache", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("miles-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("miles-data", create_if_missing=True)

# Set up the container image

image = (
    modal.Image.from_registry("radixark/miles:latest")
    .pip_install("huggingface_hub[cli]", "datasets", "pandas", "ray[default]>=2.38")
    .entrypoint([])
)

app = modal.App(
    name=os.environ.get("MILES_APP_NAME", "miles-grpo"),
    image=image,
)

# Set up the shared volume + secret config

VOLUMES = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
    str(DATA_PATH): data_volume,
}

SECRETS = [modal.Secret.from_dict({"HF_HOME": str(HF_CACHE_PATH)})]
if os.environ.get("WANDB_API_KEY"):
    SECRETS.append(modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}))

# Set up the Ray cluster

def _init_ray(rank: int, head_ip: str, my_ip: str, n_nodes: int):
    # Had to debug this for a while...
    # MILES_HOST_IP tells Miles what IP to advertise for this node. It must be
    # set before Ray starts so the raylet (and every actor it spawns) inherits
    # the correct routable address. Without this, get_host_info() falls back to
    # a UDP probe that returns 127.0.0.1 inside Modal containers, which causes
    # the SGLang router to bind to localhost and be unreachable from other nodes.

    os.environ["MILES_HOST_IP"] = my_ip

    # On Modal retries, the same container is reused, so we may have stale Ray
    # state from a previous attempt. Shut it down cleanly.
    try:
        import ray as _ray
        if _ray.is_initialized():
            _ray.shutdown()
    except Exception:
        pass
    subprocess.run(["ray", "stop", "--force"], check=False)
    time.sleep(3)

    if rank == 0:
        subprocess.run(
            [
                "ray", "start", "--head",
                f"--node-ip-address={my_ip}",
                f"--port={RAY_PORT}",
                "--dashboard-host=0.0.0.0",
                f"--dashboard-port={RAY_DASHBOARD_PORT}",
            ],
            check=True,
        )
        print(f"[Node 0] Ray head started at {my_ip}:{RAY_PORT}")
        time.sleep(15)

        import ray as _ray
        _ray.init(address=f"{my_ip}:{RAY_PORT}", ignore_reinit_error=True)

        for _ in range(120):
            alive = [n for n in _ray.nodes() if n["Alive"]]
            print(f"[Node 0] Ray cluster: {len(alive)}/{n_nodes} nodes alive")
            if len(alive) >= n_nodes:
                break
            time.sleep(5)
        else:
            raise RuntimeError("Timed out waiting for all Ray nodes to join")

    else:
        time.sleep(8)
        subprocess.run(
            [
                "ray", "start",
                f"--address={head_ip}:{RAY_PORT}",
                f"--node-ip-address={my_ip}",
            ],
            check=True,
        )
        print(f"[Node {rank}] Ray worker joined {head_ip}:{RAY_PORT}")


# Build the training command

def _build_train_cmd(n_nodes: int, head_ip: str) -> list[str]:
    from huggingface_hub import snapshot_download
    hf_model_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=str(HF_CACHE_PATH),
        local_files_only=True,
    )
    print(f"[Node 0] Model path: {hf_model_path}")

    # In colocate mode, all GPUs alternate between inference and training.
    # rollout-num-gpus controls how many SGLang GPUs are used during its turn.
    total_gpus = n_nodes * GPUS_PER_NODE
    rollout_gpus = total_gpus // 2

    return [
        "python", "/root/miles/train.py",
        # Algorithm: GRPO with policy gradient loss
        "--advantage-estimator", "grpo",
        "--loss-type", "policy_loss",
        # Model
        "--model-name", MODEL_NAME,
        "--hf-checkpoint", hf_model_path,
        "--ref-load", hf_model_path,
        # Training backend: FSDP 
        "--train-backend", "fsdp",
        # GPU layout: colocate shares all GPUs between SGLang and FSDP
        "--actor-num-nodes", str(n_nodes),
        "--actor-num-gpus-per-node", str(GPUS_PER_NODE),
        "--num-gpus-per-node", str(GPUS_PER_NODE),
        "--colocate",
        "--rollout-num-gpus", str(rollout_gpus),
        "--rollout-num-gpus-per-engine", "2",
        "--sglang-mem-fraction-static", "0.5",
        # Rollout hyperparameters
        "--rollout-batch-size", "64",
        "--n-samples-per-prompt", "4",
        "--rollout-temperature", "0.7",
        "--rollout-max-prompt-len", "512",
        "--rollout-max-response-len", "1024",
        # Training hyperparameters
        "--lr", "1e-6",
        "--kl-coef", "0.01",
        "--micro-batch-size", "4",
        "--global-batch-size", "32",
        "--num-epoch", "1",
        "--eps-clip", "0.2",
        # Reward: rule-based math checker
        "--rm-type", "deepscaler",
        # Data
        "--prompt-data", f"{DATA_PATH}/gsm8k/train.parquet",
        "--input-key", "prompt",
        "--label-key", "label",
        # Checkpointing
        "--save-interval", "25",
        "--save", f"{CHECKPOINTS_PATH}/{MODEL_NAME}-grpo",
    ]


async def run_training(n_nodes: int, head_ip: str):
    cmd = _build_train_cmd(n_nodes, head_ip)
    print(f"[Node 0] Launching: {' '.join(cmd)}")
    # Pass MILES_HOST_IP so Miles starts the SGLang router bound to the head node's routable IP.
    env = {**os.environ, "MILES_HOST_IP": head_ip}
    result = subprocess.run(cmd, cwd="/root/miles", env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Miles training failed (exit {result.returncode})")
    print("[Node 0] Training complete!")
    checkpoints_volume.commit()


# Set up the Modal functions

@app.function(
    gpu=GPU_TYPE,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
)
@modal.experimental.clustered(N_NODES, rdma=True)
async def train_multi_node():
    await hf_cache_volume.reload.aio()
    await data_volume.reload.aio()

    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank
    ips = cluster_info.container_ipv4_ips
    n_nodes = len(ips)

    print(f"[Node {rank}] IPv4: {ips[rank]} | Leader: {ips[0]} | Total nodes: {n_nodes}")

    _init_ray(rank, ips[0], ips[rank], n_nodes)

    if rank == 0:
        async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
            print(f"[Node 0] Ray Dashboard: {tunnel.url}")
            await run_training(n_nodes, ips[0])
    else:
        while True:
            time.sleep(30)


@app.function(
    gpu=GPU_TYPE,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * 60 * 60,
)
async def train_single_node():
    await hf_cache_volume.reload.aio()
    await data_volume.reload.aio()

    localhost = "127.0.0.1"
    _init_ray(0, localhost, localhost, 1)

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Ray Dashboard: {tunnel.url}")
        await run_training(1, localhost)


# Utils

@app.function(
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    secrets=[modal.Secret.from_dict({"HF_HOME": str(HF_CACHE_PATH)})],
    timeout=60 * 60,
)
def download_model():
    from huggingface_hub import snapshot_download
    print(f"Downloading {MODEL_ID} ...")
    path = snapshot_download(repo_id=MODEL_ID, cache_dir=str(HF_CACHE_PATH))
    print(f"Downloaded to: {path}")
    hf_cache_volume.commit()


@app.function(
    image=modal.Image.debian_slim().pip_install("datasets", "pandas"),
    volumes={str(DATA_PATH): data_volume},
    timeout=30 * 60,
)
def prepare_dataset():
    """Download GSM8K and save train/test splits as parquet."""
    import pandas as pd
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main")
    for split in ("train", "test"):
        records = [
            {"prompt": row["question"], "label": row["answer"]}
            for row in dataset[split]
        ]
        os.makedirs(f"{DATA_PATH}/gsm8k", exist_ok=True)
        pd.DataFrame(records).to_parquet(f"{DATA_PATH}/gsm8k/{split}.parquet")
        print(f"Saved {len(records)} {split} records")

    data_volume.commit()


@app.local_entrypoint()
def main():
    if N_NODES > 1:
        print(f"Launching {N_NODES} nodes × {GPUS_PER_NODE} H100s = {N_NODES * GPUS_PER_NODE} GPUs")
        train_multi_node.remote()
    else:
        print(f"Launching single-node: {GPUS_PER_NODE} H100s")
        train_single_node.remote()

