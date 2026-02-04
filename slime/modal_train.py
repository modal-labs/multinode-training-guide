"""
Unified SLIME GRPO training script for Modal.

Usage:
    # Sync training with Qwen 0.5B (multi-node)
    modal run modal_train.py::train_multi_node --config qwen-0.5b-sync

    # Async training with Qwen 4B (multi-node)
    modal run modal_train.py::train_multi_node --config qwen-4b-async

    # Single node training
    modal run modal_train.py::train_single_node --config qwen-0.5b-sync

    # Prepare dataset
    modal run modal_train.py::prepare_dataset

    # List available configs
    modal run modal_train.py::list_available_configs

Available configs:
    - qwen-0.5b-sync
    - qwen-0.5b-async
    - qwen-4b-sync
    - qwen-4b-async

Models are automatically downloaded and cached via the huggingface-cache volume.
"""

import os
import subprocess
from pathlib import Path
import time

import modal
import modal.experimental

from configs.base import RLConfig

GPU_NAME = os.environ.get("GPU_NAME", "H200")
GPU_COUNT = int(os.environ.get("GPU_COUNT", "8"))
NUM_NODES = int(os.environ.get("NUM_NODES", "4"))


# =============================================================================
# Modal Image & Volumes
# =============================================================================

image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260202c")
    .entrypoint([])
    .add_local_python_source("configs")
)

with image.imports():
    import ray
    from ray.job_submission import JobSubmissionClient

# Paths
DATA_PATH: Path = Path("/data")
HF_CACHE_PATH: Path = Path("/root/.cache/huggingface")

# Volumes
data_volume: modal.Volume = modal.Volume.from_name(
    "grpo-slime-example-data", create_if_missing=True
)
hf_cache_volume: modal.Volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)

# Ray configuration
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
SINGLE_NODE_MASTER_ADDR = "127.0.0.1"

# =============================================================================
# App (created dynamically based on config)
# =============================================================================

# App name from environment variable (set before running modal)
APP_NAME = os.environ.get("APP_NAME", "slime-grpo")
app = modal.App(APP_NAME)


# =============================================================================
# Ray Initialization
# =============================================================================

def _init_ray(rank: int, main_node_addr: str, node_ip_addr: str, n_nodes: int):
    """Initialize Ray cluster across Modal containers.

    Rank 0 starts the head node, opens a tunnel to the Ray dashboard, and waits
    for all worker nodes to connect. Other ranks start as workers and connect
    to the head node address.
    """
    os.environ["SLIME_HOST_IP"] = node_ip_addr

    if rank == 0:
        print(f"Starting Ray head node at {node_ip_addr}")
        subprocess.Popen(
            [
                "ray",
                "start",
                "--head",
                f"--node-ip-address={node_ip_addr}",
                "--dashboard-host=0.0.0.0",
            ]
        )

        for _ in range(10):
            try:
                ray.init(address="auto")
            except ConnectionError:
                time.sleep(1)
                continue
            print("Connected to Ray")
            break
        else:
            raise Exception("Failed to connect to Ray")

        for _ in range(60):
            print("Waiting for worker nodes to connect...")
            alive_nodes = [n for n in ray.nodes() if n["Alive"]]
            print(f"Alive nodes: {len(alive_nodes)}/{n_nodes}")

            if len(alive_nodes) == n_nodes:
                print("All worker nodes connected")
                break
            time.sleep(1)
        else:
            raise Exception("Failed to connect to all worker nodes")
    else:
        print(f"Starting Ray worker node at {node_ip_addr}, connecting to {main_node_addr}")
        subprocess.Popen(
            [
                "ray",
                "start",
                f"--node-ip-address={node_ip_addr}",
                "--address",
                f"{main_node_addr}:{RAY_PORT}",
            ]
        )


# =============================================================================
# Training Command Generation
# =============================================================================

def generate_slime_cmd(
    config: RLConfig,
    master_addr: str,
) -> tuple[str, dict]:
    """Generate the slime training command and runtime environment."""
    import slime.utils.external_utils.command_utils as U
    from huggingface_hub import snapshot_download

    is_infinite_run = U.get_env_enable_infinite_run()

    # Download model to cache and get local path
    hf_token = os.environ.get("HF_TOKEN")
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    model_path = snapshot_download(config.model_id, token=hf_token)
    print(f"Model path: {model_path}")

    # Generate all training args from config
    train_args = config.generate_train_args(model_path, DATA_PATH, is_infinite_run)
    
    # Add wandb args
    train_args += f" {U.get_default_wandb_args(__file__, config.wandb_run_name_prefix)} --wandb-project {config.wandb_project}"

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CUDA_LAUNCH_BLOCKING": "1",  # Synchronous CUDA errors for debugging
            "NCCL_NVLS_ENABLE": "1",
            "no_proxy": master_addr,
            "MASTER_ADDR": master_addr,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    }
    
    return f"python3 {config.train_script} {train_args}", runtime_env


async def run_training(
    config: RLConfig,
    n_nodes: int,
    master_addr: str,
):
    """Submit SLIME training job to Ray cluster and stream logs."""
    client = JobSubmissionClient("http://127.0.0.1:8265")

    slime_cmd, runtime_env = generate_slime_cmd(config, master_addr)
    
    print("Submitting training job...")
    print(f"  Model: {config.model_name}")
    print(f"  Mode: {'sync' if config.sync else 'async'}")
    print(f"  Nodes: {n_nodes}")
    
    job_id = client.submit_job(
        entrypoint=slime_cmd,
        runtime_env=runtime_env
    )
    print(f"Job submitted with ID: {job_id}")

    async for line in client.tail_job_logs(job_id):
        print(line, end="", flush=True)


# =============================================================================
# Modal Functions
# =============================================================================

@app.function(
    image=image,
    volumes={
        DATA_PATH.as_posix(): data_volume,
        HF_CACHE_PATH.as_posix(): hf_cache_volume,
    },
    timeout=24 * 60 * 60,
)
def prepare_dataset():
    """Download and prepare the GSM8K dataset."""
    from datasets import load_dataset

    data_volume.reload()
    dataset = load_dataset("zhuzilin/gsm8k")
    dataset["train"].to_parquet(f"{DATA_PATH}/gsm8k/train.parquet")
    dataset["test"].to_parquet(f"{DATA_PATH}/gsm8k/test.parquet")
    data_volume.commit()
    print("Dataset prepared successfully")


@app.local_entrypoint()
def list_available_configs():
    """List all available training configs."""
    from configs import list_configs
    
    print("Available configs:")
    for name in list_configs():
        print(f"  - {name}")


# =============================================================================
# CLI Entry Points
# =============================================================================

@app.function(
    image=image,
    gpu=f"{GPU_NAME}:{GPU_COUNT}",
    volumes={
        HF_CACHE_PATH.as_posix(): hf_cache_volume,
        DATA_PATH.as_posix(): data_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=24 * 60 * 60,
    experimental_options={
        "efa_enabled": True,
    },
)
@modal.experimental.clustered(NUM_NODES, rdma=True)
async def train_multi_node(config: str = "qwen-0.5b-sync"):
    """Main entry point for multi-node GRPO training on Modal.
    
    Args:
        config: Config name (e.g., "qwen-0.5b-sync", "qwen-4b-async")
    """
    from configs import get_config
    
    cfg = get_config(config)

    hf_cache_volume.reload()
    data_volume.reload()

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


@app.function(
    image=image,
    gpu=f"{GPU_NAME}:{GPU_COUNT}",
    volumes={
        HF_CACHE_PATH.as_posix(): hf_cache_volume,
        DATA_PATH.as_posix(): data_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=24 * 60 * 60,
    experimental_options={
        "efa_enabled": True,
    },
)
async def train_single_node(config: str = "qwen-0.5b-sync"):
    """Single-node GRPO training on Modal.
    
    Args:
        config: Config name (e.g., "qwen-0.5b-sync", "qwen-4b-async"). File name with underscores replaced with dashes.
    """
    from configs import get_config
    
    cfg = get_config(config)

    hf_cache_volume.reload()
    data_volume.reload()

    _init_ray(0, SINGLE_NODE_MASTER_ADDR, SINGLE_NODE_MASTER_ADDR, 1)

    with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Dashboard URL: {tunnel.url}")
        print(f"Config: {config}")
        await run_training(cfg, 1, SINGLE_NODE_MASTER_ADDR)
