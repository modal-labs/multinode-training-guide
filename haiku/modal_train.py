"""
Unified SLIME GRPO training script for Modal.

Usage:
    # Sync training with Qwen 0.5B (multi-node)
    modal run modal_train.py::train_multi_node --config qwen-4b

    # Async training with Qwen 4B (multi-node)
    modal run modal_train.py::train_multi_node --config qwen-4b

    # Single node training
    modal run modal_train.py::train_single_node --config qwen-4b

    # Single node training with LoRA (using local slime repo)
    USE_LOCAL_SLIME=/path/to/slime modal run modal_train.py::train_single_node --config qwen-4b-lora

    # Download model
    modal run modal_train.py::download_model --config qwen-4b

    # Prepare dataset
    modal run modal_train.py::prepare_dataset

    # List available configs
    modal run modal_train.py::list_available_configs

Environment variables:
    USE_LOCAL_SLIME=/path     Path to local slime repo for development
    SLIME_APP_NAME=...        Override Modal app name

Available configs (main):
    - qwen-4b, glm-4-7, glm-4-7-flash, qwen3-30B-A3B-Instruct-2507
"""

import os
import subprocess
from pathlib import Path
from typing import Optional
import time

import modal
import modal.experimental

from configs.base import RLConfig


# =============================================================================
# Modal Image & Volumes
# =============================================================================

# Path to local slime repo for development (e.g., USE_LOCAL_SLIME=/path/to/slime)
# Set to a directory path to overlay local slime code, or leave unset to use registry image
LOCAL_SLIME_PATH = os.environ.get("USE_LOCAL_SLIME", "")

image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260126a")
    .run_commands(
        "uv pip install --system git+https://github.com/huggingface/transformers.git@eebf856",  # 4.54.1
        "uv pip install --system aiohttp",  # For LLM judge reward model
        """sed -i 's/AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)/AutoImageProcessor.register(config, slow_image_processor_class=image_processor, exist_ok=True)/g' /sgl-workspace/sglang/python/sglang/srt/configs/utils.py""",
        # Fix rope_theta access for transformers 5.x (moved to rope_parameters dict)
        r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/glm/glm45_bridge.py""",
        r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/qwen/qwen3_bridge.py""",
    )
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
    .add_local_dir("tools", remote_path="/root/tools", copy=True)
)

# Overlay local slime code for development
# Install slime to /opt/slime-dev (not /root/slime) to avoid sys.path conflicts when Ray runs scripts
SLIME_DEV_PATH = "/opt/slime-dev"
if LOCAL_SLIME_PATH:
    # Copy the entire slime repo (has pyproject.toml) and install it
    image = image.add_local_dir(LOCAL_SLIME_PATH, remote_path=SLIME_DEV_PATH, copy=True, ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/modal"]).run_commands(f"uv pip install --system -e {SLIME_DEV_PATH}")
else:
    SLIME_DEV_PATH = None

with image.imports():
    import ray
    from ray.job_submission import JobSubmissionClient

# Paths
DATA_PATH: Path = Path("/data")
MODELS_PATH: Path = Path("/models")

# Volumes
data_volume: modal.Volume = modal.Volume.from_name("grpo-slime-haiku-data", create_if_missing=True)
checkpoints_volume: modal.Volume = modal.Volume.from_name("grpo-slime-haiku-checkpoints", create_if_missing=True)

# Ray configuration
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
SINGLE_NODE_MASTER_ADDR = "127.0.0.1"

# =============================================================================
# App (created dynamically based on config)
# =============================================================================

# App name from environment variable (set before running modal)
# Usage: SLIME_APP_NAME="my-experiment" modal run modal_train.py ...
APP_NAME = os.environ.get("SLIME_APP_NAME", "slime-haiku-grpo")
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

        for _ in range(30):
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
    experiment_name: str,
) -> tuple[str, dict]:
    """Generate the slime training command and runtime environment."""
    import datetime
    import random

    # Check for infinite run mode
    is_infinite_run = os.environ.get("SLIME_TEST_ENABLE_INFINITE_RUN", "0").lower() in ("true", "1")

    # Generate all training args from config
    train_args = config.generate_train_args(MODELS_PATH, DATA_PATH, is_infinite_run)

    checkpoint_dir = MODELS_PATH / experiment_name
    train_args += f" --save {checkpoint_dir} --save-interval {config.save_steps if hasattr(config, 'save_steps') else 100}"

    # Add wandb args if API key is available
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        run_id = datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S") + f"-{random.randint(0, 999):03d}"
        wandb_run_name = f"{config.wandb_run_name_prefix}_{run_id}" if config.wandb_run_name_prefix else run_id
        train_args += f" --use-wandb --wandb-project {config.wandb_project} --wandb-group {wandb_run_name} --wandb-key '{wandb_key}' --disable-wandb-random-suffix"

    # Build PYTHONPATH by appending to existing (don't clobber)
    import os as _os
    existing_pythonpath = _os.environ.get("PYTHONPATH", "")
    megatron_path = "/root/Megatron-LM/"
    pythonpath = f"{megatron_path}:{existing_pythonpath}" if existing_pythonpath else megatron_path

    runtime_env = {
        "env_vars": {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "no_proxy": master_addr,
            "MASTER_ADDR": master_addr,
            # Megatron-LM requires PYTHONPATH (pip install doesn't work due to package name mismatch)
            # slime is pip installed so doesn't need to be on PYTHONPATH
            "PYTHONPATH": pythonpath,
        }
    }

    # Use full path when local slime is installed
    # Note: config.train_script returns "slime/train.py" for base image,
    # but local repo has train.py at root level
    # Check at runtime if dev path exists (USE_LOCAL_SLIME is only set during image build)
    train_script = config.train_script
    dev_path = "/opt/slime-dev"
    if os.path.exists(dev_path):
        script_name = "train.py" if config.sync else "train_async.py"
        train_script = f"{dev_path}/{script_name}"

    return f"python3 {train_script} {train_args}", runtime_env


async def run_training(
    config: RLConfig,
    n_nodes: int,
    master_addr: str,
    experiment_name: str, 
):
    """Submit SLIME training job to Ray cluster and stream logs."""
    client = JobSubmissionClient("http://127.0.0.1:8265")

    slime_cmd, runtime_env = generate_slime_cmd(config, master_addr, experiment_name)

    print("Submitting training job...")
    print(f"  Model: {config.model_name}")
    print(f"  Mode: {'sync' if config.sync else 'async'}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Checkpoint dir: {MODELS_PATH / experiment_name}")

    job_id = client.submit_job(entrypoint=slime_cmd, runtime_env=runtime_env)
    print(f"Job submitted with ID: {job_id}")

    async for line in client.tail_job_logs(job_id):
        print(line, end="", flush=True)

    await checkpoints_volume.commit.aio()
    print("Checkpoints saved and committed to volume")

        


# =============================================================================
# Modal Functions
# =============================================================================


@app.function(
    image=image,
    volumes={MODELS_PATH.as_posix(): checkpoints_volume},
    timeout=24 * 60 * 60,
)
def download_model(
    config: str = "qwen-0.5b",
    revision: Optional[str] = None,
):
    """Download model from HuggingFace.

    Args:
        config: Config name (e.g., "qwen-0.5b", "qwen-4b")
        revision: Optional HF revision to pin
    """
    from huggingface_hub import snapshot_download
    from configs import get_config

    cfg = get_config(config)

    snapshot_download(
        repo_id=cfg.model_id,
        local_dir=MODELS_PATH / cfg.model_name,
        revision=revision,
    )
    print(f"Model downloaded to {MODELS_PATH / cfg.model_name}")

    checkpoints_volume.commit()


@app.function(
    image=image,
    volumes={DATA_PATH.as_posix(): data_volume},
    timeout=24 * 60 * 60,
)
def prepare_dataset():
    """Download and prepare the Haiku dataset."""
    from datasets import load_dataset
    import pandas as pd

    data_volume.reload()
    
    ds = load_dataset("statworx/haiku")
    
    def transform_example(example):
        question = f"Write me a haiku about {example['keywords'].lower()}."
        answer = example["text"]
        return {
            "question": question,
            "label": answer,
            "messages": [
                {
                    "content": "You are a helpful assistant. /no_think",
                    "role": "system"
                },
                {
                    "content": question,
                    "role": "user"
                }
            ]
        }
    
    # this dataset only has "train", but no "test", so we manually split out the last 20% of the train dataset as test
    # and remove them from the train dataset
    test_size = min(1000, int(len(ds["train"]) * 0.2))
    test_ds = ds["train"].select(range(len(ds["train"]) - test_size, len(ds["train"])))
    ds["train"] = ds["train"].select(range(test_size))
    ds["test"] = test_ds
    
    train_transformed = ds["train"].map(transform_example, remove_columns=["keywords"])
    test_transformed = ds["test"].map(transform_example, remove_columns=["keywords"])
    
    # Save as parquet
    train_transformed.to_parquet(f"{DATA_PATH}/haiku/train.parquet")
    test_transformed.to_parquet(f"{DATA_PATH}/haiku/test.parquet")
    
    data_volume.commit()
    print("Haiku dataset prepared successfully")
    print(f"Train examples: {len(train_transformed)}")
    print(f"Test examples: {len(test_transformed)}")
    print("\nExample:")
    print(f"Prompt: {train_transformed[0]['question']}")
    print(f"Text: {train_transformed[0]['label']}")


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
    gpu="H200:8",
    volumes={
        MODELS_PATH.as_posix(): checkpoints_volume,
        DATA_PATH.as_posix(): data_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
    timeout=24 * 60 * 60,
    experimental_options={
        "efa_enabled": True,
    },
)
async def train_single_node(config: str, run_name: str):
    """Single-node GRPO training on Modal.

    Args:
        config: Config name (e.g., "qwen-0.5b-sync", "qwen-4b-async"). File name with underscores replaced with dashes.
    """
    from configs import get_config
    from datetime import datetime

    cfg = get_config(config,run_name=run_name)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_short = cfg.model_name.split("/")[-1]
    experiment_name = f"{model_short}-singlenode-{timestamp}"

    await checkpoints_volume.reload.aio()
    await data_volume.reload.aio()

    _init_ray(0, SINGLE_NODE_MASTER_ADDR, SINGLE_NODE_MASTER_ADDR, 1)

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Dashboard URL: {tunnel.url}")
        print(f"Config: {config}")
        print(f"Experiment: {experiment_name}")
        await run_training(cfg, 1, SINGLE_NODE_MASTER_ADDR, experiment_name)