"""
SLIME GRPO Haiku training script for Modal.

Usage:
    # deploy
    modal deploy llm_judges.deploy

    # Train model
    modal run modal_train.py::download_model
    modal run modal_train.py::prepare_dataset
    modal run modal_train.py::train_single_node --run-name my-experiment

    # With local slime repo for development:
    USE_LOCAL_SLIME=/path/to/slime modal run modal_train.py::train_single_node

Environment variables:
    USE_LOCAL_SLIME=/path     Path to local slime repo for development
"""

import os
import subprocess
from pathlib import Path
from typing import Optional
import time

from llm_judges.base import MODAL_VOCABS
import modal

from config import RLConfig, get_config, ACTIVE_JUDGE_TYPE, ACTIVE_JUDGE_MODEL_SIZE


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
    .add_local_python_source("config", copy=True)
    .add_local_dir("tools", remote_path="/root/tools", copy=True)
    .add_local_dir("llm_judges", remote_path="/root/llm_judges", copy=True)
    .pip_install("nltk>=3.8.0")
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
HF_CACHE_PATH = "/root/.cache/huggingface"
DATA_PATH: Path = Path(f"{HF_CACHE_PATH}/processed")
CHECKPOINTS_PATH: Path = Path("/checkpoints")

# Volumes
hf_cache_vol: modal.Volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_volume: modal.Volume = modal.Volume.from_name("slime-haiku-checkpoints", create_if_missing=True)

# Ray configuration
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
SINGLE_NODE_MASTER_ADDR = "127.0.0.1"

app = modal.App(f"train-haiku-judge_{ACTIVE_JUDGE_TYPE.value}_{ACTIVE_JUDGE_MODEL_SIZE.shorthand}")


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

    train_args = config.generate_train_args(DATA_PATH)

    checkpoint_dir = CHECKPOINTS_PATH / experiment_name
    train_args += f" --save {checkpoint_dir} --save-interval {config.save_steps if hasattr(config, 'save_steps') else 10}"

    # Add wandb args if API key is available
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d-%H%M%S") + f"-{random.randint(0, 999):03d}"
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
    dev_path = "/opt/slime-dev"
    if os.path.exists(dev_path):
        train_script = f"{dev_path}/train.py"
    else:
        train_script = "slime/train.py"

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
    print(f"  Nodes: {n_nodes}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Checkpoint dir: {CHECKPOINTS_PATH / experiment_name}")

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
    volumes={HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=24 * 60 * 60,
)
def download_model(
    revision: Optional[str] = None,
):
    """Download model from HuggingFace."""
    from huggingface_hub import snapshot_download

    cfg = get_config()

    path = snapshot_download(
        repo_id=cfg.model_id,
        revision=revision,
    )
    print(f"Model downloaded to {path}")

    hf_cache_vol.commit()




@app.function(
    image=image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=24 * 60 * 60,
)
def prepare_dataset():
    """Download and prepare the Haiku dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    cfg = get_config()

    hf_cache_vol.reload()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    
    ds = load_dataset("statworx/haiku")
    
    def format_chat_template(example, tokenizer):
        system_prompt = f"You are a haiku poet. You will be given a prompt and you will need to write a haiku about the prompt. Try to incorporate these words into the haiku if possible: {', '.join(MODAL_VOCABS)}"

        keyword = example['keywords'].lower()
        question = f"Write me a haiku about {keyword}."

        messages = [
            {"content": system_prompt, "role": "system"},
            {"content": question, "role": "user"},
        ]

        return {
            "question": question,
            "label": example["text"],
            "messages": messages,
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False),
        }
    
    # this dataset only has "train", but no "test", so we manually split out the last 20% of the train dataset as test
    # and remove them from the train dataset
    test_size = min(1000, int(len(ds["train"]) * 0.2))
    test_ds = ds["train"].select(range(len(ds["train"]) - test_size, len(ds["train"])))
    ds["train"] = ds["train"].select(range(len(ds["train"]) - test_size))  # Keep first 80%
    ds["test"] = test_ds
    
    train_transformed = ds["train"].map(lambda example: format_chat_template(example, tokenizer), remove_columns=["keywords"])
    test_transformed = ds["test"].map(lambda example: format_chat_template(example, tokenizer), remove_columns=["keywords"])
    
    # Save as parquet
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    (DATA_PATH / "haiku").mkdir(parents=True, exist_ok=True)
    train_transformed.to_parquet(f"{DATA_PATH}/haiku/train.parquet")
    test_transformed.to_parquet(f"{DATA_PATH}/haiku/test.parquet")
    
    hf_cache_vol.commit()
    print("Haiku dataset prepared successfully")
    print(f"Train examples: {len(train_transformed)}")
    print(f"Test examples: {len(test_transformed)}")
    print("\nExample:")
    print(f"Prompt: {train_transformed[0]['question']}")
    print(f"Text: {train_transformed[0]['label']}")




# =============================================================================
# CLI Entry Points
# =============================================================================


@app.function(
    image=image,
    gpu="H200:8",
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        CHECKPOINTS_PATH.as_posix(): checkpoints_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
    timeout=24 * 60 * 60,
    experimental_options={
        "efa_enabled": True,
    },
)
async def train(
    run_name: str = "qwen3-4b-haiku",
    judge_type = ACTIVE_JUDGE_TYPE,
    judge_model_size = ACTIVE_JUDGE_MODEL_SIZE,
):
    """Single-node GRPO training on Modal."""
    from datetime import datetime

    cfg = get_config(run_name=run_name, judge_type=judge_type, judge_model_size=judge_model_size)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_short = cfg.model_name.split("/")[-1]
    experiment_name = f"{run_name}-{model_short}-{timestamp}"

    await hf_cache_vol.reload.aio()
    await checkpoints_volume.reload.aio()

    _init_ray(0, SINGLE_NODE_MASTER_ADDR, SINGLE_NODE_MASTER_ADDR, 1)

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Dashboard URL: {tunnel.url}")
        print(f"Experiment: {experiment_name}")
        await run_training(cfg, 1, SINGLE_NODE_MASTER_ADDR, experiment_name)