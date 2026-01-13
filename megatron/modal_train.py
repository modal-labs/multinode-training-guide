"""
GLM-4.7 LoRA Training on Modal

Usage:
    modal run train_glm47.py --download        # Download model (4+ hours)
    modal run train_glm47.py --convert         # Convert HF to Megatron
    modal run preprocess_dataset.py            # Build dataset indexes (single process)
    modal run --detach train_glm47.py --train  # Run distributed training
"""

import os

import modal
import modal.experimental

LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CODE_DIR = "/root/"
REMOTE_TRAIN_SCRIPT_PATH = "/root/train.py"

app = modal.App("glm47-lora")

# Volumes - separate from GLM-4.5 to avoid conflicts
models_volume = modal.Volume.from_name("glm47-models", create_if_missing=True)
data_volume = modal.Volume.from_name("glm45-training-data")  # Reuse training data
checkpoints_volume = modal.Volume.from_name("glm47-checkpoints", create_if_missing=True)

MODELS_DIR = "/models"
DATA_DIR = "/data"
CHECKPOINTS_DIR = "/checkpoints"
HF_MODEL = "zai-org/GLM-4.7"
PREPROCESSED_DIR = f"{DATA_DIR}/preprocessed_glm47"

# Simple image for downloading
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "transformers", "torch", "safetensors", "sentencepiece")
    .env({"HF_HOME": "/models/huggingface"})
)

# NeMo 25.11 with fixes
nemo_image = (
    modal.Image.from_registry("nvcr.io/nvidia/nemo:25.11")
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc-dev",
        "libnl-route-3-200",
    )
    .pip_install("wandb")
    .run_commands(
        # Fix missing 'warnings' import bug in Megatron-Core
        "sed -i '1s/^/import warnings\\n/' /opt/megatron-lm/megatron/core/model_parallel_config.py",
        # Fix Manager().Queue() crash in Modal
        r"""sed -i 's/_results_queue = ctx.Manager().Queue()/_results_queue = ctx.Queue()/' /opt/megatron-lm/megatron/core/dist_checkpointing/strategies/filesystem_async.py""",
        # Disable fully_parallel_save by default - Modal multiprocessing is limited
        r"""sed -i 's/fully_parallel_save: bool = True/fully_parallel_save: bool = False/' /opt/Megatron-Bridge/src/megatron/bridge/training/config.py""",
    )
    .env({"HF_HOME": "/models/huggingface"})
    .add_local_dir(LOCAL_CODE_DIR, remote_path=REMOTE_CODE_DIR)
)


@app.function(
    image=download_image,
    volumes={MODELS_DIR: models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,  # 4 hours - 358B model is huge
)
def download_model():
    """Download GLM-4.7 (358B) to volume."""
    import os
    from huggingface_hub import snapshot_download

    models_volume.reload()

    cache_dir = f"{MODELS_DIR}/huggingface"
    print(f"Downloading {HF_MODEL} (358B MoE - this will take a while)...")

    path = snapshot_download(
        HF_MODEL,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"Downloaded to: {path}")
    models_volume.commit()
    return {"path": path}


MEGATRON_CHECKPOINT = f"{CHECKPOINTS_DIR}/glm47-megatron"  # Checkpoint is in checkpoints volume


@app.function(
    image=nemo_image,
    gpu="B200",  # Single GPU - avoids distributed checkpoint bugs
    volumes={MODELS_DIR: models_volume, CHECKPOINTS_DIR: checkpoints_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=21600,  # 6 hours for conversion
    memory=1048576,  # 1TB max - Modal limit
    ephemeral_disk=2048000,  # 2TB scratch disk for checkpoint writes
)
def convert_to_megatron():
    """Convert GLM-4.7 HF to Megatron format."""
    import subprocess
    import sys

    models_volume.reload()

    print(f"Converting {HF_MODEL} (358B) to Megatron format...")
    print(f"Output: {MEGATRON_CHECKPOINT}")

    cmd = [
        sys.executable,
        "-c",
        f"""
import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import save_megatron_model

try:
    # Load HF model
    print("Loading HuggingFace model...")
    bridge = AutoBridge.from_hf_pretrained(
        "{HF_MODEL}",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Convert to Megatron model
    print("Converting to Megatron format...")
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False, use_cpu_initialization=True)

    # Save with torch_dist format (fully_parallel_save disabled via image patch)
    print("Saving checkpoint in torch_dist format...")
    save_megatron_model(
        megatron_model,
        "{MEGATRON_CHECKPOINT}",
        ckpt_format="torch_dist",
        hf_tokenizer_path="{HF_MODEL}",
    )
    print("Conversion complete!")
finally:
    # Clean up distributed process group if initialized
    if dist.is_initialized():
        dist.destroy_process_group()
"""
    ]

    print("Running conversion...")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed with code {result.returncode}")

    checkpoints_volume.commit()
    return {"megatron_path": MEGATRON_CHECKPOINT}


# 358B MoE needs more GPUs than 106B
# Estimate: ~3x the parallelism of GLM-4.5-Air
N_NODES = 4  # 4 nodes x 8 GPUs = 32 GPUs


@app.function(
    image=nemo_image,
    gpu="B200:8",
    volumes={
        MODELS_DIR: models_volume,
        DATA_DIR: data_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=86400,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train_lora(run_id: str):
    """Train GLM-4.7 with pre-built dataset (no dataset construction)."""
    import subprocess
    import os

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["NCCL_TIMEOUT"] = "7200"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_IB_TIMEOUT"] = "23"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TORCH_DIST_INIT_BARRIER_TIMEOUT"] = "7200"
    # os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"  # Experimental for P2P, causes warnings
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    num_nodes = N_NODES
    master_addr = cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    master_port = 29500

    print(f"Node {node_rank}/{num_nodes}, Master: {master_addr}:{master_port}")

    models_volume.reload()
    data_volume.reload()
    checkpoints_volume.reload()

    # Verify preprocessed data exists
    if not os.path.exists(PREPROCESSED_DIR):
        raise RuntimeError(
            f"Preprocessed data not found at {PREPROCESSED_DIR}. "
            "Run: modal run preprocess_dataset.py first!"
        )

    print(f"Using pre-built dataset from: {PREPROCESSED_DIR}")
    for item in os.listdir(PREPROCESSED_DIR):
        item_path = os.path.join(PREPROCESSED_DIR, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            print(f"  📄 {item} ({size:,} bytes)")

    # Verify checkpoint exists
    if not os.path.exists(MEGATRON_CHECKPOINT):
        raise RuntimeError(f"Checkpoint not found at {MEGATRON_CHECKPOINT}")

    # run_id passed from local entrypoint - same for all nodes
    print(f"Using run_id: {run_id}")

    script_args = [
        "--run_id", run_id,
        "--preprocessed_dir", PREPROCESSED_DIR,
        "--megatron_checkpoint", MEGATRON_CHECKPOINT,
        "--checkpoints_dir", CHECKPOINTS_DIR,
        "--hf_model", HF_MODEL,
    ]

    cmd = [
        "torchrun",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--nproc_per_node=8",
        REMOTE_TRAIN_SCRIPT_PATH,
        *script_args,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    # Persist training checkpoints to volume
    checkpoints_volume.commit()
    return {"status": "training_complete"}


@app.local_entrypoint()
def main(
    download: bool = False,
    convert: bool = False,
    train: bool = False,
):
    """
    GLM-4.7 LoRA Training

    Usage:
        modal run train_glm47.py --download        # Download model (4+ hours)
        modal run train_glm47.py --convert         # Convert HF to Megatron
        modal run preprocess_dataset.py            # Build dataset indexes
        modal run --detach train_glm47.py --train  # Run training
    """
    if download:
        print("Downloading GLM-4.7 (358B) - this will take several hours...")
        result = download_model.remote()
        print(f"Done: {result}")
    elif convert:
        print("Converting GLM-4.7 to Megatron format...")
        result = convert_to_megatron.remote()
        print(f"Done: {result}")
    elif train:
        print("Starting GLM-4.7 LoRA training with pre-built dataset...")
        # Generate WandB run_id locally, pass to all nodes
        import wandb
        wandb.init(project="glm47-lora")
        run_id = wandb.run.id
        print(f"Generated WandB run_id: {run_id}")
        wandb.finish()
        result = train_lora.remote(run_id=run_id)
        print(f"Done: {result}")
    else:
        print("Usage:")
        print("  modal run train_glm47.py --download        # Download model")
        print("  modal run train_glm47.py --convert         # Convert to Megatron")
        print("  modal run preprocess_dataset.py            # Build dataset indexes")
        print("  modal run --detach train_glm47.py --train  # Run training")
