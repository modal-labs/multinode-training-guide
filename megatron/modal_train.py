"""
GLM-4.7 LoRA Training on Modal

Usage:
    modal run --detach modal_train.py::download_and_convert  # Download + convert
    modal run modal_train.py::prep_dataset                   # Download and preprocess dataset
    modal run --detach modal_train.py::train_lora            # Run distributed training
"""

import glob
import json
import os
import pathlib
import subprocess

import modal
import modal.experimental

LOCAL_CODE_DIR = pathlib.Path(__file__).parent.resolve()

REMOTE_CODE_DIR = "/root/"
REMOTE_TRAIN_SCRIPT_PATH = "/root/train.py"

app = modal.App("glm47-lora")

# Volumes for model, data, and checkpoints
models_volume = modal.Volume.from_name("big-model-hfcache", create_if_missing=True)
data_volume = modal.Volume.from_name("glm47-training-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("glm47-checkpoints", create_if_missing=True)

# In-container paths for model, data, and checkpoints
MODELS_DIR = "/models"
HF_CACHE = "/root/.cache/huggingface"
DATA_DIR = "/data"
CHECKPOINTS_DIR = "/checkpoints"
HF_MODEL = "zai-org/GLM-4.7"
HF_DATASET = "glaiveai/glaive-code-assistant"
PREPROCESSED_DIR = f"{DATA_DIR}/glaive-code-assistant"
MEGATRON_CHECKPOINT = f"{CHECKPOINTS_DIR}/glm47-megatron"

# Number of nodes in the cluster
N_NODES = 4  # 4 nodes x 8 GPUs = 32 GPUs

# Simple image for downloading
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface_hub==0.36.0",
        "transformers==4.57.4",
        "torch==2.9.1",
        "safetensors==0.7.0",
        "sentencepiece==0.2.1",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # Enable fast data transfer from HF Hub
)

# NeMo 25.11
nemo_image = (
    modal.Image.from_registry("nvcr.io/nvidia/nemo:25.11")
    .entrypoint([])
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc-dev",
        "libnl-route-3-200",
    )
    .uv_pip_install(
        "wandb==0.23.1",
        "datasets==3.1.0",
    )
    .run_commands(f"rm -Rf {HF_CACHE}")
    .add_local_dir(LOCAL_CODE_DIR, remote_path=REMOTE_CODE_DIR)
)


@app.function(
    image=download_image,
    volumes={HF_CACHE: models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,  # 4 hours - 358B model is huge
)
def download_model():
    """Download GLM-4.7 (358B) to volume."""
    from huggingface_hub import snapshot_download

    models_volume.reload()

    print(f"Downloading {HF_MODEL} (358B MoE - this will take a while)...")

    path = snapshot_download(
        HF_MODEL,
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"Downloaded to: {path}")
    models_volume.commit()
    return {"path": path}


@app.function(
    image=nemo_image,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    cpu=4,
)
def prep_dataset():
    """Download and prepare glaive-code-assistant for Megatron training."""

    from datasets import load_dataset
    from megatron.bridge.data.datasets.utils import build_index_files

    data_volume.reload()

    # Create output directory
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # Download dataset from HuggingFace
    print(f"Downloading dataset: {HF_DATASET}")
    dataset = load_dataset(HF_DATASET, split="train")
    print(f"Dataset loaded: {len(dataset)} examples")

    # Save as JSONL for FinetuningDatasetConfig
    # FinetuningDatasetConfig expects "training.jsonl" by default
    train_jsonl = f"{PREPROCESSED_DIR}/training.jsonl"
    print(f"Saving to: {train_jsonl}")

    with open(train_jsonl, "w") as f:
        for example in dataset:
            # Megatron SFT expects "input" and "output" fields
            json.dump({"input": example["question"], "output": example["answer"]}, f)
            f.write("\n")

    # Report file size
    size = os.path.getsize(train_jsonl)
    print(f"Created {train_jsonl} ({size:,} bytes)")
    print(f"Total examples: {len(dataset)}")

    # Remove any existing index files before rebuilding
    for idx_file in glob.glob(f"{train_jsonl}.idx*"):
        print(f"Removing old index file: {idx_file}")
        os.remove(idx_file)

    # Build index files using Megatron's official API
    print("Building index files...")

    build_index_files(
        dataset_paths=[train_jsonl],
        newline_int=10,  # ASCII newline character
        workers=os.cpu_count(),
        index_mapping_dir=None,  # Write next to the jsonl
    )

    # Verify index files were created
    assert os.path.exists(f"{train_jsonl}.idx.npy"), f"{train_jsonl}.idx.npy missing"
    assert os.path.exists(f"{train_jsonl}.idx.info"), f"{train_jsonl}.idx.info missing"
    print("Index files created successfully")

    # List all files created
    for item in os.listdir(PREPROCESSED_DIR):
        item_path = os.path.join(PREPROCESSED_DIR, item)
        if os.path.isfile(item_path):
            fsize = os.path.getsize(item_path)
            print(f"  📄 {item} ({fsize:,} bytes)")

    data_volume.commit()
    return {"preprocessed_dir": PREPROCESSED_DIR, "examples": len(dataset)}


@app.function(
    image=nemo_image,
    gpu="B200",
    volumes={
        HF_CACHE: models_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=21600,  # 6 hours for conversion
    cpu=16,
    memory=1048576,  # 1TB max - Modal limit
    ephemeral_disk=2048000,  # 2TB scratch disk for checkpoint writes
)
def convert_to_megatron():
    """Convert GLM-4.7 HF to Megatron format."""
    import torch
    import torch.distributed as dist
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.model_load_save import save_megatron_model

    models_volume.reload()

    print(f"Converting {HF_MODEL} (358B) to Megatron format...")
    print(f"Output: {MEGATRON_CHECKPOINT}")

    print("Loading HuggingFace model...")
    bridge = AutoBridge.from_hf_pretrained(
        HF_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Converting to Megatron format...")
    megatron_model = bridge.to_megatron_model(
        wrap_with_ddp=False, use_cpu_initialization=True
    )

    print("Saving checkpoint in torch_dist format...")
    save_megatron_model(
        megatron_model,
        MEGATRON_CHECKPOINT,
        ckpt_format="torch_dist",
        hf_tokenizer_path=HF_MODEL,
    )

    print("Conversion complete! Saving checkpoint to volume...")

    checkpoints_volume.commit()
    return {"megatron_path": MEGATRON_CHECKPOINT}


@app.function(image=modal.Image.debian_slim(), timeout=36000)
def download_and_convert():
    """Orchestrate download (CPU) then convert (GPU) as separate steps."""
    print("Step 1/2: Downloading model (on CPU)...")
    download_result = download_model.remote()
    print(f"Download complete: {download_result}")

    print("Step 2/2: Converting to Megatron format (on GPU)...")
    convert_result = convert_to_megatron.remote()
    print(f"Convert complete: {convert_result}")

    return {"download": download_result, "convert": convert_result}


@app.function(
    image=nemo_image,
    gpu="H100:8",
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
def train_lora():
    """Train GLM-4.7 with LoRA on pre-built dataset."""

    # The following environment variables are required to avoid OOMs when training large models.
    # These can be tuned to improve performance, but they need to be tuned along with TP, PP, EP, and DP settings.
    #
    # Disable `torch.compile` to allocate less memory for the model.
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    # Enable expandable segments for more efficient memory allocations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Disable NVLS to avoid allocating extra memory during NCCL communication.
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    # Disable fused attn to avoid cudnn bugs with GLM-4.7
    os.environ["NVTE_FUSED_ATTN"] = "0"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    num_nodes = N_NODES
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )
    master_port = 29500

    print(f"Node {node_rank}/{num_nodes}, Master: {master_addr}:{master_port}")

    # Verify preprocessed data exists
    if not os.path.exists(PREPROCESSED_DIR):
        raise RuntimeError(
            f"Preprocessed data not found at {PREPROCESSED_DIR}. "
            "Run: modal run modal_train.py::prep_dataset first!"
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

    script_args = [
        "--preprocessed_dir",
        PREPROCESSED_DIR,
        "--megatron_checkpoint",
        MEGATRON_CHECKPOINT,
        "--checkpoints_dir",
        CHECKPOINTS_DIR,
        "--hf_model",
        HF_MODEL,
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


REMOTE_OPTIMIZED_SCRIPT = "/root/train_optimized.py"


@app.function(
    image=nemo_image,
    gpu="H100:8",
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
def train_lora_optimized():
    """
    Train GLM-4.7 with LoRA - OPTIMIZED VERSION.

    Key differences from train_lora():
    1. Uses NVIDIA-recommended parallelism (TP=2, PP=4, EP=4)
    2. NO activation recomputation needed
    3. torch.compile ENABLED
    4. NVLS ENABLED for better NCCL performance
    5. Higher micro_batch_size (2) for better throughput

    Memory analysis (per GPU):
    - Model weights: ~23 GB (PP=4 reduces layer count per GPU)
    - Activations:   ~24 GB (with mbs=2)
    - Other:         ~6 GB
    - TOTAL:         ~53 GB → 27 GB headroom on H100
    """

    # OPTIMIZED environment - only truly necessary settings
    # Enable expandable segments for MoE memory patterns (always needed for MoE)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # REMOVED: TORCHDYNAMO_DISABLE - torch.compile works with PP=4 config
    # REMOVED: NCCL_NVLS_ENABLE=0 - NVLS improves performance

    # INVESTIGATE: May still be needed for GLM-4.7 cudnn compatibility
    # Try without first, enable only if you see errors
    # os.environ["NVTE_FUSED_ATTN"] = "0"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    num_nodes = N_NODES
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )
    master_port = 29500

    print(
        f"[OPTIMIZED] Node {node_rank}/{num_nodes}, Master: {master_addr}:{master_port}"
    )
    print(
        "Config: TP=2, PP=4, EP=4, micro_batch=2, NO recompute, torch.compile ENABLED"
    )

    # Verify preprocessed data exists
    if not os.path.exists(PREPROCESSED_DIR):
        raise RuntimeError(
            f"Preprocessed data not found at {PREPROCESSED_DIR}. "
            "Run: modal run modal_train.py::prep_dataset first!"
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

    script_args = [
        "--preprocessed_dir",
        PREPROCESSED_DIR,
        "--megatron_checkpoint",
        MEGATRON_CHECKPOINT,
        "--checkpoints_dir",
        CHECKPOINTS_DIR,
        "--hf_model",
        HF_MODEL,
        "--micro_batch_size",
        "1",  # Safe for H100 80GB; use 2 for H200
        # Uncomment if OOM still occurs:
        # "--moe_layer_recompute",  # Adds ~5-10% overhead but saves significant memory
    ]

    cmd = [
        "torchrun",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--nproc_per_node=8",
        REMOTE_OPTIMIZED_SCRIPT,
        *script_args,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    checkpoints_volume.commit()
    return {"status": "optimized_training_complete"}


REMOTE_OPTIMIZED_V2_SCRIPT = "/root/train_optimized_v2.py"


@app.function(
    image=nemo_image,
    gpu="H100:8",
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
def train_lora_optimized_v2():
    """
    Train GLM-4.7 with LoRA - OPTIMIZED V2 (Performance-tuned).

    Key optimizations:
    1. micro_batch_size=2 for better compute utilization
    2. FULL activation recomputation (~30% overhead) to fit mbs=2 in memory
    3. Manual GC alignment to reduce jitter
    4. Larger DDP bucket size for better batching
    5. Environment tunings (SM margins)

    NOTE: Communication overlaps (overlap_grad_reduce, overlap_param_gather)
    are INCOMPATIBLE with LoRA/PEFT - they expect all params to have gradients.

    The trade-off:
    - mbs=2 gives ~100% more compute per step vs mbs=1
    - Full recompute adds ~30% overhead
    - Net gain: ~40% better throughput than mbs=1 without recompute

    Expected performance:
    - ~350-400 TFLOP/s/GPU (vs ~250 TFLOP/s with mbs=1)
    - ~5-6s per iteration (vs 7-10s with mbs=1)
    - Stable memory usage with full recompute
    """

    # Environment optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # NOTE: TORCH_NCCL_AVOID_RECORD_STREAMS=1 is incompatible with PP batch P2P
    os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] = "16"
    os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] = "16"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    num_nodes = N_NODES
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )
    master_port = 29500

    print(
        f"[OPTIMIZED V2] Node {node_rank}/{num_nodes}, Master: {master_addr}:{master_port}"
    )
    print("Config: TP=2, PP=4, EP=4, mbs=2, FULL recompute")
    print("Optimizations: manual_gc, larger bucket_size, SM margins")

    # Verify preprocessed data exists
    if not os.path.exists(PREPROCESSED_DIR):
        raise RuntimeError(
            f"Preprocessed data not found at {PREPROCESSED_DIR}. "
            "Run: modal run modal_train.py::prep_dataset first!"
        )

    print(f"Using pre-built dataset from: {PREPROCESSED_DIR}")

    # Verify checkpoint exists
    if not os.path.exists(MEGATRON_CHECKPOINT):
        raise RuntimeError(f"Checkpoint not found at {MEGATRON_CHECKPOINT}")

    script_args = [
        "--preprocessed_dir",
        PREPROCESSED_DIR,
        "--megatron_checkpoint",
        MEGATRON_CHECKPOINT,
        "--checkpoints_dir",
        CHECKPOINTS_DIR,
        "--hf_model",
        HF_MODEL,
        "--micro_batch_size",
        "2",  # Good compute utilization
        "--global_batch_size",
        "32",
        # Key optimizations in V2:
        # - FULL recompute (default ON) - required for mbs=2 to fit in memory
        # - manual_gc_interval=10 - reduces jitter
        # - bucket_size=100MB - better DDP batching
        # - SM margins for LayerNorm - prevents blocking DP comm
        # Trade-off: 30% recompute overhead, but mbs=2 gives 100% more compute
        # Net gain: ~40% better than mbs=1 without recompute
    ]

    cmd = [
        "torchrun",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--nproc_per_node=8",
        REMOTE_OPTIMIZED_V2_SCRIPT,
        *script_args,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    checkpoints_volume.commit()
    return {"status": "optimized_v2_training_complete"}


REMOTE_OPTIMIZED_V3_SCRIPT = "/root/train_optimized_v3.py"


@app.function(
    image=nemo_image,
    gpu="H100:8",
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
def train_lora_optimized_v3():
    """
    Train GLM-4.7 with LoRA - OPTIMIZED V3 (Leverage unused VRAM).

    V2 status: mbs=2, full recompute, ~63GB peak, ~17GB headroom, ~375 TFLOP/s/GPU

    V3 experiment: Try mbs=3 with full recompute to use the ~17GB/GPU headroom
    - Expected: ~80% of H100 memory utilized
    - Target: ~50% throughput improvement over V2

    Memory math (per GPU):
    - V2 (mbs=2): ~63GB peak → 17GB headroom
    - V3 (mbs=3): ~79GB estimated → 1GB headroom (risky but worth trying)

    If mbs=3 OOMs, fallback options:
    1. Use recompute_num_layers=2 with mbs=2 (less overhead)
    2. Stay with V2 config (stable, decent throughput)
    """

    # Environment optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] = "16"
    os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] = "16"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    num_nodes = N_NODES
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )
    master_port = 29500

    print(
        f"[OPTIMIZED V3] Node {node_rank}/{num_nodes}, Master: {master_addr}:{master_port}"
    )
    print("Config: TP=2, PP=4, EP=4, mbs=3, FULL recompute")
    print("Goal: Leverage ~17GB/GPU unused VRAM for ~50% throughput gain")

    # Verify preprocessed data exists
    if not os.path.exists(PREPROCESSED_DIR):
        raise RuntimeError(
            f"Preprocessed data not found at {PREPROCESSED_DIR}. "
            "Run: modal run modal_train.py::prep_dataset first!"
        )

    print(f"Using pre-built dataset from: {PREPROCESSED_DIR}")

    # Verify checkpoint exists
    if not os.path.exists(MEGATRON_CHECKPOINT):
        raise RuntimeError(f"Checkpoint not found at {MEGATRON_CHECKPOINT}")

    script_args = [
        "--preprocessed_dir",
        PREPROCESSED_DIR,
        "--megatron_checkpoint",
        MEGATRON_CHECKPOINT,
        "--checkpoints_dir",
        CHECKPOINTS_DIR,
        "--hf_model",
        HF_MODEL,
        "--micro_batch_size",
        "3",  # V3: Push mbs to 3 to use headroom
        "--global_batch_size",
        "36",  # Must be divisible by mbs × DP = 3 × 4 = 12
        # Full recompute enabled by default
    ]

    cmd = [
        "torchrun",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--nproc_per_node=8",
        REMOTE_OPTIMIZED_V3_SCRIPT,
        *script_args,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    checkpoints_volume.commit()
    return {"status": "optimized_v3_training_complete"}
