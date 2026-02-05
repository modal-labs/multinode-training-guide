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
REMOTE_EVAL_SCRIPT_PATH = "/root/eval.py"

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
HF_DATASET = "donmaclean/LongMIT-128K"
PREPROCESSED_DIR = f"{DATA_DIR}/longmit-128k"
MAX_SFT_TOKENS = 131_072
MEGATRON_CHECKPOINT = f"{CHECKPOINTS_DIR}/glm47-megatron"
PREP_CPU = 32

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
        "transformers",
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
    cpu=PREP_CPU,
)
def prep_dataset():
    """Download and prepare LongMIT-128K for Megatron training."""

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from megatron.bridge.data.datasets.utils import build_index_files

    data_volume.reload()

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # Download dataset from HuggingFace
    print(f"Downloading dataset: {HF_DATASET}")
    dataset = load_dataset(HF_DATASET, split="train", trust_remote_code=True)
    print(f"Dataset loaded: {len(dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL, use_fast=True, trust_remote_code=True
    )

    def format_longmit(example):
        passages = "\n".join(
            [
                f"Passage {i + 1}:\n{doc['content']}"
                for i, doc in enumerate(example["all_docs"])
            ]
        )
        prompt = (
            "Answer the question based on the given passages.\n\n"
            f"{passages}\n\n"
            f"Question: {example['question']}\nAnswer:"
        )
        answer = example["answer"]
        text = prompt + answer
        token_count = len(tokenizer(text).input_ids)
        return {"input": prompt, "output": answer, "n_tokens": token_count}

    print(f"Formatting for SFT and filtering to <= {MAX_SFT_TOKENS} tokens")
    original_len = len(dataset)
    dataset = dataset.map(
        format_longmit,
        remove_columns=dataset.column_names,
        num_proc=PREP_CPU,
    )
    dataset = dataset.filter(
        lambda ex: ex["n_tokens"] <= MAX_SFT_TOKENS,
        num_proc=PREP_CPU,
    )
    filtered_len = len(dataset)
    print(
        "Filtered dataset size: "
        f"{filtered_len} (from {original_len} examples)"
    )

    # Save as JSONL for FinetuningDatasetConfig
    # FinetuningDatasetConfig expects "training.jsonl" by default
    train_jsonl = f"{PREPROCESSED_DIR}/training.jsonl"
    print(f"Saving to: {train_jsonl}")

    with open(train_jsonl, "w") as f:
        for example in dataset:
            # Megatron SFT expects "input" and "output" fields
            json.dump({"input": example["input"], "output": example["output"]}, f)
            f.write("\n")

    size = os.path.getsize(train_jsonl)
    print(f"Created {train_jsonl} ({size:,} bytes)")
    print(f"Total examples: {len(dataset)}")

    # Remove any existing index files before rebuilding. By default, indexes will not
    # be rebuilt even if the source json file is modified.
    for idx_file in glob.glob(f"{train_jsonl}.idx*"):
        print(f"Removing old index file: {idx_file}")
        os.remove(idx_file)

    print("Building index files...")

    build_index_files(
        dataset_paths=[train_jsonl],
        newline_int=10,  # ASCII newline character
        workers=PREP_CPU,
        index_mapping_dir=None,  # Write next to the jsonl
    )

    # Verify index files were created
    assert os.path.exists(f"{train_jsonl}.idx.npy"), f"{train_jsonl}.idx.npy missing"
    assert os.path.exists(f"{train_jsonl}.idx.info"), f"{train_jsonl}.idx.info missing"
    print("Index files created successfully")

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
def train_lora():
    """
    Train GLM-4.7 with LoRA

    This function uses:
    - TP=2
    - PP=4
    - EP=4
    - micro batch size=1
    - global batch size=32
    - FULL recompute

    Note that this means DP = N_NODES * GPUs per node / (TP x PP) = 4 * 8 / (2 x 4) = 4

    Memory note (per GPU):
    - Long-context training is memory heavy; start with mbs=1 and scale up carefully.

    Tuning:

    This function targets LongMIT-128K, which contains long-context examples. Start
    conservatively with micro batch size 1 and increase only if you have headroom.

    If you do see OOMs, try (in order):
    - Reducing the micro batch size in coordination with the global batch size
    - Upgrading to H200 GPUs
    - Increasing tensor parallelism
    - Increasing pipeline parallelism
    """

    # Environment optimizations

    # Required for MoE memory patterns - allows fragmentation-friendly allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Layer norm SM margin - prevents persistent LN kernels from blocking DP comm
    os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] = "16"
    os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] = "16"

    # Enable TP overlap on H100 (Hopper)
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    num_nodes = N_NODES
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )
    master_port = 29500

    print(f"Node {node_rank}/{num_nodes}, Master: {master_addr}:{master_port}")
    print("Config: TP=2, PP=4, EP=4, mbs=1, FULL recompute")
    print("Goal: Start safe for long context and scale up if stable")

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
        "1",  # Long-context default; increase only if stable
        "--global_batch_size",
        "32",  # Must be divisible by micro batch size × DP = 1 × 4 = 4
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

    checkpoints_volume.commit()
    return {"status": "optimized_v3_training_complete"}


@app.function(
    image=nemo_image,
    gpu="B200:8",
    volumes={
        MODELS_DIR: models_volume,
        DATA_DIR: data_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def eval_lora(
    run_id: str = "",
    checkpoint_dir: str = "",
    checkpoint_subdir: str = "",
    preprocessed_dir: str = PREPROCESSED_DIR,
    context: str = "128k",
    lora_rank: int = 128,
    eval_iters: int = 2,
):
    """Run Megatron native evaluation on the latest LoRA checkpoint."""
    import shutil
    import time

    def resolve_checkpoint_dir():
        if checkpoint_dir:
            if checkpoint_dir.startswith("/"):
                return checkpoint_dir
            return f"{CHECKPOINTS_DIR}/{checkpoint_dir}"
        if checkpoint_subdir:
            return f"{CHECKPOINTS_DIR}/{checkpoint_subdir}"
        if run_id:
            return f"{CHECKPOINTS_DIR}/glm47_lora_{run_id}"
        return f"{CHECKPOINTS_DIR}/glm47_lora"

    def link_or_copy(src, dst):
        if os.path.exists(dst):
            return
        try:
            os.symlink(src, dst)
            return
        except FileExistsError:
            return
        except OSError:
            shutil.copy(src, dst)

    def ensure_validation_files(dataset_dir):
        train_file = os.path.join(dataset_dir, "training.jsonl")
        val_file = os.path.join(dataset_dir, "validation.jsonl")

        if os.path.exists(val_file):
            return
        if not os.path.exists(train_file):
            raise RuntimeError(f"Training dataset not found: {train_file}")

        print("validation.jsonl missing; linking to training.jsonl for eval")
        link_or_copy(train_file, val_file)
        for suffix in [".idx.npy", ".idx.info"]:
            train_idx = f"{train_file}{suffix}"
            val_idx = f"{val_file}{suffix}"
            if os.path.exists(train_idx):
                link_or_copy(train_idx, val_idx)

    # Environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] = "16"
    os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] = "16"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )
    master_port = 29500

    resolved_preprocessed_dir = preprocessed_dir
    if not resolved_preprocessed_dir.startswith("/"):
        resolved_preprocessed_dir = os.path.join(DATA_DIR, resolved_preprocessed_dir)

    resolved_checkpoint_dir = resolve_checkpoint_dir()

    print(f"Node {node_rank}/{N_NODES}, Master: {master_addr}:{master_port}")

    if not os.path.exists(resolved_checkpoint_dir):
        raise RuntimeError(f"Checkpoint not found: {resolved_checkpoint_dir}")
    if not os.path.exists(MEGATRON_CHECKPOINT):
        raise RuntimeError(f"Base checkpoint not found: {MEGATRON_CHECKPOINT}")
    if not os.path.exists(resolved_preprocessed_dir):
        raise RuntimeError(
            f"Preprocessed data not found: {resolved_preprocessed_dir}"
        )

    ensure_validation_files(resolved_preprocessed_dir)

    tracker_file = os.path.join(
        resolved_checkpoint_dir, "latest_checkpointed_iteration.txt"
    )
    if not os.path.exists(tracker_file):
        raise RuntimeError(f"No tracker file found: {tracker_file}")

    with open(tracker_file, "r") as f:
        iteration = int(f.read().strip())

    print("\n" + "=" * 70)
    print(f"EVALUATING CHECKPOINT: iter_{iteration:07d}")
    print("=" * 70)
    print(f"Checkpoint: {resolved_checkpoint_dir}")
    print(f"Base checkpoint: {MEGATRON_CHECKPOINT}")
    print(f"Dataset: {resolved_preprocessed_dir}")
    print("=" * 70 + "\n")

    script_args = [
        "--preprocessed_dir",
        resolved_preprocessed_dir,
        "--checkpoint_dir",
        resolved_checkpoint_dir,
        "--base_checkpoint",
        MEGATRON_CHECKPOINT,
        "--full_checkpoint",
        "--hf_model",
        HF_MODEL,
        "--context",
        context,
        "--lora_rank",
        str(lora_rank),
        "--eval_iters",
        str(eval_iters),
    ]

    cmd = [
        "torchrun",
        f"--nnodes={N_NODES}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--nproc_per_node=8",
        REMOTE_EVAL_SCRIPT_PATH,
        *script_args,
    ]

    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Eval exited with code {process.returncode}")

    # Ensure logs are flushed before returning.
    time.sleep(1)

    print("[DONE] Evaluation complete")
    return {
        "status": "ok",
        "iteration": iteration,
        "checkpoint_dir": resolved_checkpoint_dir,
    }
