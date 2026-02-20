"""
GLM-4.7 Training via ms-swift Megatron v4
"""

import os

import modal
import modal.experimental

HF_MODEL = "zai-org/GLM-4.7"
MODEL_NAME = "GLM-4.7"

DEFAULT_MAX_EPOCHS = 4
DEFAULT_MAX_LENGTH = 2048
DEFAULT_LORA_RANK = 128
DEFAULT_LORA_ALPHA = 32

TP_SIZE = 2
PP_SIZE = 4
EP_SIZE = 4
CP_SIZE = 1

WANDB_PROJECT = "glm-4-7-sft"

app = modal.App("example-msswift-glm_4_7-lora")

# Volumes — use volumes V2
models_volume = modal.Volume.from_name(
    "glm-4-7-models", create_if_missing=True, version=2
)
data_volume = modal.Volume.from_name("example-msswift-glm-4-7-data", create_if_missing=True, version=2)
checkpoints_volume = modal.Volume.from_name(
    "example-msswift-glm-4-7-checkpoints", create_if_missing=True, version=2
)

MODELS_DIR = "/models"
DATA_DIR = "/data"
CHECKPOINTS_DIR = "/checkpoints"

# N_NODES via env var (evaluated at decoration time by @clustered).
# Default 2 (2 x B200:8).
# Usage: N_NODES=2 modal run --detach train_glm_4_7.py --train ...
N_NODES = int(os.environ.get("N_NODES", "4"))


# ------------------------------------------------------------
# Image dependencies
# ------------------------------------------------------------
PYTHON_VERSION = "3.11"
TORCH_VERSION = "2.8.0"
FLASH_ATTN_VERSION = "2.8.3"
MEGATRON_VERSION = "0.14.1"

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==0.27.1",
        "transformers>=4.50",
        "torch==2.5.1",
        "safetensors==0.4.5",
        "datasets>=2.14.0",
    )
    .env(
        {
            "HF_HOME": "/models/huggingface",
        }
    )
)

# ms-swift v4, PyTorch 2.8, CUDA 12.8, FA 2.8, Megatron 0.14.1
msswift_v4_image = (
    modal.Image.from_registry(
        "baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3"
    )
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc-dev",
        "libnl-route-3-200",
    )
    # Force-remove pre-installed transformers + ms-swift to get compatible versions.
    # Baseten ships older transformers that may lack Glm4MoeForCausalLM support.
    .run_commands(
        "pip uninstall -y transformers ms-swift swift 2>/dev/null; true",
    )
    .pip_install(
        # Reinstall transformers with GLM-4 MoE support
        # Stay in 4.x — Baseten's peft is incompatible with transformers 5.x
        "transformers==4.57.3",
        # ms-swift v4, patched to support pipeline parallelism and n_steps in logging
        "ms-swift @ git+https://github.com/joyliu-q/ms-swift.git@joy/patch-pp-log-emission-issue",
        "einops==0.8.2",
        "wandb==0.19.1",
    )
    .env(
        {
            "HF_HOME": "/models/huggingface",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NVTE_FWD_LAYERNORM_SM_MARGIN": "16",
            "NVTE_BWD_LAYERNORM_SM_MARGIN": "16",
            "SWIFT_DISABLE_LOGGING": "1",
            "SWIFT_DISABLE_LOGGING_CALLBACK": "1",
            # Disable torch.compile — inductor materializes the full logits tensor
            # (122K tokens × 152K vocab × fp32 = 69 GB) instead of using fused CE loss
            "TORCHDYNAMO_DISABLE": "1",
            # NCCL debug — surface real error behind "unhandled cuda error"
            "NCCL_DEBUG": "WARN",
            "NCCL_DEBUG_SUBSYS": "ALL",
            "NCCL_TIMEOUT": "300",
            "TORCH_NCCL_ENABLE_MONITORING": "1",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "20000",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_DESYNC_DEBUG": "1",
            "TORCH_NCCL_BLOCKING_WAIT": "1",
        }
    )
)


@app.function(
    image=download_image,
    volumes={MODELS_DIR: models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,  # 4 hours
)
def download_model(force: bool = False):
    from huggingface_hub import snapshot_download

    models_volume.reload()

    local_dir = f"{MODELS_DIR}/{MODEL_NAME}"
    print(f"Downloading {HF_MODEL} to {local_dir}{'  [force]' if force else ''}...")

    path = snapshot_download(
        HF_MODEL,
        local_dir=local_dir,
        token=os.environ.get("HF_TOKEN"),
        force_download=force,
    )

    print(f"Downloaded to: {path}")
    models_volume.commit()
    return {"path": path}


@app.function(
    image=download_image,
    volumes={DATA_DIR: data_volume},
    timeout=3600,
)
def prepare_dataset(
    data_folder: str = "gsm8k",
    hf_dataset: str = "openai/gsm8k",
    split: str = "train",
    input_col: str = "question",
    output_col: str = "answer",
):
    """Download a HuggingFace dataset to the data volume.

    ms-swift expects the dataset to be in the format of:
    {
        "messages": [
            {
                "role": "user",
                "content": "Question?"},
            {
                "role": "assistant",
                "content": "Answer."
            }
        ]
    }
    """
    import json

    from datasets import load_dataset

    data_volume.reload()
    output_dir = f"{DATA_DIR}/{data_folder}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {hf_dataset} (split={split})...")

    # Handle datasets with configs (e.g., "openai/gsm8k" needs "main")
    try:
        ds = load_dataset(hf_dataset, split=split)
    except ValueError:
        ds = load_dataset(hf_dataset, "main", split=split)

    columns = ds.column_names
    print(f"  Columns: {columns}")

    if input_col not in columns:
        raise ValueError(
            f"Input column {input_col} not found in dataset columns: {columns}"
        )
    if output_col not in columns:
        raise ValueError(
            f"Output column {output_col} not found in dataset columns: {columns}"
        )

    print(f"  Using: input_col={input_col}, output_col={output_col}")

    all_examples = []
    for row in ds:
        messages = [
            {"role": "user", "content": row[input_col]},
            {"role": "assistant", "content": row[output_col]},
        ]
        all_examples.append({"messages": messages})

    output_path = f"{output_dir}/training.jsonl"
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} examples to {output_path}")
    data_volume.commit()
    return {"path": output_path, "count": len(all_examples)}


@app.function(
    image=msswift_v4_image,
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
    timeout=86400,  # 24 hours
    memory=1048576,  # 1TB
    ephemeral_disk=2048000,  # 2TB scratch
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train_model(
    run_id: str | None = None,
    data_folder: str = "gsm8k",
    merge_lora: bool = False,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    max_length: int = DEFAULT_MAX_LENGTH,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    global_batch_size: int = 8,
    lr: float = 1e-4,
    moe_aux_loss_coeff: float = 1e-3,
    recompute_num_layers: int = 1,
    save_interval: int = 50,
    eval_iters: int = 10,
    eval_interval: int = 50,
    disable_packing: bool = True,
):
    """Train GLM-4.7 via ms-swift v4 Megatron with LoRA."""
    import json
    import subprocess
    import time

    if run_id is None:
        run_id = f"train_glm_4_7_lora_{time.time()}"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else 1
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )

    if tp_size <= 0 or ep_size <= 0 or pp_size <= 0 or cp_size <= 0:
        raise ValueError(
            f"TP/EP/PP/CP must be positive, got TP={tp_size}, EP={ep_size}, PP={pp_size}, CP={cp_size}"
        )
    print(f"Node {node_rank}/{n_nodes}, Master: {master_addr}")
    print(f"Model: {HF_MODEL}")
    print(
        f"Parallelism: TP={tp_size}, EP={ep_size}, PP={pp_size}, CP={cp_size}, "
    )

    model_dir = f"{MODELS_DIR}/{MODEL_NAME}"
    if not os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")):
        raise RuntimeError(
            f"Model not found at {model_dir}. "
            f"Download first: modal run train_glm_4_7.py --download"
        )

    dataset_path = f"{DATA_DIR}/{data_folder}/training.jsonl"
    if not os.path.exists(dataset_path):
        raise RuntimeError(
            f"No training data found at {dataset_path}. "
            f"Upload data first: modal volume put glm-4-7-data /path/to/training.jsonl /{data_folder}/training.jsonl"
        )

    dataset = dataset_path
    print(f"Using local dataset: {dataset}")
        
    split_dataset_ratio = 0.01
    packing_enabled = not disable_packing

    checkpoint_dir = f"{CHECKPOINTS_DIR}/train_glm_4_7_{run_id}"

    # Pre-create args.json on ALL ranks so save_checkpoint can find it.
    # ms-swift writes args.json on is_master() (rank 0) but reads it on
    # is_last_rank() (rank 31) — different containers on Modal.
    # With --add_version false, args.save = checkpoint_dir (no versioned subdir).
    os.makedirs(checkpoint_dir, exist_ok=True)
    args_json_path = os.path.join(checkpoint_dir, "args.json")
    if not os.path.exists(args_json_path):
        with open(args_json_path, "w") as f:
            json.dump({"run_id": run_id, "placeholder": True}, f)

    # Set distributed env vars
    os.environ["NPROC_PER_NODE"] = "8"
    os.environ["NNODES"] = str(n_nodes)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"

    # Build megatron sft command
    # For the full set of parameters: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Megatron-SWIFT/Command-line-parameters.md
    cmd = [
        "megatron",
        "sft",
        "--model",
        model_dir,
        "--output_dir",
        checkpoint_dir,
        "--dataset",
        dataset,
        "--tuner_type",
        "lora",
        "--perform_initialization",
        "--split_dataset_ratio",
        str(split_dataset_ratio),
        "--tensor_model_parallel_size",
        str(tp_size),
        "--expert_model_parallel_size",
        str(ep_size),
        "--pipeline_model_parallel_size",
        str(pp_size),
        "--context_parallel_size",
        str(cp_size),
        "--sequence_parallel",
        "true",
        # MoE settings
        "--moe_permute_fusion",
        "true",
        "--moe_grouped_gemm",
        "true",
        "--moe_shared_expert_overlap",
        "true",
        "--moe_aux_loss_coeff",
        str(moe_aux_loss_coeff),
        # Batch
        "--global_batch_size",
        str(global_batch_size),
        "--packing",
        str(packing_enabled).lower(),
        # Memory optimization — recompute every layer (ms-swift example uses 1)
        "--recompute_granularity",
        "full",
        "--recompute_method",
        "uniform",
        "--recompute_num_layers",
        str(recompute_num_layers),
        "--use_precision_aware_optimizer",
        "true",
        # Training — epoch-based instead of iteration-based
        "--num_train_epochs",
        str(max_epochs),
        "--lr",
        str(lr),
        "--lr_warmup_fraction",
        "0.05",
        "--lr_decay_iters",
        "100000",  # Large default, capped by actual train steps
        "--min_lr",
        str(lr / 10),
        # Context
        "--max_length",
        str(max_length),
        "--attention_backend",
        "flash",
        # IO
        "--dataset_num_proc",
        "8",
        "--save_interval",
        str(save_interval),
        "--no_save_optim",
        "true",
        "--no_save_rng",
        "true",
        "--use_hf",
        "1",
        # Disable versioned subdirectory (v0-timestamp) in save path.
        # ms-swift writes args.json on is_master() (rank 0, node 0) but reads it
        # on is_last_rank() (rank 31, node 3) — cross-node volume sync race condition.
        "--add_version",
        "false",
        # Logging — report_to is required to actually enable WandB (v4 default is None)
        "--report_to",
        "wandb",
        "--wandb_project",
        WANDB_PROJECT,
        "--wandb_exp_name",
        run_id,
        "--log_interval",
        "1",
        "--eval_iters",
        str(eval_iters),
    ]
    if eval_iters > 0:
        cmd.extend(["--eval_interval", str(eval_interval)])

    # LoRA-specific args
    cmd.extend(
        [
            "--target_modules",
            "all-linear",
            "--lora_rank",
            str(lora_rank),
            "--lora_alpha",
            str(lora_alpha),
            "--merge_lora",
            str(merge_lora).lower(),
        ]
    )

    print(f"Running megatron command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ms-swift failed with code {result.returncode}")

    os.sync()
    checkpoints_volume.commit()
    print(
        f"Training {run_id} completed successfully with return code {result.returncode}"
    )
    print(f"Results saved to {checkpoint_dir}")
