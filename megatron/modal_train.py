"""
GLM-4.7 LoRA Training on Modal

Usage:
    modal run train_glm47.py --download        # Download model (4+ hours)
    modal run train_glm47.py --convert         # Convert HF to Megatron
    modal run preprocess_dataset.py            # Build dataset indexes (single process)
    modal run --detach train_glm47.py --train  # Run distributed training
"""

import modal
import modal.experimental

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

    # Training script using pre-built dataset
    train_script = f'''
import os
import torch
import wandb
from functools import wraps
from megatron.core.transformer.transformer_block import TransformerBlock

# Run ID generated before script launch - same for all ranks
run_id = "{run_id}"
rank = int(os.environ.get("RANK", 0))

# Only rank 0 initializes WandB
if rank == 0:
    wandb.init(project="glm47-lora", id=run_id, resume="allow")
    print(f"WandB initialized with run_id: {{run_id}}")

print(f"[Rank {{rank}}] Using run_id: {{run_id}}")

from megatron.bridge import AutoBridge
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.peft.lora import LoRA

print("=" * 60)
print(f"GLM-4.7 (358B MoE) - Run: {{run_id}}")
print("=" * 60)

# Monkey-patch for PP=1 + Recompute + Frozen Base gradient fix
_original_transformer_forward = TransformerBlock.forward

@wraps(_original_transformer_forward)
def _patched_transformer_forward(self, hidden_states, *args, **kwargs):
    if (
        torch.is_tensor(hidden_states)
        and not hidden_states.requires_grad
        and hidden_states.is_floating_point()
    ):
        hidden_states = hidden_states.detach().requires_grad_(True)
    return _original_transformer_forward(self, hidden_states, *args, **kwargs)

TransformerBlock.forward = _patched_transformer_forward
print("[PEFT+Recompute FIX] Patched TransformerBlock.forward")

# LoRA config
lora_config = LoRA(
    dim=128,
    alpha=32,
    dropout=0.05,
)

# GLM-4.7 uses same architecture class (Glm4MoeForCausalLM) as GLM-4.5
# So we can use from_hf_pretrained directly - it will read the correct architecture
print(f"Creating config from HF model: zai-org/GLM-4.7")

# Use AutoBridge to get the model provider from HF config
# This reads the actual architecture (92 layers, 160 experts) from HF
bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-4.7", trust_remote_code=True)
model_cfg = bridge.to_megatron_provider(load_weights=False)

print(f"Model loaded: num_layers={{model_cfg.num_layers}}, num_moe_experts={{getattr(model_cfg, 'num_moe_experts', 'N/A')}}")

# Import the config classes we need
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    FinetuningDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)

import os
rank = int(os.environ.get("RANK", 0))
print(f"[Rank {{rank}}] Loading pre-built dataset from: {PREPROCESSED_DIR}")

# Optimizer with cosine annealing
opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
    lr_warmup_iters=50,
    lr_decay_iters=None,
    max_lr=1e-4,
    min_lr=0.0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_eps=1e-8,
    weight_decay=0.1,
)

# Build the full config
config = ConfigContainer(
    model=model_cfg,
    train=TrainingConfig(
        train_iters=650,
        eval_interval=9999,
        eval_iters=0,
        global_batch_size=16,
        micro_batch_size=1,
    ),
    optimizer=opt_cfg,
    scheduler=scheduler_cfg,
    ddp=DistributedDataParallelConfig(check_for_nan_in_grad=True),
    dataset=FinetuningDatasetConfig(
        dataset_root="{PREPROCESSED_DIR}",  # Pre-built data location
        seq_length=131072,  # 128k context
        seed=5678,
        dataloader_type="batch",
        num_workers=1,
        do_validation=False,
        do_test=False,
    ),
    logger=LoggerConfig(
        log_interval=1,
        tensorboard_dir="/tmp/tensorboard",
        wandb_project="glm47-lora",
        wandb_exp_name=run_id,  # Use WandB run_id
    ),
    tokenizer=TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model="zai-org/GLM-4.7",
    ),
    checkpoint=CheckpointConfig(
        save_interval=130,
        save=f"{CHECKPOINTS_DIR}/glm47_lora_" + run_id,  # Use WandB run_id
        pretrained_checkpoint="{MEGATRON_CHECKPOINT}",
        ckpt_format="torch_dist",
        fully_parallel_save=False,  # Must be False - Modal multiprocessing is limited
        async_save=False,  # Disable async save - Modal queue issues
    ),
    rng=RNGConfig(seed=5678),
    peft=lora_config,
    mixed_precision="bf16_mixed",
)

print(f"Config created successfully")

# Parallelism for GLM-4.7 on 32 GPUs (TP=2 x PP=1 x EP=8 x CP=2 = 32)
config.model.tensor_model_parallel_size = 2
config.model.pipeline_model_parallel_size = 1
config.model.expert_model_parallel_size = 8
config.model.context_parallel_size = 2  # 64k per GPU
# config.model.cp_comm_type = "a2a+p2p"
config.model.calculate_per_token_loss = True  # Required for CP>1
config.model.sequence_parallel = True
config.model.attention_backend = "flash"

# MoE optimization
config.model.moe_grouped_gemm = True

# DDP optimization
# config.ddp.overlap_param_gather = True

# Memory optimization - activation recomputation
config.model.recompute_granularity = "full"
config.model.recompute_method = "uniform"
config.model.recompute_num_layers = 1  # Must be 1 for MTP (Multi-Token Prediction)
# Sequence length
config.model.seq_length = 131072  # 128k context

print("Config:")
print(f"  Model: GLM-4.7 (358B MoE)")
print(f"  seq_length: {{config.model.seq_length}}")
print(f"  Dataset: {PREPROCESSED_DIR} (PRE-BUILT)")
print(f"  TP={{config.model.tensor_model_parallel_size}}, PP={{config.model.pipeline_model_parallel_size}}, EP={{config.model.expert_model_parallel_size}}, CP={{config.model.context_parallel_size}}")
print(f"  moe_grouped_gemm: {{config.model.moe_grouped_gemm}}")
print(f"  overlap_param_gather: {{config.ddp.overlap_param_gather}}")

finetune(config=config, forward_step_func=forward_step)
print("Training complete!")
'''

    script_path = "/tmp/train_glm47_lora.py"
    with open(script_path, "w") as f:
        f.write(train_script)

    cmd = [
        "torchrun",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--nproc_per_node=8",
        script_path,
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