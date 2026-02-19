"""
Qwen3-Coder-Next (80B MoE) Training via ms-swift Megatron v4

Separate script from msswift.py because:
  - Requires ms-swift v4 (qwen3_next model support, added 2026-01-14)
  - Different image (ms-swift upgrade + einops + newer transformers)
  - Doesn't risk breaking existing GLM/Qwen235B training

Defaults to LoRA. Use --train-type full for full SFT.
ms-swift handles HF→Megatron conversion internally (no separate convert step needed).

Usage:
    # Download model (once)
    modal run msswift_coder_next.py --download

    # LoRA (default), 2 epochs, 2 nodes — parallelism auto-resolved from constants
    modal run --detach msswift_coder_next.py --train --data-folder gsm8k

    # Full SFT at 128K — parallelism auto-resolved: TP=2, EP=8, PP=2 (32 GPUs)
    N_NODES=4 modal run --detach msswift_coder_next.py --train \
        --data-folder 2026-02-14 --cached-dataset --train-type full

    # Pre-tokenized cached dataset
    MODEL=qwen-coder modal run preprocess/msswift.py --data-folder 2026-02-14 --upload ./data.jsonl
    modal run --detach msswift_coder_next.py --train --data-folder 2026-02-14 --cached-dataset

Pipeline: Step 2 of 5 (preprocess -> train -> convert -> serve -> eval)

Parallelism notes:
    - Megatron has TWO DP values: non-expert DP and expert DP
    - Non-expert DP = world_size / (TP * PP) — used for batch size divisibility check
    - Expert DP = world_size / (TP * EP * PP) — how expert layers are replicated
    - global_batch_size must be divisible by micro_batch_size * non-expert-DP
    - Full SFT at 128K: TP=2, EP=8, PP=2 → 32 GPUs → N_NODES=4 (auto-resolved from constants)
    - TP max is 2 (model has only 2 KV heads)
"""

import os
import sys

import modal
import modal.experimental

HF_MODEL = "Qwen/Qwen3-Coder-Next"

COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common")

app = modal.App("msswift-more-like-mscursed")

# Volumes — all V2
models_volume = modal.Volume.from_name("qwen-coder-next-models-v2", create_if_missing=True, version=2)
data_volume = modal.Volume.from_name("qwen-coder-next-data-v2", create_if_missing=True, version=2)
checkpoints_volume = modal.Volume.from_name("qwen-coder-next-checkpoints-v2", create_if_missing=True, version=2)

MODELS_DIR = "/models"
DATA_DIR = "/data"
CHECKPOINTS_DIR = "/checkpoints"

# N_NODES via env var (evaluated at decoration time by @clustered).
# Default 2 (LoRA). Full SFT at 128K needs 4 (TP=2 × EP=4 × PP=4 = 32 GPUs).
# Usage: N_NODES=4 modal run --detach msswift_coder_next.py --train ...
N_NODES = int(os.environ.get("N_NODES", "2"))

# Download image (lightweight)
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==0.27.1",
        "transformers>=4.50,<5.0",
        "torch==2.5.1",
        "safetensors==0.4.5",
        "datasets==3.4.0",
    )
    .env({
        "HF_HOME": "/models/huggingface",
    })
)

# NeMo 25.11 base + ms-swift v4 + DeltaNet kernels
# NeMo provides: PyTorch 2.8, CUDA 13.0, FA, MC ≥0.15, nvshmem, RDMA/EFA
# We add: ms-swift v4, transformers upgrade, DeltaNet kernels (fla), DeepEP
msswift_v4_image = (
    modal.Image.from_registry("nvcr.io/nvidia/nemo:25.11")
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc-dev",
        "libnl-route-3-200",
    )
    .pip_install(
        "transformers==4.57.3",
        "peft>=0.15.0",  # NeMo 25.11 ships 0.13.2, ms-swift v4 requires >=0.15
        "ms-swift @ git+https://github.com/modelscope/ms-swift.git@d2a67bf8c",
        "einops",
        "wandb==0.19.1",
        # DeltaNet optimized kernels — 5-10x faster than torch fallback for 75% of layers
        "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@v0.4.1",
        # NOT installing causal-conv1d — transformers falls back to PyTorch ops (F.silu + conv1d)
    )
    .run_commands(
        # Fix missing 'warnings' import bug in Megatron-Core
        "sed -i '1s/^/import warnings\\n/' /opt/megatron-lm/megatron/core/model_parallel_config.py",
        # Fix Manager().Queue() crash in Modal
        r"""sed -i 's/_results_queue = ctx.Manager().Queue()/_results_queue = ctx.Queue()/' /opt/megatron-lm/megatron/core/dist_checkpointing/strategies/filesystem_async.py""",
        # Disable fully_parallel_save by default - Modal multiprocessing is limited
        r"""sed -i 's/fully_parallel_save: bool = True/fully_parallel_save: bool = False/' /opt/Megatron-Bridge/src/megatron/bridge/training/config.py""",
    )
    # DeepEP: fused MoE token dispatcher (available but not activated — Coder-Next uses overlap)
    .run_commands(
        # CUDA 13.0 moved CCCL headers under include/cccl/ — DeepEP expects them at top level
        "CUDA_INC=/usr/local/cuda/targets/x86_64-linux/include && "
        "if [ -d \"$CUDA_INC/cccl/cuda\" ]; then "
        "mkdir -p $CUDA_INC/cuda && "
        "for item in $CUDA_INC/cccl/cuda/*; do "
        "name=$(basename \"$item\") && "
        "[ ! -e \"$CUDA_INC/cuda/$name\" ] && ln -sfn \"$item\" \"$CUDA_INC/cuda/$name\"; "
        "done; fi && "
        "if [ -d \"$CUDA_INC/cccl/nv\" ] && [ ! -e \"$CUDA_INC/nv\" ]; then "
        "ln -sfn $CUDA_INC/cccl/nv $CUDA_INC/nv; fi",
        # nvshmem symlink (nvshmem already in NeMo 25.11, just needs the unversioned .so)
        'NVSHMEM_LIB=$(python3 -c "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0] + \'/lib\')") && '
        "ln -sf $NVSHMEM_LIB/libnvshmem_host.so.3 $NVSHMEM_LIB/libnvshmem_host.so",
    )
    .run_commands(
        # Build DeepEP for B200 (sm_100) + H200 (sm_90)
        "git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP && "
        "cd /tmp/DeepEP && git checkout eb9cee7",
        # Force arch list at top of setup.py — NeMo 25.11 may set TORCH_CUDA_ARCH_LIST broadly
        "cd /tmp/DeepEP && "
        "sed -i '1s/^/import os; os.environ[\"TORCH_CUDA_ARCH_LIST\"] = \"9.0 10.0\"\\n/' setup.py && "
        "python setup.py build_ext --inplace > /tmp/deepep_build.log 2>&1 || "
        "{ echo '=== DeepEP BUILD FAILED ===' && tail -80 /tmp/deepep_build.log && exit 1; }",
        "cd /tmp/DeepEP && pip install --no-build-isolation --no-deps . && rm -rf /tmp/DeepEP /tmp/deepep_build.log",
    )
    .env({
        "HF_HOME": "/models/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_FWD_LAYERNORM_SM_MARGIN": "16",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "16",
        # Disable torch.compile — inductor materializes the full logits tensor
        # (122K tokens × 152K vocab × fp32 = 69 GB) instead of using fused CE loss
        "TORCHDYNAMO_DISABLE": "1",
        # NCCL debug — surface real error behind "unhandled cuda error"
        "NCCL_DEBUG": "WARN",
    })
    .add_local_dir(COMMON_DIR, remote_path="/root/common/")
)


@app.function(
    image=download_image,
    volumes={MODELS_DIR: models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,  # 4 hours
)
def download_model(force: bool = False):
    """Download Qwen3-Coder-Next (80B MoE) to volume.

    Uses local_dir (flat files) instead of cache_dir (symlinked blobs)
    because Modal V2 volumes may not handle HF cache symlinks correctly.
    """
    from huggingface_hub import snapshot_download

    models_volume.reload()

    local_dir = f"{MODELS_DIR}/Qwen/Qwen3-Coder-Next"
    print(f"Downloading {HF_MODEL} (80B MoE) to {local_dir}{'  [force]' if force else ''}...")

    path = snapshot_download(
        HF_MODEL,
        local_dir=local_dir,
        token=os.environ.get("HF_TOKEN"),
        force_download=force,
    )

    # Verify index file integrity
    import json
    index_path = os.path.join(path, "model.safetensors.index.json")
    with open(index_path) as f:
        data = json.load(f)
    print(f"Index file OK: {len(data.get('weight_map', {})):,} weight entries")

    print(f"Downloaded to: {path}")
    models_volume.commit()
    return {"path": path}

@app.function(
    image=download_image,
    volumes={DATA_DIR: data_volume},
    timeout=3600,
)
def prepare_dataset(
    data_folder: str = "aime",
    hf_dataset: str = "AI-MO/aimo-validation-aime",
    split: str = "train",
    input_col: str = "",
    output_col: str = "",
):
    """Download a HuggingFace dataset to the data volume.

    Auto-detects input/output columns from common patterns:
      problem/solution, question/answer, input/output, prompt/response

    Args:
        data_folder: Folder name in volume
        hf_dataset: HuggingFace dataset path (e.g., "AI-MO/aimo-validation-aime", "openai/gsm8k")
        split: Dataset split to use
        input_col: Override input column name
        output_col: Override output column name
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

    # Auto-detect input/output columns
    input_patterns = ["problem", "question", "input", "prompt", "instruction"]
    output_patterns = ["solution", "answer", "output", "response", "completion"]

    if not input_col:
        for pat in input_patterns:
            if pat in columns:
                input_col = pat
                break
    if not output_col:
        for pat in output_patterns:
            if pat in columns:
                output_col = pat
                break

    if not input_col or not output_col:
        raise ValueError(
            f"Could not auto-detect columns. Found: {columns}. "
            f"Specify --input-col and --output-col explicitly."
        )

    print(f"  Using: input_col={input_col}, output_col={output_col}")

    all_examples = []
    skipped = 0
    for row in ds:
        input_text = row[input_col]
        output_text = row[output_col]
        
        # Skip empty examples
        if not input_text or not output_text:
            skipped += 1
            continue
        
        # Ensure strings (some datasets have nested structures)
        if not isinstance(input_text, str):
            input_text = str(input_text)
        if not isinstance(output_text, str):
            output_text = str(output_text)
            
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        all_examples.append({"messages": messages})

    if not all_examples:
        raise ValueError(f"No valid examples found! Skipped {skipped} empty rows.")

    output_path = f"{output_dir}/training.jsonl"
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} examples to {output_path}")
    if skipped:
        print(f"  (skipped {skipped} empty rows)")
    
    # Print sample for verification
    print(f"\nSample row:")
    print(f"  User: {all_examples[0]['messages'][0]['content'][:100]}...")
    print(f"  Assistant: {all_examples[0]['messages'][1]['content'][:100]}...")
    
    data_volume.commit()
    return {"path": output_path, "count": len(all_examples), "skipped": skipped}

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
    timeout=86400,      # 24 hours
    memory=1048576,     # 1TB
    ephemeral_disk=2048000,  # 2TB scratch
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train_model(
    run_id: str,
    data_folder: str = "gsm8k",
    dataset: str = "",
    cached_dataset: bool = False,
    train_type: str = "lora",
    merge_lora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    max_epochs: int = 2,
    max_length: int = 8192,
    ep_size: int = 0,
    tp_size: int = 0,
    pp_size: int = 0,
    global_batch_size: int = 25,
    lr: float = 1e-4,
    moe_aux_loss_coeff: float = 1e-3,
    recompute_num_layers: int = 1,
    save_interval: int = 50,
    eval_iters: int = 10,
    eval_interval: int = 50,
):
    """Train Qwen3-Coder-Next via ms-swift v4 Megatron. Defaults to LoRA."""
    import subprocess
    sys.path.insert(0, "/root/common")
    from constants import MODEL_CONFIGS

    # Resolve parallelism defaults from constants based on train_type
    model_cfg = MODEL_CONFIGS["qwen3-coder-next"]
    parallelism_key = "train_parallelism_full" if train_type == "full" else "train_parallelism"
    parallelism = model_cfg[parallelism_key]
    if tp_size == 0:
        tp_size = parallelism["tp"]
    if ep_size == 0:
        ep_size = parallelism["ep"]
    if pp_size == 0:
        pp_size = parallelism["pp"]
    if global_batch_size == 0:
        # gbs must be divisible by mbs * non_expert_dp
        total_gpus = int(os.environ.get("N_NODES", "2")) * 8
        non_expert_dp = total_gpus // (tp_size * pp_size)
        global_batch_size = non_expert_dp  # mbs=1, so gbs=DP is the minimum valid value

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else 1
    master_addr = cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"

    total_gpus = n_nodes * 8
    non_expert_dp = total_gpus // (tp_size * pp_size)
    print(f"Node {node_rank}/{n_nodes}, Master: {master_addr}")
    print(f"Model: {HF_MODEL}")
    print(f"Parallelism: TP={tp_size}, EP={ep_size}, PP={pp_size}, non-expert DP={non_expert_dp}")

    # Reload volumes
    models_volume.reload()
    data_volume.reload()
    checkpoints_volume.reload()

    # Use local model path (flat files, no HF cache symlinks)
    model_dir = f"{MODELS_DIR}/Qwen/Qwen3-Coder-Next"
    if not os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")):
        raise RuntimeError(
            f"Model not found at {model_dir}. "
            f"Download first: modal run msswift_coder_next.py --download"
        )

    split_dataset_ratio = 0.01
    train_iters = None

    if not dataset:
        dataset_path = f"{DATA_DIR}/{data_folder}/training.jsonl"
        if os.path.exists(dataset_path):
            dataset = dataset_path
            print(f"Using local dataset: {dataset}")
        else:
            raise RuntimeError(
                f"No training data found at {dataset_path}. "
                f"Upload data first: modal volume put qwen-coder-next-data-v2 /path/to/training.jsonl /{data_folder}/training.jsonl"
            )
    else:
        print(f"Using specified dataset: {dataset}")

    if not cached_dataset and dataset.endswith(".jsonl") and os.path.exists(dataset):
        total_rows = 0
        with open(dataset) as f:
            for line in f:
                if line.strip():
                    total_rows += 1
        if total_rows == 0:
            raise RuntimeError(f"Dataset is empty: {dataset}")
        val_rows = max(1, int(total_rows * split_dataset_ratio))
        train_samples = max(1, total_rows - val_rows)
        steps_per_epoch = max(1, (train_samples + global_batch_size - 1) // global_batch_size)
        train_iters = max(1, steps_per_epoch * max_epochs)
        print(
            f"Dataset preflight: rows={total_rows}, val_rows~={val_rows}, "
            f"train_samples={train_samples}, steps_per_epoch={steps_per_epoch}, train_iters={train_iters}"
        )

    checkpoint_dir = f"{CHECKPOINTS_DIR}/msswift_coder_next_{run_id}"

    # Pre-create args.json on ALL ranks so save_checkpoint can find it.
    # ms-swift writes args.json on is_master() (rank 0) but reads it on
    # is_last_rank() (rank 31) — different containers on Modal.
    # With --add_version false, args.save = checkpoint_dir (no versioned subdir).
    import json as _json
    os.makedirs(checkpoint_dir, exist_ok=True)
    args_json_path = os.path.join(checkpoint_dir, "args.json")
    if not os.path.exists(args_json_path):
        with open(args_json_path, "w") as f:
            _json.dump({"run_id": run_id, "placeholder": True}, f)

    # Set distributed env vars
    os.environ["NPROC_PER_NODE"] = "8"
    os.environ["NNODES"] = str(n_nodes)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WANDB_PROJECT"] = "coder-next-train-msswift"
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "allow"
    if train_iters is not None:
        os.environ["MEGATRON_TRAIN_ITERS_FALLBACK"] = str(train_iters)

    # Build megatron sft command — epoch-based
    cmd = [
        "megatron", "sft",
        "--model", model_dir,
        "--save", checkpoint_dir
    ]
    cmd.extend(["--dataset", dataset])
    cmd.extend([
        "--load_safetensors", "true",
        "--save_safetensors", "true",
        "--train_type", train_type,
        "--no_initialization", "false",
        "--split_dataset_ratio", str(split_dataset_ratio),
        # Passthrough template: bypass ms-swift template processing
        # patch_wandb_artifacts: disable Megatron's artifact upload (path mismatch with ms-swift)
        # "--external_plugins", "/root/common/passthrough_template.py", "/root/common/patch_wandb_artifacts.py",
        # "--template", "passthrough",
        "--external_plugins", "/root/common/patch_wandb_artifacts.py", "/root/common/patch_ms_swift_n_steps.py",

        # Parallelism — LoRA: EP=4, TP=1. Full SFT 128K: TP=2, EP=4, PP=4.
        "--expert_model_parallel_size", str(ep_size),
        "--tensor_model_parallel_size", str(tp_size),
        "--pipeline_model_parallel_size", str(pp_size),
        "--sequence_parallel", "true",
        # MoE settings
        "--moe_permute_fusion", "true",
        "--moe_grouped_gemm", "true",

        # TODO: JOY LIU FLAGS
        "--moe_token_dispatcher_type", "flex",
        "--moe_enable_deepep",
        # END TODO: JOY LIU FLAGS
        
        "--moe_aux_loss_coeff", str(moe_aux_loss_coeff),
        # Batch
        "--micro_batch_size", "1",
        "--global_batch_size", str(global_batch_size),
        "--packing", "false",
        # Memory optimization — recompute every layer (ms-swift example uses 1)
        "--recompute_granularity", "full",
        "--recompute_method", "uniform",
        "--recompute_num_layers", str(recompute_num_layers),
        "--optimizer_cpu_offload", "true",
        "--use_precision_aware_optimizer", "true",
        # Training — epoch-based instead of iteration-based
        # Training
        "--finetune", "true",
        "--cross_entropy_loss_fusion", "true",
        "--lr", str(lr),
        "--lr_warmup_fraction", "0.1",
        "--min_lr", str(lr / 10),
        # Context
        "--max_length", str(max_length),
        "--attention_backend", "flash",
        # IO
        "--num_workers", "8",
        "--dataset_num_proc", "8",
        "--save_interval", str(save_interval),
        "--no_save_optim", "true",
        "--no_save_rng", "true",
        "--use_hf", "1",
        # Disable versioned subdirectory (v0-timestamp) in save path.
        # ms-swift writes args.json on is_master() (rank 0, node 0) but reads it
        # on is_last_rank() (rank 31, node 3) — cross-node volume sync race condition.
        "--add_version", "false",
        # Logging — report_to is required to actually enable WandB (v4 default is None)
        "--report_to", "wandb",
        "--wandb_project", "coder-next-train-msswift",
        "--wandb_exp_name", run_id,
        "--log_interval", "1",

    ])

    if train_iters is not None:
        cmd.extend(["--train_iters", str(train_iters)])

    # Eval settings
    cmd.extend(["--eval_iters", str(eval_iters)])
    if eval_iters > 0:
        cmd.extend(["--eval_interval", str(eval_interval)])

    # LoRA-specific args
    if train_type == "lora":
        cmd.extend([
            "--target_modules", "all-linear",
            "--lora_rank", str(lora_rank),
            "--lora_alpha", str(lora_alpha),
            "--merge_lora", str(merge_lora).lower(),
        ])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ms-swift training failed with code {result.returncode}")

    os.sync()
    checkpoints_volume.commit()
    return {"status": "training_complete", "run_id": run_id, "checkpoint": checkpoint_dir}


@app.local_entrypoint()
def main(
    download: bool = False,
    train: bool = False,
    data_folder: str = "",
    dataset: str = "",
    cached_dataset: bool = False,
    train_type: str = "lora",
    merge_lora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    max_epochs: int = 2,
    max_length: int = 131072,
    ep_size: int = 0,
    tp_size: int = 0,
    pp_size: int = 0,
    global_batch_size: int = 0,
    lr: float = 1e-4,
    moe_aux_loss_coeff: float = 1e-3,
    recompute_num_layers: int = 1,
    save_interval: int = 50,
    eval_iters: int = 10,
    eval_interval: int = 50,
):
    """Qwen3-Coder-Next training via ms-swift v4 Megatron.

    Defaults to LoRA. Parallelism auto-resolved from constants.py based on train_type.
    LoRA: TP=1, EP=4, PP=1 (16 GPUs). Full SFT: TP=2, EP=8, PP=2 (32 GPUs).
    CLI overrides take priority when non-zero.
    """
    if download:
        print(f"Downloading {HF_MODEL} (80B MoE)...")
        result = download_model.remote(force=True)
        print(f"Done: {result}")
    elif train:
        if not data_folder:
            print("ERROR: --data-folder is required when using --train")
            return

        # Resolve parallelism defaults from constants based on train_type
        from common.constants import MODEL_CONFIGS
        model_cfg = MODEL_CONFIGS["qwen3-coder-next"]
        parallelism_key = "train_parallelism_full" if train_type == "full" else "train_parallelism"
        parallelism = model_cfg[parallelism_key]
        if tp_size == 0:
            tp_size = parallelism["tp"]
        if ep_size == 0:
            ep_size = parallelism["ep"]
        if pp_size == 0:
            pp_size = parallelism["pp"]
        if global_batch_size == 0:
            total_gpus = N_NODES * 8
            non_expert_dp = total_gpus // (tp_size * pp_size)
            global_batch_size = non_expert_dp

        total_gpus = N_NODES * 8
        non_expert_dp = total_gpus // (tp_size * pp_size)
        print(f"Starting Qwen3-Coder-Next training ({train_type}):")
        print(f"  model={HF_MODEL}")
        print(f"  nodes={N_NODES} ({total_gpus} GPUs)")
        print(f"  data_folder={data_folder}")
        print(f"  dataset={dataset or '(auto-detect)'}")
        print(f"  cached_dataset={cached_dataset}")
        print(f"  train_type={train_type}")
        if train_type == "lora":
            print(f"  lora_rank={lora_rank}, lora_alpha={lora_alpha}")
            print(f"  merge_lora={merge_lora}")
        print(f"  max_epochs={max_epochs}")
        print(f"  max_length={max_length}")
        print(f"  ep_size={ep_size}, tp_size={tp_size}, pp_size={pp_size}, non_expert_dp={non_expert_dp}")
        print(f"  global_batch_size={global_batch_size}")
        print(f"  recompute_num_layers={recompute_num_layers}")

        import wandb
        generated_run_id = wandb.util.generate_id()
        print(f"WandB run_id: {generated_run_id}")

        result = train_model.remote(
            run_id=generated_run_id,
            data_folder=data_folder,
            dataset=dataset,
            cached_dataset=cached_dataset,
            train_type=train_type,
            merge_lora=merge_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            max_epochs=max_epochs,
            max_length=max_length,
            ep_size=ep_size,
            tp_size=tp_size,
            pp_size=pp_size,
            global_batch_size=global_batch_size,
            lr=lr,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            recompute_num_layers=recompute_num_layers,
            save_interval=save_interval,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
        )
        print(f"Done: {result}")
    else:
        print(f"Qwen3-Coder-Next Training ({HF_MODEL})")
        print()
        print("Usage:")
        print("  modal run msswift_coder_next.py --download")
        print("  modal run --detach msswift_coder_next.py --train --data-folder 2026-02-14")
        print("  N_NODES=4 modal run --detach msswift_coder_next.py --train \\")
        print("      --data-folder 2026-02-14 --cached-dataset --train-type full --tp-size 2 --ep-size 4 --pp-size 4")
        print()
        print(f"Current: N_NODES={N_NODES} ({N_NODES * 8} GPUs)")
        print("Defaults: LoRA, EP=4, TP=1, PP=1, recompute_num_layers=1")
        print()
        print("Parallelism (set N_NODES env var to change cluster size):")
        print("  N_NODES=2  16 GPUs — LoRA (EP=4, TP=1, PP=1)")
        print("  N_NODES=4  32 GPUs — Full SFT 128K (TP=2, EP=4, PP=4)")
        print()
        print("Non-expert DP = total_gpus / (TP * PP). global_batch_size must be divisible by it.")
        print("TP max is 2 (model has only 2 KV heads). PP=2 halves per-GPU model memory.")