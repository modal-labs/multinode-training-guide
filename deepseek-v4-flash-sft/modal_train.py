# pyright: reportMissingImports=false, reportCallIssue=false, reportOptionalCall=false
"""DeepSeek-V4-Flash 60k LoRA SFT and vLLM serving on Modal."""

from __future__ import annotations

import json
import os
from pathlib import Path
from string import Template

import modal
import modal.experimental

from dsv4_images import (
    REMOTE_CHECKPOINT_HELPER,
    REMOTE_TRAIN_RECIPE,
    build_automodel_image,
    build_vllm_image,
)

HERE = Path(__file__).parent

HF_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
HF_CACHE = "/root/.cache/huggingface"
CHECKPOINTS_DIR = "/checkpoints"

GPUS_PER_NODE = 8
N_NODES = int(os.environ.get("N_NODES", "8"))
HOST_MEMORY = (128, 256 * 1024)
SEQ_LENGTH = 60_000
MAX_STEPS = 5
CP_SIZE = 16
PP_SIZE = 4
MAX_EP_SIZE = 32
GLOBAL_BATCH_SIZE = 8
LOCAL_BATCH_SIZE = 4
LORA_RANK = 64
LORA_ALPHA = 64
SYNTHETIC_EXAMPLES = 40
DIST_BACKEND = "cpu:gloo,cuda:nccl"

VLLM_ADAPTER_NAME = "deepseek-v4-flash-60k-lora"
VLLM_PORT = 8000
VLLM_PIPELINE_PARALLEL_SIZE = 4
VLLM_HF_CONFIG_DIR = Path("/tmp/deepseek-v4-flash-vllm-config")
SERVE_RUN_ID = os.environ.get("SERVE_RUN_ID")
SERVE_MAX_MODEL_LEN = 64 * 1024

app = modal.App("example-deepseek-v4-flash-sft")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name(
    "example-deepseek-v4-flash-sft-checkpoints",
    create_if_missing=True,
    version=2,
)

automodel_image = build_automodel_image(HERE)
vllm_image = build_vllm_image(HERE)


def _validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path component")


def _training_topology(n_nodes: int) -> tuple[int, dict[str, int]]:
    if n_nodes < 1:
        raise ValueError("N_NODES must be positive")

    total_gpus = n_nodes * GPUS_PER_NODE
    if total_gpus % (PP_SIZE * CP_SIZE):
        raise ValueError(
            f"{total_gpus} GPUs cannot be divided across PP={PP_SIZE} and CP={CP_SIZE}"
        )

    non_pp_size = total_gpus // PP_SIZE
    ep_size = min(MAX_EP_SIZE, non_pp_size)
    if non_pp_size % ep_size:
        raise ValueError(f"{non_pp_size=} must be divisible by {ep_size=}")

    return total_gpus, {
        "tp": 1,
        "dp": total_gpus // (PP_SIZE * CP_SIZE),
        "pp": PP_SIZE,
        "cp": CP_SIZE,
        "ep": ep_size,
    }


def _select_uccl_ep_variant() -> str:
    """Activate the UCCL extension for the scheduled RDMA provider."""
    import importlib.util
    import shutil

    vendors = {
        path.read_text().strip()
        for path in Path("/sys/class/infiniband").glob("*/device/vendor")
    }
    if "0x1d0f" in vendors:
        provider = "efa"
    elif "0x15b3" in vendors:
        provider = "mellanox"
    else:
        raise RuntimeError(f"Unsupported RDMA vendors: {sorted(vendors)}")

    uccl_spec = importlib.util.find_spec("uccl")
    if uccl_spec is None or not uccl_spec.submodule_search_locations:
        raise RuntimeError("UCCL is missing from the training image")
    package_dir = Path(next(iter(uccl_spec.submodule_search_locations)))
    source = package_dir / f"ep.{provider}.abi3.so"
    if not source.is_file():
        raise RuntimeError(f"Missing UCCL-EP {provider} extension: {source}")

    target = package_dir / "ep.abi3.so"
    shutil.copy2(source, target)
    print(f"selected_uccl_ep_provider={provider}")
    return provider


def _write_synthetic_dataset(path: Path) -> None:
    """Write fixed-length supervised examples to local storage on each node."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = " ".join(f"fact_{idx % 4096}" for idx in range(SEQ_LENGTH * 2))
    with path.open("w") as output:
        for index in range(SYNTHETIC_EXAMPLES):
            row = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Repeat the synthetic facts for example {index}.",
                    },
                    {
                        "role": "assistant",
                        "content": (f"Synthetic facts for example {index}: {payload}"),
                    },
                ]
            }
            output.write(json.dumps(row) + "\n")


def _render_recipe(
    *,
    dataset_path: Path,
    checkpoint_dir: Path,
    ep_size: int,
) -> str:
    return Template(Path(REMOTE_TRAIN_RECIPE).read_text()).substitute(
        CHECKPOINT_DIR=json.dumps(str(checkpoint_dir)),
        CP_SIZE=CP_SIZE,
        DATASET_PATH=json.dumps(str(dataset_path)),
        DIST_BACKEND=json.dumps(DIST_BACKEND),
        EP_SIZE=ep_size,
        GLOBAL_BATCH_SIZE=GLOBAL_BATCH_SIZE,
        HF_MODEL=json.dumps(HF_MODEL),
        LOCAL_BATCH_SIZE=LOCAL_BATCH_SIZE,
        LORA_ALPHA=LORA_ALPHA,
        LORA_RANK=LORA_RANK,
        MAX_STEPS=MAX_STEPS,
        PP_SIZE=PP_SIZE,
        SEQ_LENGTH=SEQ_LENGTH,
    )


@app.function(
    image=automodel_image,
    gpu="H200:8",
    volumes={HF_CACHE: hf_cache_vol, CHECKPOINTS_DIR: checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
    retries=0,
    memory=HOST_MEMORY,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train_cluster(run_id: str, expected_nodes: int):
    import subprocess

    _validate_run_id(run_id)
    cluster = modal.experimental.get_cluster_info()
    node_rank = cluster.rank
    n_nodes = len(cluster.container_ips) if cluster.container_ips else 1
    if n_nodes != expected_nodes:
        raise RuntimeError(f"Expected {expected_nodes} nodes, scheduled {n_nodes}")

    total_gpus, topology = _training_topology(n_nodes)
    master_addr = cluster.container_ips[0] if cluster.container_ips else "localhost"
    os.environ.update(
        {
            "GLOO_SOCKET_IFNAME": "eth1",
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "NCCL_DEBUG": "WARN",
            "NCCL_MAX_NCHANNELS": "8",
            "NCCL_SOCKET_FAMILY": "AF_INET6",
            "NCCL_SOCKET_IFNAME": "eth1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "UCCL_EP_CPU_TIMEOUT_SECS": "600",
            "UCCL_SOCKET_FAMILY": "AF_INET6",
            "UCCL_SOCKET_IFNAME": "eth1",
        }
    )

    hf_cache_vol.reload()
    checkpoints_vol.reload()
    uccl_provider = _select_uccl_ep_variant()

    dataset_path = Path("/tmp/dsv4_60k_synthetic/training.jsonl")
    _write_synthetic_dataset(dataset_path)

    checkpoint_dir = Path(CHECKPOINTS_DIR) / run_id
    if checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint directory already exists: {checkpoint_dir}")
    recipe_path = Path("/tmp/dsv4_train.yaml")
    recipe_path.write_text(
        _render_recipe(
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            ep_size=topology["ep"],
        )
    )

    summary = {
        "run_id": run_id,
        "nodes": n_nodes,
        "total_gpus": total_gpus,
        "topology": topology,
        "sequence_length": SEQ_LENGTH,
        "optimizer_steps": MAX_STEPS,
        "checkpoint_dir": str(checkpoint_dir),
        "uccl_provider": uccl_provider,
    }
    if node_rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True))

    command = [
        "torchrun",
        "--nproc-per-node",
        str(GPUS_PER_NODE),
        "--nnodes",
        str(n_nodes),
        "--node-rank",
        str(node_rank),
        "--master-addr",
        master_addr,
        "--master-port",
        "29501",
        "-m",
        "nemo_automodel.cli.app",
        str(recipe_path),
    ]
    subprocess.run(command, check=True)
    checkpoints_vol.commit()
    return summary


@app.function(
    image=automodel_image,
    volumes={HF_CACHE: hf_cache_vol, CHECKPOINTS_DIR: checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    memory=HOST_MEMORY,
)
def finalize(run_id: str):
    """Merge the four pipeline-stage shards into a serving adapter."""
    import sys

    sys.path.insert(0, str(Path(REMOTE_CHECKPOINT_HELPER).parent))
    from dsv4_checkpoint import finalize_adapter

    checkpoints_vol.reload()
    manifest = finalize_adapter(CHECKPOINTS_DIR, run_id)
    checkpoints_vol.commit()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _download_vllm_config() -> None:
    from huggingface_hub import hf_hub_download

    config_path = Path(
        hf_hub_download(
            repo_id=HF_MODEL,
            filename="config.json",
            cache_dir="/tmp/vllm-hf-config-cache",
            local_dir=VLLM_HF_CONFIG_DIR,
            force_download=True,
            token=os.environ.get("HF_TOKEN"),
        )
    )
    config = json.loads(config_path.read_text())
    expected_quantization = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    }
    if config.get("quantization_config") != expected_quantization:
        raise RuntimeError(
            "Unexpected DeepSeek-V4-Flash quantization configuration: "
            f"{config.get('quantization_config')}"
        )


def _vllm_command(adapter_dir: Path) -> list[str]:
    return [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        HF_MODEL,
        "--served-model-name",
        HF_MODEL,
        "--pipeline-parallel-size",
        str(VLLM_PIPELINE_PARALLEL_SIZE),
        "--enable-expert-parallel",
        "--trust-remote-code",
        "--hf-config-path",
        str(VLLM_HF_CONFIG_DIR),
        "--tokenizer-mode",
        "deepseek_v4",
        "--reasoning-parser",
        "deepseek_v4",
        "--max-model-len",
        str(SERVE_MAX_MODEL_LEN),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.92",
        "--kv-cache-dtype",
        "fp8",
        "--block-size",
        "256",
        "--enable-lora",
        "--max-loras",
        "1",
        "--max-lora-rank",
        str(LORA_RANK),
        "--lora-target-modules",
        "fused_wqa_wkv",
        "wq_b",
        "--lora-modules",
        f"{VLLM_ADAPTER_NAME}={adapter_dir}",
        "--enforce-eager",
        "--no-enable-flashinfer-autotune",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--port",
        str(VLLM_PORT),
    ]


@app.function(
    image=vllm_image,
    gpu=f"H200:{VLLM_PIPELINE_PARALLEL_SIZE}",
    volumes={HF_CACHE: hf_cache_vol, CHECKPOINTS_DIR: checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
    memory=HOST_MEMORY,
    experimental_options={"efa_enabled": True},
)
@modal.web_server(
    VLLM_PORT,
    startup_timeout=3600,
    requires_proxy_auth=True,
)
def serve():
    """Serve the finalized adapter selected by SERVE_RUN_ID."""
    import subprocess
    import sys

    if not SERVE_RUN_ID:
        raise RuntimeError("Set SERVE_RUN_ID to a finalized checkpoint run ID")

    sys.path.insert(0, str(Path(REMOTE_CHECKPOINT_HELPER).parent))
    from dsv4_checkpoint import finalized_adapter

    hf_cache_vol.reload()
    checkpoints_vol.reload()
    _download_vllm_config()
    adapter_dir, manifest = finalized_adapter(CHECKPOINTS_DIR, SERVE_RUN_ID)
    print(f"adapter_sha256={manifest['adapter_sha256']}")
    subprocess.Popen(_vllm_command(adapter_dir))


@app.local_entrypoint()
def train(run_id: str):
    """Start a detached multi-node training call."""
    _validate_run_id(run_id)
    _, topology = _training_topology(N_NODES)
    call = train_cluster.spawn(run_id=run_id, expected_nodes=N_NODES)
    call_id = (
        getattr(call, "object_id", None)
        or getattr(call, "function_call_id", None)
        or str(call)
    )
    print(
        json.dumps(
            {
                "function_call_id": call_id,
                "run_id": run_id,
                "nodes": N_NODES,
                "topology": topology,
            },
            indent=2,
        )
    )
