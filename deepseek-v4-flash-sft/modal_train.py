# pyright: reportMissingImports=false
"""DeepSeek-V4-Flash SFT via ms-swift Megatron on Modal."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from typing import Any, cast

import modal
import modal.experimental

HF_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
MODEL_NAME = "DeepSeek-V4-Flash"
WANDB_PROJECT = "deepseek-v4-flash-sft"

DEFAULT_MAX_EPOCHS = 1
DEFAULT_MAX_LENGTH = 4096
DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 16

TP_SIZE = 1
PP_SIZE = 1
EP_SIZE = 8
CP_SIZE = 1
GPUS_PER_NODE = 8

MS_SWIFT_COMMIT = "5bbdfc5e5d458fda520b1b7cf4643dfa9e0bd348"
FAST_HADAMARD_TRANSFORM_COMMIT = "e7706faf8d1c3b9f241e36860640ad1dac644ede"
clustered = cast(Callable[..., Callable[..., Any]], modal.experimental.clustered)

app = modal.App("example-deepseek-v4-flash-sft")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name(
    "example-deepseek-v4-flash-sft-data",
    create_if_missing=True,
)
checkpoints_volume = modal.Volume.from_name(
    "example-deepseek-v4-flash-sft-checkpoints",
    create_if_missing=True,
    version=2,
)

HF_CACHE = "/root/.cache/huggingface"
DATA_DIR = "/data"
CHECKPOINTS_DIR = "/checkpoints"

# Evaluated at decoration time by @clustered.
N_NODES = int(os.environ.get("N_NODES", "1"))


def default_run_id(
    *,
    data_folder: str,
    lora_rank: int,
    lora_alpha: int,
    max_epochs: int,
    train_iters: int,
    max_length: int,
    tp_size: int,
    ep_size: int,
    pp_size: int,
    cp_size: int,
    global_batch_size: int,
    micro_batch_size: int,
    lr: float,
) -> str:
    run_config = {
        "cp_size": cp_size,
        "data_folder": data_folder,
        "ep_size": ep_size,
        "global_batch_size": global_batch_size,
        "lora_alpha": lora_alpha,
        "lora_rank": lora_rank,
        "lr": lr,
        "max_epochs": max_epochs,
        "max_length": max_length,
        "micro_batch_size": micro_batch_size,
        "model": HF_MODEL,
        "pp_size": pp_size,
        "tp_size": tp_size,
        "train_iters": train_iters,
    }
    digest = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()
    ).hexdigest()[:12]
    return f"deepseek_v4_flash_sft_{digest}"

DEEPSEEK_V4_CONFIG_PATCH = (
    r"""cat >>/usr/local/lib/python3.11/site-packages/transformers/models/auto/configuration_auto.py <<'PY'
try:
    try:
        from ...configuration_utils import PreTrainedConfig as _BaseConfig
    except ImportError:
        from ...configuration_utils import PretrainedConfig as _BaseConfig

    class DeepseekV4Config(_BaseConfig):
        model_type = "deepseek_v4"
        has_no_defaults_at_init = True
        keys_to_ignore_at_inference = ["past_key_values"]
        attribute_map = {
            "intermediate_size": "moe_intermediate_size",
            "num_local_experts": "n_routed_experts",
        }
        default_compress_rates = {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }
        default_num_hash_layers = 3
        default_partial_rotary_factor = 64 / 512

        def __init__(self, **kwargs):
            compress_ratios = kwargs.pop("compress_ratios", None)
            compress_rate_csa = kwargs.pop("compress_rate_csa", None)
            compress_rate_hca = kwargs.pop("compress_rate_hca", None)
            num_hash_layers = kwargs.pop("num_hash_layers", None)
            qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", None)
            super().__init__(**kwargs)

            n_layers = self.num_hidden_layers
            if getattr(self, "compress_rates", None) is None:
                self.compress_rates = dict(self.default_compress_rates)
            if compress_rate_csa is not None:
                self.compress_rates["compressed_sparse_attention"] = compress_rate_csa
            if compress_rate_hca is not None:
                self.compress_rates["heavily_compressed_attention"] = compress_rate_hca

            if getattr(self, "layer_types", None) is None and compress_ratios is not None:
                ratio_to_layer_type = {
                    0: "sliding_attention",
                    4: "compressed_sparse_attention",
                    128: "heavily_compressed_attention",
                }
                self.layer_types = [ratio_to_layer_type[r] for r in compress_ratios]
            if getattr(self, "layer_types", None) is None:
                interleave = [
                    "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                    for i in range(max(n_layers - 2, 0))
                ]
                self.layer_types = ["heavily_compressed_attention"] * min(n_layers, 2) + interleave
            self.layer_types = list(self.layer_types[:n_layers])

            if getattr(self, "mlp_layer_types", None) is None:
                n_hash = num_hash_layers if num_hash_layers is not None else self.default_num_hash_layers
                self.mlp_layer_types = ["hash_moe"] * min(n_layers, n_hash) + ["moe"] * max(
                    0, n_layers - n_hash
                )
            self.mlp_layer_types = list(self.mlp_layer_types[:n_layers])

            if getattr(self, "partial_rotary_factor", None) is None:
                self.partial_rotary_factor = (
                    qk_rope_head_dim / self.head_dim
                    if qk_rope_head_dim is not None
                    else self.default_partial_rotary_factor
                )
            self.qk_rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

    if "deepseek_v4" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("deepseek_v4", DeepseekV4Config)
except Exception:
    pass
PY"""
)
DEEPSEEK_V4_CONFIG_VERIFY = (
    "python -c \"from transformers.models.auto.configuration_auto import "
    "CONFIG_MAPPING; assert 'deepseek_v4' in CONFIG_MAPPING\""
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets==3.1.0",
        "huggingface_hub[hf_xet]==0.36.0",
        "safetensors==0.7.0",
        "sentencepiece==0.2.1",
        "torch==2.9.1",
        "transformers==4.57.4",
    )
    .run_commands(DEEPSEEK_V4_CONFIG_PATCH)
    .run_commands(DEEPSEEK_V4_CONFIG_VERIFY)
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

msswift_image = (
    modal.Image.from_registry(
        "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.3"
    )
    .apt_install(
        "build-essential",
        "git",
        "libhwloc-dev",
        "libibverbs-dev",
        "libibverbs1",
        "libnl-route-3-200",
        "ninja-build",
    )
    .run_commands("pip uninstall -y transformers ms-swift swift 2>/dev/null; true")
    .pip_install(
        "datasets==3.1.0",
        "einops==0.8.2",
        "huggingface_hub[hf_xet]==0.36.0",
        f"ms-swift @ git+https://github.com/modelscope/ms-swift.git@{MS_SWIFT_COMMIT}",
        "safetensors==0.7.0",
        "sentencepiece==0.2.1",
        "transformers==4.57.4",
        "wandb==0.19.1",
    )
    .run_commands("pip install --no-deps mcore-bridge==1.4.2")
    .run_commands(DEEPSEEK_V4_CONFIG_PATCH)
    .run_commands(DEEPSEEK_V4_CONFIG_VERIFY)
    .run_commands(
        "pip install --no-build-isolation "
        f"git+https://github.com/Dao-AILab/fast-hadamard-transform.git@{FAST_HADAMARD_TRANSFORM_COMMIT}"
    )
    .env(
        {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


@app.function(
    image=download_image,
    volumes={HF_CACHE: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,
)
def download_model(force: bool = False):
    from huggingface_hub import snapshot_download

    hf_cache_vol.reload()
    path = snapshot_download(
        HF_MODEL,
        force_download=force,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded {HF_MODEL} to {path}")
    hf_cache_vol.commit()
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
    max_examples: int | None = 4096,
):
    from datasets import load_dataset

    data_volume.reload()
    output_dir = f"{DATA_DIR}/{data_folder}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        ds = load_dataset(hf_dataset, split=split, trust_remote_code=True)
    except ValueError:
        ds = load_dataset(hf_dataset, "main", split=split, trust_remote_code=True)

    columns = ds.column_names
    if input_col not in columns:
        raise ValueError(f"{input_col=} not found in dataset columns: {columns}")
    if output_col not in columns:
        raise ValueError(f"{output_col=} not found in dataset columns: {columns}")

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    output_path = f"{output_dir}/training.jsonl"
    with open(output_path, "w") as f:
        for row in ds:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": row[input_col]},
                            {"role": "assistant", "content": row[output_col]},
                        ]
                    }
                )
                + "\n"
            )

    data_volume.commit()
    print(f"Wrote {len(ds)} examples to {output_path}")
    return {"path": output_path, "count": len(ds)}


@app.function(
    image=msswift_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def smoke_test():
    import shutil
    import subprocess

    from transformers import AutoConfig, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    config = AutoConfig.from_pretrained(HF_MODEL, token=token, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL,
        token=token,
        trust_remote_code=True,
        use_fast=True,
    )

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a one-line Modal training smoke test."}],
        tokenize=False,
        add_generation_prompt=True,
    )

    megatron_path = shutil.which("megatron")
    if megatron_path is None:
        raise RuntimeError("ms-swift Megatron CLI was not installed")

    subprocess.run(["megatron", "sft", "--help"], check=True)
    print(f"{MODEL_NAME}: model_type={config.model_type}")
    print(f"{MODEL_NAME}: layers={config.num_hidden_layers}")
    print(f"{MODEL_NAME}: experts={config.n_routed_experts}")
    print(f"{MODEL_NAME}: chat template chars={len(chat_prompt)}")
    return {
        "model_type": config.model_type,
        "num_hidden_layers": config.num_hidden_layers,
        "n_routed_experts": config.n_routed_experts,
        "megatron": megatron_path,
    }


@app.function(
    image=msswift_image,
    gpu="B200:8",
    volumes={
        HF_CACHE: hf_cache_vol,
        DATA_DIR: data_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret", required_keys=[]),
    ],
    timeout=86400,
    retries=1,
    memory=1048576,
    ephemeral_disk=2048000,
    experimental_options={"efa_enabled": True},
)
@clustered(size=N_NODES, rdma=True)
def train_model(
    run_id: str | None = None,
    data_folder: str = "gsm8k",
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    train_iters: int = 0,
    max_length: int = DEFAULT_MAX_LENGTH,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    lr: float = 1e-4,
    save_interval: int = 25,
    report_to: str = "none",
):
    import subprocess

    from huggingface_hub import snapshot_download

    cluster_info = modal.experimental.get_cluster_info()
    if run_id is None:
        run_id = default_run_id(
            data_folder=data_folder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            max_epochs=max_epochs,
            train_iters=train_iters,
            max_length=max_length,
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=pp_size,
            cp_size=cp_size,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            lr=lr,
        )

    node_rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else 1
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )

    model_parallel_size = tp_size * ep_size * pp_size * cp_size
    total_gpus = n_nodes * GPUS_PER_NODE
    if model_parallel_size <= 0:
        raise ValueError("Parallelism sizes must be positive")
    if total_gpus % model_parallel_size != 0:
        raise ValueError(
            f"TP×EP×PP×CP={model_parallel_size} must divide {total_gpus} total GPUs"
        )

    hf_cache_vol.reload()
    data_volume.reload()
    checkpoints_volume.reload()
    model_dir = snapshot_download(
        HF_MODEL,
        local_files_only=True,
        token=os.environ.get("HF_TOKEN"),
    )

    dataset_path = f"{DATA_DIR}/{data_folder}/training.jsonl"
    if not os.path.exists(dataset_path):
        raise RuntimeError(
            f"No dataset found at {dataset_path}; run prepare_dataset first"
        )

    checkpoint_dir = f"{CHECKPOINTS_DIR}/{run_id}"

    resuming = False
    if os.path.exists(checkpoint_dir):
        iter_dirs = sorted(
            d for d in os.listdir(checkpoint_dir) if d.startswith("iter_")
        )
        if iter_dirs:
            resuming = True
            print(f"Resuming from existing checkpoint ({iter_dirs[-1]})")

    os.makedirs(checkpoint_dir, exist_ok=True)
    args_json_path = f"{checkpoint_dir}/args.json"
    if not os.path.exists(args_json_path):
        with open(args_json_path, "w") as f:
            json.dump({"run_id": run_id}, f)

    megatron_cmd = [
        "megatron",
        "sft",
        "--model",
        model_dir,
        "--output_dir",
        checkpoint_dir,
        "--dataset",
        dataset_path,
        "--tuner_type",
        "lora",
        "--target_modules",
        "all-linear",
        "--lora_rank",
        str(lora_rank),
        "--lora_alpha",
        str(lora_alpha),
        "--perform_initialization",
        "--split_dataset_ratio",
        "0.01",
        "--tensor_model_parallel_size",
        str(tp_size),
        "--expert_model_parallel_size",
        str(ep_size),
        "--pipeline_model_parallel_size",
        str(pp_size),
        "--context_parallel_size",
        str(cp_size),
        "--sequence_parallel",
        "false",
        "--moe_permute_fusion",
        "true",
        "--moe_grouped_gemm",
        "true",
        "--moe_shared_expert_overlap",
        "true",
        "--megatron_extra_kwargs",
        json.dumps({"moe_router_score_function": "sigmoid"}),
        "--global_batch_size",
        str(global_batch_size),
        "--micro_batch_size",
        str(micro_batch_size),
        "--packing",
        "false",
        "--use_precision_aware_optimizer",
        "true",
        "--lr",
        str(lr),
        "--lr_warmup_fraction",
        "0.05",
        "--lr_decay_iters",
        "100000",
        "--min_lr",
        str(lr / 10),
        "--max_length",
        str(max_length),
        "--attention_backend",
        "flash",
        "--dataset_num_proc",
        "8",
        "--save_steps",
        str(save_interval),
        "--no_save_optim",
        "true",
        "--no_save_rng",
        "true",
        "--use_hf",
        "1",
        "--add_version",
        "false",
        "--logging_steps",
        "1",
        "--eval_iters",
        "0",
    ]
    if train_iters > 0:
        megatron_cmd.extend(["--train_iters", str(train_iters)])
    else:
        megatron_cmd.extend(["--num_train_epochs", str(max_epochs)])

    if report_to == "wandb":
        if "WANDB_API_KEY" not in os.environ:
            raise RuntimeError("WANDB_API_KEY must be set to use report_to='wandb'")
        megatron_cmd.extend(
            [
                "--report_to",
                "wandb",
                "--wandb_project",
                WANDB_PROJECT,
                "--wandb_exp_name",
                run_id,
            ]
        )
    elif report_to != "none":
        raise ValueError("report_to must be either 'none' or 'wandb'")

    if resuming:
        megatron_cmd.extend(["--load", checkpoint_dir])

    cmd = [
        "torchrun",
        "--no_python",
        "--nproc_per_node",
        str(GPUS_PER_NODE),
        "--nnodes",
        str(n_nodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        "29500",
        *megatron_cmd,
    ]

    print(f"Node {node_rank}/{n_nodes}, master={master_addr}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ms-swift failed with code {result.returncode}")

    checkpoints_volume.commit()
    print(f"Saved checkpoints to {checkpoint_dir}")
    return {"checkpoint_dir": checkpoint_dir, "run_id": run_id}
