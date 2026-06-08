# pyright: reportMissingImports=false
"""DeepSeek-V4-Flash SFT via ms-swift Megatron on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import PurePosixPath
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
MEGATRON_CORE_COMMIT = "cefc2520158c7ceba3f9adbe4b547a6f7a118da1"
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


DEEPSEEK_V4_CONFIG_PATCH = r"""cat >>/usr/local/lib/python3.11/site-packages/transformers/models/auto/configuration_auto.py <<'PY'
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

            n_layers = getattr(self, "num_hidden_layers", 0)
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
                    qk_rope_head_dim / getattr(self, "head_dim", 1)
                    if qk_rope_head_dim is not None
                    else self.default_partial_rotary_factor
                )
            self.qk_rope_head_dim = int(getattr(self, "head_dim", 1) * self.partial_rotary_factor)

    if "deepseek_v4" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("deepseek_v4", DeepseekV4Config)
except Exception:
    pass
PY"""
DEEPSEEK_V4_MODELING_PATCH = r"""cat >>/usr/local/lib/python3.11/site-packages/transformers/models/auto/modeling_auto.py <<'PY'
try:
    from ...modeling_utils import PreTrainedModel as _PreTrainedModel
    from ...models.auto.configuration_auto import CONFIG_MAPPING

    class DeepseekV4ForCausalLM(_PreTrainedModel):
        config_class = CONFIG_MAPPING["deepseek_v4"]
        base_model_prefix = "model"
        _no_split_modules = []

        def __init__(self, config):
            super().__init__(config)

        def forward(self, *args, **kwargs):
            raise NotImplementedError("DeepSeek-V4 export only uses this as a meta model")

    MODEL_FOR_CAUSAL_LM_MAPPING.register(CONFIG_MAPPING["deepseek_v4"], DeepseekV4ForCausalLM)
except Exception:
    pass
PY"""
DEEPSEEK_V4_CONFIG_VERIFY = (
    "python - <<'PY'\n"
    "from transformers.models.auto.configuration_auto import CONFIG_MAPPING\n"
    "from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING\n"
    "config_cls = CONFIG_MAPPING['deepseek_v4']\n"
    "assert MODEL_FOR_CAUSAL_LM_MAPPING[config_cls].__name__ == 'DeepseekV4ForCausalLM'\n"
    "PY"
)
MCORE_BRIDGE_DSV4_PATCH = r"""python - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/config/model_config.py")
text = path.read_text()
rope_fields_old = (
    "    original_max_position_embeddings: Optional[int] = None\n"
    "    partial_rotary_factor: Optional[float] = None\n"
)
rope_fields_new = (
    "    original_max_position_embeddings: int = 4096\n"
    "    rotary_scaling_factor: float = 40\n"
    "    beta_fast: float = 32\n"
    "    beta_slow: float = 1\n"
    "    mscale: float = 1.0\n"
    "    mscale_all_dim: float = 0.0\n"
    "    partial_rotary_factor: Optional[float] = None\n"
)
if rope_fields_old not in text:
    raise RuntimeError("mcore_bridge rope field patch target not found")
text = text.replace(rope_fields_old, rope_fields_new)

rope_scaling_old = (
    "            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:\n"
    "                self.rope_scaling['rope_type'] = self.rope_scaling['type']\n"
)
rope_scaling_new = (
    "            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:\n"
    "                self.rope_scaling['rope_type'] = self.rope_scaling['type']\n"
    "            if 'factor' in self.rope_scaling:\n"
    "                self.rotary_scaling_factor = self.rope_scaling['factor']\n"
    "            if 'original_max_position_embeddings' in self.rope_scaling:\n"
    "                self.original_max_position_embeddings = self.rope_scaling['original_max_position_embeddings']\n"
    "            if 'beta_fast' in self.rope_scaling:\n"
    "                self.beta_fast = self.rope_scaling['beta_fast']\n"
    "            if 'beta_slow' in self.rope_scaling:\n"
    "                self.beta_slow = self.rope_scaling['beta_slow']\n"
    "            if 'mscale' in self.rope_scaling:\n"
    "                self.mscale = self.rope_scaling['mscale']\n"
    "            if 'mscale_all_dim' in self.rope_scaling:\n"
    "                self.mscale_all_dim = self.rope_scaling['mscale_all_dim']\n"
    "            if self.llm_model_type == 'deepseek_v4' and 'main' not in self.rope_scaling:\n"
    "                self.rope_scaling = {\n"
    "                    'main': dict(self.rope_scaling),\n"
    "                    'compress': dict(self.rope_scaling),\n"
    "                }\n"
)
if rope_scaling_old not in text:
    raise RuntimeError("mcore_bridge rope scaling patch target not found")
text = text.replace(rope_scaling_old, rope_scaling_new)
path.write_text(text)
PY"""
MCORE_BRIDGE_DSV4_VERIFY = r"""python - <<'PY'
from pathlib import Path

text = Path("/usr/local/lib/python3.11/site-packages/mcore_bridge/config/model_config.py").read_text()
assert "rotary_scaling_factor: float = 40" in text
assert "if self.llm_model_type == 'deepseek_v4' and 'main' not in self.rope_scaling" in text
PY"""

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
    .run_commands(DEEPSEEK_V4_MODELING_PATCH)
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
    .run_commands(MCORE_BRIDGE_DSV4_PATCH)
    .run_commands(MCORE_BRIDGE_DSV4_VERIFY)
    .run_commands(
        "pip install --no-deps "
        f"'megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@{MEGATRON_CORE_COMMIT}'"
    )
    .run_commands(DEEPSEEK_V4_CONFIG_PATCH)
    .run_commands(DEEPSEEK_V4_MODELING_PATCH)
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

vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.22.1")
    .run_commands("ln -sf $(which python3) /usr/local/bin/python")
    .run_commands("python -m pip install datasets==3.1.0")
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py')\n"
        "text = path.read_text()\n"
        "old = '        self.scale_fmt = config.quantization_config[\"scale_fmt\"]\\n'\n"
        'new = \'        self.scale_fmt = getattr(config, "quantization_config", {"scale_fmt": "ue8m0"})["scale_fmt"]\\n\'\n'
        "if old not in text:\n"
        "    raise RuntimeError('DeepSeek V4 vLLM scale_fmt patch target not found')\n"
        "path.write_text(text.replace(old, new))\n"
        "PY"
    )
    .run_commands(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/attention.py')\n"
        "text = path.read_text()\n"
        "old = '        if current_platform.is_rocm():\\n'\n"
        "new = '        if current_platform.is_rocm() or not hasattr(self.wo_a, \"weight_scale_inv\"):\\n'\n"
        "if old not in text:\n"
        "    raise RuntimeError('DeepSeek V4 vLLM BF16 attention patch target not found')\n"
        "path.write_text(text.replace(old, new, 1))\n"
        "PY"
    )
    .entrypoint([])
    .env({"VLLM_USE_V1": "1"})
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


TRAINING_VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    HF_CACHE: hf_cache_vol,
    DATA_DIR: data_volume,
    CHECKPOINTS_DIR: checkpoints_volume,
}


def _train_model_impl(
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

        def checkpoint_sort_key(name: str) -> int:
            match = re.search(r"(?:checkpoint-|iter_)(\d+)", name)
            return int(match.group(1)) if match else -1

        checkpoint_dirs = sorted(
            (
                d
                for d in os.listdir(checkpoint_dir)
                if d.startswith("checkpoint-") or d.startswith("iter_")
            ),
            key=checkpoint_sort_key,
        )
        if checkpoint_dirs:
            resuming = True
            print(f"Resuming from existing checkpoint ({checkpoint_dirs[-1]})")

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
        "--recompute_modules",
        "mhc",
        "--megatron_extra_kwargs",
        json.dumps({"moe_router_score_function": "sigmoid"}),
        "--global_batch_size",
        str(global_batch_size),
        "--micro_batch_size",
        str(micro_batch_size),
        "--packing",
        "false",
        "--padding_free",
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
        "--save_safetensors",
        "false",
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


@app.function(
    image=msswift_image,
    gpu="B200:8",
    volumes=TRAINING_VOLUMES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
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
):
    return _train_model_impl(
        run_id=run_id,
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
        save_interval=save_interval,
        report_to="none",
    )


@app.function(
    image=msswift_image,
    gpu="B200:8",
    volumes=TRAINING_VOLUMES,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=86400,
    retries=1,
    memory=1048576,
    ephemeral_disk=2048000,
    experimental_options={"efa_enabled": True},
)
@clustered(size=N_NODES, rdma=True)
def train_model_wandb(
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
):
    return _train_model_impl(
        run_id=run_id,
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
        save_interval=save_interval,
        report_to="wandb",
    )


def _extract_gsm8k_answer(text: str) -> str | None:
    """Extract the numerical answer from a GSM8K-style response.

    Looks for ``#### <number>`` first, then falls back to the last number.
    """
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def _numeric_eq(left: str, right: str) -> bool:
    try:
        return float(left) == float(right)
    except ValueError:
        return left == right


def _make_config_vllm_compatible(config_path: str) -> None:
    with open(config_path) as f:
        config = json.load(f)

    config["architectures"] = ["DeepseekV4ForCausalLM"]
    config.setdefault("pad_token_id", config.get("eos_token_id", 1))
    if "n_routed_experts" in config:
        config.setdefault("num_local_experts", config["n_routed_experts"])
    if "moe_intermediate_size" in config:
        config.setdefault("intermediate_size", config["moe_intermediate_size"])
    vllm_defaults = {
        "compress_rope_theta": 160000,
        "num_nextn_predict_layers": 1,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "mlp_bias": False,
        "output_router_logits": False,
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.0,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "swiglu_limit": 10.0,
    }
    for key, value in vllm_defaults.items():
        config.setdefault(key, value)
    if config.get("dtype") == "bfloat16":
        config.pop("expert_dtype", None)
        config.pop("quantization_config", None)
    else:
        config.setdefault(
            "quantization_config",
            {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
            },
        )
    if isinstance(config.get("mlp_layer_types"), list):
        config["mlp_layer_types"] = [
            "moe" if layer_type == "hash_moe" else layer_type
            for layer_type in config["mlp_layer_types"]
        ]
    if not isinstance(config.get("compress_ratios"), list) and isinstance(
        config.get("layer_types"), list
    ):
        layer_type_to_ratio = {
            "sliding_attention": 0,
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }
        config["compress_ratios"] = [
            layer_type_to_ratio[layer_type] for layer_type in config["layer_types"]
        ]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


@app.function(
    image=msswift_image,
    gpu="B200:8",
    volumes=TRAINING_VOLUMES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
    retries=1,
    memory=1048576,
    ephemeral_disk=2048000,
)
def export_checkpoint(
    run_id: str,
    checkpoint_step: int = 5,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
):
    """Export a Megatron LoRA checkpoint into merged HF safetensors."""
    import subprocess

    from huggingface_hub import snapshot_download

    hf_cache_vol.reload()
    checkpoints_volume.reload()

    model_dir = snapshot_download(
        HF_MODEL,
        local_files_only=True,
        token=os.environ.get("HF_TOKEN"),
    )

    ckpt_dir = f"{CHECKPOINTS_DIR}/{run_id}/checkpoint-{checkpoint_step}"
    merged_dir = f"{CHECKPOINTS_DIR}/{run_id}/merged-hf"
    if not os.path.exists(ckpt_dir):
        raise RuntimeError(f"Checkpoint not found at {ckpt_dir}")

    os.environ["NPROC_PER_NODE"] = str(GPUS_PER_NODE)
    export_cmd = [
        "megatron",
        "export",
        "--model",
        model_dir,
        "--mcore_adapter",
        ckpt_dir,
        "--merge_lora",
        "true",
        "--to_hf",
        "true",
        "--output_dir",
        merged_dir,
        "--tensor_model_parallel_size",
        str(tp_size),
        "--expert_model_parallel_size",
        str(ep_size),
        "--pipeline_model_parallel_size",
        str(pp_size),
        "--context_parallel_size",
        str(cp_size),
        "--exist_ok",
        "true",
    ]

    print(f"[export] {' '.join(export_cmd)}")
    result = subprocess.run(export_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"megatron export failed (exit {result.returncode})")

    _make_config_vllm_compatible(f"{merged_dir}/config.json")

    checkpoints_volume.commit()
    print(f"[export] Merged HF model saved to {merged_dir}")
    return {"merged_dir": merged_dir, "run_id": run_id}


@app.function(
    image=vllm_image,
    gpu="B200:8",
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
    timeout=86400,
    retries=1,
    memory=1048576,
    ephemeral_disk=2048000,
)
def deploy_and_eval_merged(
    run_id: str,
    eval_limit: int = 20,
    max_model_len: int = 4096,
):
    """Deploy the exported model with vLLM, run smoke inference, and eval GSM8K."""
    import subprocess
    import time
    import urllib.request

    from datasets import load_dataset

    checkpoints_volume.reload()
    merged_dir = f"{CHECKPOINTS_DIR}/{run_id}/merged-hf"
    config_path = f"{merged_dir}/config.json"
    if not os.path.exists(config_path):
        raise RuntimeError(f"Merged HF model not found at {merged_dir}")
    _make_config_vllm_compatible(config_path)
    checkpoints_volume.commit()

    server_log_path = "/tmp/vllm-server.log"
    server_log = open(server_log_path, "w")
    server_proc = subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            merged_dir,
            "--served-model-name",
            "deepseek-v4-flash-sft",
            "--tensor-parallel-size",
            str(GPUS_PER_NODE),
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            "0.9",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "fp8",
            "--moe-backend",
            "triton",
            "--port",
            "8000",
            "--no-enable-log-requests",
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("[deploy] Waiting for vLLM server...")
    for attempt in range(240):
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            print(f"[deploy] Server ready after ~{attempt * 5}s")
            break
        except Exception:
            if server_proc.poll() is not None:
                server_log.flush()
                with open(server_log_path) as f:
                    tail = f.read()[-20000:]
                server_log.close()
                raise RuntimeError(f"vLLM server exited during startup:\n{tail}")
            time.sleep(5)
    else:
        server_proc.kill()
        server_proc.wait(timeout=30)
        server_log.flush()
        with open(server_log_path) as f:
            tail = f.read()[-20000:]
        server_log.close()
        raise RuntimeError(f"vLLM server failed to start:\n{tail}")

    try:

        def _chat(prompt: str, max_tokens: int = 512) -> str:
            body = json.dumps(
                {
                    "model": "deepseek-v4-flash-sft",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                }
            ).encode()
            req = urllib.request.Request(
                "http://localhost:8000/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=120)
            return json.loads(resp.read())["choices"][0]["message"]["content"]

        smoke_prompts = [
            "What is 2+2? Give just the number.",
            "If a train travels 60 mph for 2.5 hours, how far does it go?",
            "A store sells apples for $0.50 each. How much do 12 cost?",
        ]
        print("\n=== Smoke Inference ===")
        for prompt in smoke_prompts:
            ans = _chat(prompt, max_tokens=256)
            print(f"Q: {prompt}\nA: {ans[:400]}\n")

        print(f"\n=== GSM8K Evaluation (limit={eval_limit}) ===")
        gsm8k = load_dataset("openai/gsm8k", "main", split="test")
        if eval_limit:
            gsm8k = gsm8k.select(range(min(eval_limit, len(gsm8k))))

        correct = 0
        total = 0
        for i, row in enumerate(gsm8k):
            gold = _extract_gsm8k_answer(row["answer"])
            if gold is None:
                continue
            response = _chat(row["question"], max_tokens=512)
            pred = _extract_gsm8k_answer(response)
            hit = pred is not None and _numeric_eq(pred, gold)
            correct += hit
            total += 1
            tag = "PASS" if hit else "FAIL"
            print(f"  [{i + 1}/{len(gsm8k)}] {tag}  pred={pred}  gold={gold}")

        accuracy = correct / total if total > 0 else 0.0
        print(f"\n=== GSM8K Result: {correct}/{total} ({accuracy:.1%}) ===")

        return {
            "run_id": run_id,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
    finally:
        if server_proc.poll() is None:
            server_proc.kill()
        server_proc.wait(timeout=30)
        server_log.close()


@app.local_entrypoint()
def export_and_eval(
    run_id: str,
    checkpoint_step: int = 5,
    eval_limit: int = 20,
    max_model_len: int = 4096,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
):
    """Export a checkpoint, deploy the merged model, and run a small eval."""
    export_checkpoint.remote(
        run_id=run_id,
        checkpoint_step=checkpoint_step,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        cp_size=cp_size,
    )
    result = deploy_and_eval_merged.remote(
        run_id=run_id,
        eval_limit=eval_limit,
        max_model_len=max_model_len,
    )
    print(json.dumps(result, indent=2))
    return result
