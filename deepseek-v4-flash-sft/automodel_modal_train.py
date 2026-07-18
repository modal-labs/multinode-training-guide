# pyright: reportMissingImports=false, reportCallIssue=false, reportOptionalCall=false
"""DeepSeek-V4-Flash 60k LoRA SFT and vLLM serving on Modal."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal
import modal.experimental

HF_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
VLLM_IMAGE = "vllm/vllm-openai:v0.25.1"
VLLM_ADAPTER_NAME = "deepseek-v4-flash-60k-lora"
VLLM_PORT = 8000
VLLM_PIPELINE_PARALLEL_SIZE = 4
VLLM_GPU = f"H200:{VLLM_PIPELINE_PARALLEL_SIZE}"
VLLM_HF_CONFIG_DIR = Path("/tmp/deepseek-v4-flash-vllm-config")
GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
GSM8K_EVAL_INDICES = tuple(range(0, 600, 50))

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
MODEL_LAYERS = 43
LORA_TARGET_MODULES = (
    "*.self_attn.wq_a",
    "*.self_attn.wq_b",
    "*.self_attn.wkv",
)
UCCL_SOCKET_IFNAME = "eth1"
UCCL_SOCKET_FAMILY = "AF_INET6"
DIST_BACKEND = "cpu:gloo,cuda:nccl"
UCCL_CUDA_ALLOC_CONF = "expandable_segments:False"
NCCL_MAX_NCHANNELS = "8"
UCCL_EP_CPU_TIMEOUT_SECS = "600"

HF_CACHE = "/root/.cache/huggingface"
CHECKPOINTS_DIR = "/checkpoints"

# Includes DSv4 model-owned CP support and the TileLang dependency fixes.
AUTOMODEL_COMMIT = "1197b6281255957cc2c58e79d50796c1b256c57c"
AUTOMODEL_IMAGE = "nvcr.io/nvidia/nemo-automodel:26.06.00"
# v0.1.1's EFA wheel predates USE_DMABUF support and requires the host's
# efa_nv_peermem module. This commit includes the public DMA-BUF build path.
UCCL_EP_COMMIT = "66170bc299205228f0170bc1638594a39af9ffd5"
UCCL_EFA_INSTALLER_VERSION = "1.42.0"
UCCL_IPV6_PATCH = Path(__file__).with_name("uccl_ipv6_oob.patch")
AUTOMODEL_UCCL_TEARDOWN_PATCH = Path(__file__).with_name(
    "automodel_uccl_teardown.patch"
)
AUTOMODEL_CHECKPOINT_DEQUANT_PATCH = Path(__file__).with_name(
    "automodel_checkpoint_dequant.patch"
)
AUTOMODEL_COMPOSITE_BACKEND_PATCH = Path(__file__).with_name(
    "automodel_composite_backend.patch"
)
AUTOMODEL_PP_PEFT_CHECKPOINT_PATCH = Path(__file__).with_name(
    "automodel_pp_peft_checkpoint.patch"
)
VLLM_LORA_PATCH = Path(__file__).with_name("vllm_deepseek_v4_lora.patch")
SERVE_RUN_ID = os.environ.get("SERVE_RUN_ID")
SERVE_CHECKPOINT_STEP = MAX_STEPS - 1
SERVE_MAX_MODEL_LEN = 64 * 1024

app = modal.App("example-deepseek-v4-flash-automodel")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name(
    "example-deepseek-v4-flash-sft-checkpoints",
    create_if_missing=True,
    version=2,
)

automodel_image = (
    modal.Image.from_registry(AUTOMODEL_IMAGE)
    .entrypoint([])
    .run_commands(
        "cd / && rm -rf /opt/Automodel && "
        "git clone https://github.com/NVIDIA-NeMo/Automodel.git /opt/Automodel && "
        f"cd /opt/Automodel && git checkout {AUTOMODEL_COMMIT} && "
        "pip install --no-deps --force-reinstall -e /opt/Automodel"
    )
    .run_commands(
        "pip install --no-deps --force-reinstall "
        "'tilelang>=0.1.11' 'tile-kernels==1.0.0' 'apache-tvm-ffi<=0.1.11'"
    )
    .env(
        {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "NEMO_AUTOMODEL_DSV4_EXPERT_LAYOUT": "fp4",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        }
    )
)

automodel_image = (
    automodel_image.apt_install(
        "build-essential",
        "curl",
        "libibverbs-dev",
        "libnl-3-dev",
        "libnl-route-3-dev",
        "libnuma-dev",
        "ninja-build",
        "patch",
        "rdma-core",
    )
    .uv_pip_install(
        "intervaltree==3.1.0",
        "nanobind==2.13.0",
        "pybind11==3.0.1",
    )
    .run_commands(
        "cd /tmp && "
        f"curl -fsSLO https://efa-installer.amazonaws.com/aws-efa-installer-{UCCL_EFA_INSTALLER_VERSION}.tar.gz && "
        f"tar -xzf aws-efa-installer-{UCCL_EFA_INSTALLER_VERSION}.tar.gz && "
        "cd aws-efa-installer && "
        "./efa_installer.sh -y --skip-kmod -g --no-verify && "
        "rm -rf /tmp/aws-efa-installer*"
    )
    .add_local_file(str(UCCL_IPV6_PATCH), "/tmp/uccl_ipv6_oob.patch", copy=True)
    .run_commands(
        "rm -rf /opt/uccl && mkdir -p /opt/uccl && "
        f"curl -fsSL https://github.com/uccl-project/uccl/archive/{UCCL_EP_COMMIT}.tar.gz "
        "| tar -xz --strip-components=1 -C /opt/uccl && "
        "cd /opt/uccl && patch -p1 < /tmp/uccl_ipv6_oob.patch",
        "cd /opt/uccl/ep && rm -rf build ep*.so && "
        "EFA_HOME=/opt/amazon/efa USE_DMABUF=1 "
        "TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 "
        "/opt/venv/bin/python setup.py build_ext --inplace && "
        "cp ep*.so /opt/uccl/uccl/ep.efa.abi3.so",
        "cd /opt/uccl/ep && rm -rf build ep*.so && "
        "EFA_HOME=/opt/uccl/no-efa USE_DMABUF=1 "
        "TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 "
        "/opt/venv/bin/python setup.py build_ext --inplace && "
        "cp ep*.so /opt/uccl/uccl/ep.mellanox.abi3.so && "
        "cp ep*.so /opt/uccl/uccl/ep.abi3.so",
        "cd /opt/uccl && "
        "uv pip install --python /opt/venv/bin/python --no-build-isolation "
        "--no-deps --force-reinstall .",
    )
)

automodel_image = (
    automodel_image.add_local_file(
        str(AUTOMODEL_UCCL_TEARDOWN_PATCH),
        "/tmp/automodel_uccl_teardown.patch",
        copy=True,
    )
    .add_local_file(
        str(AUTOMODEL_CHECKPOINT_DEQUANT_PATCH),
        "/tmp/automodel_checkpoint_dequant.patch",
        copy=True,
    )
    .add_local_file(
        str(AUTOMODEL_COMPOSITE_BACKEND_PATCH),
        "/tmp/automodel_composite_backend.patch",
        copy=True,
    )
    .add_local_file(
        str(AUTOMODEL_PP_PEFT_CHECKPOINT_PATCH),
        "/tmp/automodel_pp_peft_checkpoint.patch",
        copy=True,
    )
    .run_commands(
        "cd /opt/Automodel && git apply /tmp/automodel_uccl_teardown.patch"
        " && git apply /tmp/automodel_checkpoint_dequant.patch"
        " && git apply /tmp/automodel_composite_backend.patch"
        " && git apply /tmp/automodel_pp_peft_checkpoint.patch"
    )
)

vllm_image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    .run_commands("ln -sf $(which python3) /usr/local/bin/python")
    .apt_install("patch")
    .add_local_file(
        str(VLLM_LORA_PATCH),
        "/tmp/vllm_deepseek_v4_lora.patch",
        copy=True,
    )
    .run_commands(
        "cd $(python -c 'import pathlib, vllm; "
        "print(pathlib.Path(vllm.__file__).resolve().parent.parent)') && "
        "patch --batch --forward -p1 < /tmp/vllm_deepseek_v4_lora.patch"
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        }
    )
)


def _finalized_adapter_dir(run_id: str, checkpoint_step: int) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path component")
    return Path(CHECKPOINTS_DIR) / run_id / f"epoch_0_step_{checkpoint_step}" / "model"


def _vllm_server_command(
    adapter_dir: Path,
    *,
    max_model_len: int,
    port: int = VLLM_PORT,
) -> list[str]:
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
        str(max_model_len),
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
        str(port),
    ]


def _prepare_vllm_hf_config() -> Path:
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
    quantization_config = config.get("quantization_config")
    expected = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    }
    if quantization_config != expected:
        raise RuntimeError(
            f"Unexpected DeepSeek V4 Flash quantization config: {quantization_config}"
        )
    return config_path.parent


def _validate_finalized_adapter_for_vllm(
    run_id: str,
    checkpoint_step: int,
    max_model_len: int,
) -> tuple[Path, dict[str, Any]]:
    import hashlib

    import torch
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM

    adapter_dir = _finalized_adapter_dir(run_id, checkpoint_step)
    adapter_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    manifest_path = adapter_dir.parent / "checkpoint_manifest.json"
    for path in (adapter_path, config_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing finalized adapter artifact: {path}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_id") != run_id or manifest.get("step") != checkpoint_step:
        raise RuntimeError(
            "Checkpoint manifest does not match the requested adapter: "
            f"run_id={manifest.get('run_id')}, step={manifest.get('step')}"
        )
    digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    if digest != manifest.get("adapter_sha256"):
        raise RuntimeError(
            f"Adapter SHA-256 {digest} does not match manifest "
            f"{manifest.get('adapter_sha256')}"
        )

    adapter_config = json.loads(config_path.read_text())
    if (
        adapter_config.get("r") != LORA_RANK
        or adapter_config.get("lora_alpha") != LORA_ALPHA
    ):
        raise RuntimeError(
            f"Serving expects rank-{LORA_RANK}/alpha-{LORA_ALPHA}, found "
            f"r={adapter_config.get('r')}, "
            f"alpha={adapter_config.get('lora_alpha')}"
        )

    peft_helper = PEFTHelper.from_local_dir(str(adapter_dir), max_model_len)
    mapper = DeepseekV4ForCausalLM.hf_to_vllm_mapper.get_unstacked_mapper()
    lora_model = LoRAModel.from_local_checkpoint(
        str(adapter_dir),
        {"q_a_proj", "kv_proj", "wq_b"},
        peft_helper,
        device="cpu",
        dtype=torch.bfloat16,
        weights_mapper=mapper,
    )
    expected_shapes = {
        "q_a_proj": ((LORA_RANK, 4096), (1024, LORA_RANK)),
        "kv_proj": ((LORA_RANK, 4096), (512, LORA_RANK)),
        "wq_b": ((LORA_RANK, 1024), (32768, LORA_RANK)),
    }
    target_counts = {target: 0 for target in expected_shapes}
    invalid_shapes: list[str] = []
    for name, weights in lora_model.loras.items():
        target = name.rsplit(".", 1)[-1]
        if target not in expected_shapes:
            invalid_shapes.append(f"unexpected module {name}")
            continue
        if weights.lora_a is None or weights.lora_b is None:
            invalid_shapes.append(f"{name}: missing LoRA A or B")
            continue
        actual = (tuple(weights.lora_a.shape), tuple(weights.lora_b.shape))
        if actual != expected_shapes[target]:
            invalid_shapes.append(f"{name}: {actual} != {expected_shapes[target]}")
        target_counts[target] += 1
    if invalid_shapes:
        raise RuntimeError(
            f"vLLM preflight found invalid adapter modules: {invalid_shapes[:10]}"
        )
    expected_target_counts = {target: MODEL_LAYERS for target in expected_shapes}
    if target_counts != expected_target_counts:
        raise RuntimeError(
            f"vLLM mapped unexpected adapter target counts: {target_counts}"
        )

    result = {
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": digest,
        "logical_lora_modules": len(lora_model.loras),
        "target_counts": target_counts,
    }
    return adapter_dir, result


def _network_fingerprint() -> dict[str, object]:
    """Return enough sysfs detail to distinguish EFA from Mellanox RDMA."""
    devices: list[dict[str, object]] = []
    infiniband_root = Path("/sys/class/infiniband")
    if infiniband_root.exists():
        for device_path in sorted(infiniband_root.iterdir()):
            vendor_path = device_path / "device" / "vendor"
            vendor = vendor_path.read_text().strip() if vendor_path.exists() else None
            link_layers: dict[str, str] = {}
            ports_path = device_path / "ports"
            if ports_path.exists():
                for port_path in sorted(ports_path.iterdir()):
                    link_layer_path = port_path / "link_layer"
                    if link_layer_path.exists():
                        link_layers[port_path.name] = (
                            link_layer_path.read_text().strip()
                        )
            devices.append(
                {
                    "name": device_path.name,
                    "vendor": vendor,
                    "link_layers": link_layers,
                }
            )

    vendors = {device["vendor"] for device in devices}
    if "0x1d0f" in vendors:
        provider = "efa"
    elif "0x15b3" in vendors:
        provider = "mellanox"
    elif devices:
        provider = "unknown-rdma"
    else:
        provider = "none"

    return {
        "provider": provider,
        "devices": devices,
        "efa_runtime_present": Path("/opt/amazon/efa").exists(),
    }


def _select_uccl_ep_variant(network: dict[str, object]) -> str:
    """Install the UCCL extension matching the runtime RDMA provider."""
    import importlib.util
    import shutil

    uccl_spec = importlib.util.find_spec("uccl")
    if uccl_spec is None or not uccl_spec.submodule_search_locations:
        raise RuntimeError("UCCL is missing from the training image")
    package_dir = Path(next(iter(uccl_spec.submodule_search_locations)))
    variants = {
        "efa": package_dir / "ep.efa.abi3.so",
        "mellanox": package_dir / "ep.mellanox.abi3.so",
    }
    if not any(path.exists() for path in variants.values()):
        raise RuntimeError("UCCL-EP extensions are missing from the training image")

    provider = str(network["provider"])
    if provider not in {"efa", "mellanox"}:
        raise RuntimeError(
            f"No UCCL-EP build is available for RDMA provider {provider!r}"
        )

    source = variants[provider]
    target = package_dir / "ep.abi3.so"
    if not source.exists():
        raise RuntimeError(f"Missing UCCL-EP {provider} extension at {source}")
    shutil.copy2(source, target)
    print(f"selected_uccl_ep_provider={provider}, extension={target}")
    return provider


def _write_synthetic_chat_jsonl(path: str, *, examples: int, target_words: int) -> None:
    """Write local OpenAI-chat JSONL on every node to avoid shared-volume sync."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = " ".join(f"fact_{idx % 4096}" for idx in range(target_words))
    with open(path, "w") as f:
        for idx in range(examples):
            row = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Repeat the synthetic facts for example {idx}.",
                    },
                    {
                        "role": "assistant",
                        "content": f"Synthetic facts for example {idx}: {payload}",
                    },
                ]
            }
            f.write(json.dumps(row) + "\n")


def _print_cgroup_memory_status(node_rank: int) -> None:
    for name in ("memory.events", "memory.current", "memory.max", "memory.peak"):
        path = Path("/sys/fs/cgroup") / name
        try:
            value = path.read_text().strip().replace("\n", ",")
        except OSError as exc:
            value = f"unavailable ({exc})"
        print(f"[node {node_rank}] cgroup_{name.replace('.', '_')}={value}")


def _training_topology(n_nodes: int) -> tuple[int, dict[str, int]]:
    if n_nodes < 1:
        raise ValueError("N_NODES must be positive")

    total_gpus = n_nodes * GPUS_PER_NODE
    if total_gpus % (PP_SIZE * CP_SIZE):
        raise ValueError(
            f"{total_gpus} GPUs cannot be divided across "
            f"PP={PP_SIZE} and CP={CP_SIZE}"
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


def _recipe_yaml(*, dataset_path: str, checkpoint_dir: str, ep_size: int) -> str:
    target_lines = "\n".join(
        f"  - {json.dumps(target)}" for target in LORA_TARGET_MODULES
    )
    return f"""
recipe: TrainFinetuneRecipeForNextTokenPrediction

seed: 1234

step_scheduler:
  global_batch_size: {GLOBAL_BATCH_SIZE}
  local_batch_size: {LOCAL_BATCH_SIZE}
  ckpt_every_steps: {MAX_STEPS}
  save_checkpoint_every_epoch: false
  val_every_steps: 100000
  gc_every_steps: 1
  num_epochs: 1
  max_steps: {MAX_STEPS}

distributed:
  strategy: fsdp2
  tp_size: 1
  cp_size: {CP_SIZE}
  pp_size: {PP_SIZE}
  ep_size: {ep_size}
  sequence_parallel: false
  activation_checkpointing: true
  pipeline:
    pp_schedule: 1f1b
    pp_microbatch_size: 1
    layers_per_stage: 11
    round_virtual_stages_to_pp_multiple: down
    scale_grads_in_schedule: false
    patch_inner_model: false
    patch_causal_lm_model: false
  moe:
    ignore_router_for_ac: true
    reshard_after_forward: false
    wrap_outer_model: false

dist_env:
  backend: {DIST_BACKEND}
  timeout_minutes: 60

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_config
  config:
    _target_: nemo_automodel.components.models.deepseek_v4.config.DeepseekV4Config.from_pretrained
    pretrained_model_name_or_path: {HF_MODEL}
    name_or_path: {HF_MODEL}
    num_nextn_predict_layers: 0
  trust_remote_code: false
  load_base_model: true
  mtp_loss_scaling_factor: 0.1
  backend:
    _target_: nemo_automodel.components.models.common.BackendConfig
    attn: tilelang
    linear: torch
    rms_norm: torch_fp32
    rope_fusion: false
    dispatcher: uccl_ep
    experts: torch_mm
    enable_hf_state_dict_adapter: true
    enable_fsdp_optimizations: true

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:
{target_lines}
  dim: {LORA_RANK}
  alpha: {LORA_ALPHA}
  dropout: 0.0
  use_memory_efficient_lora: true
  use_triton: false


checkpoint:
  enabled: true
  checkpoint_dir: {json.dumps(checkpoint_dir)}
  model_save_format: safetensors
  save_consolidated: final
  save_optimizer: false
  dequantize_base_checkpoint: true

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: {dataset_path}
  split: train
  shuffle_seed: 42
  truncation: true
  seq_length: {SEQ_LENGTH}
  padding: max_length
  chat_template: |-
    {{% for message in messages %}}{{% if message['role'] == 'user' %}}User:
    {{{{ message['content'] }}}}
    {{% elif message['role'] == 'assistant' %}}Assistant:
    {{% generation %}}{{{{ message['content'] }}}}{{% endgeneration %}}
    {{% endif %}}{{% endfor %}}
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: {HF_MODEL}

packed_sequence:
  packed_sequence_size: 0

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn:
    _target_: nemo_automodel.components.datasets.utils.default_collater
    pad_seq_len_divisible: 8
  shuffle: true

optimizer:
  _target_: torch.optim.AdamW
  betas:
  - 0.9
  - 0.95
  eps: 1e-8
  lr: 1e-5
  weight_decay: 0.1
"""


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
def train_h200_60k_lora(run_id: str, expected_nodes: int):
    import subprocess

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path component")

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else 1
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )
    if n_nodes != expected_nodes:
        raise RuntimeError(f"Expected {expected_nodes} nodes, scheduled {n_nodes}")
    total_gpus, topology = _training_topology(n_nodes)
    ep_size = topology["ep"]

    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET6"
    os.environ["NCCL_SOCKET_IFNAME"] = UCCL_SOCKET_IFNAME
    os.environ["GLOO_SOCKET_IFNAME"] = UCCL_SOCKET_IFNAME
    os.environ["UCCL_SOCKET_IFNAME"] = UCCL_SOCKET_IFNAME
    os.environ["UCCL_SOCKET_FAMILY"] = UCCL_SOCKET_FAMILY
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = UCCL_CUDA_ALLOC_CONF
    os.environ["NCCL_MAX_NCHANNELS"] = NCCL_MAX_NCHANNELS
    os.environ["UCCL_EP_CPU_TIMEOUT_SECS"] = UCCL_EP_CPU_TIMEOUT_SECS
    hf_cache_vol.reload()
    checkpoints_vol.reload()

    network = _network_fingerprint()
    uccl_provider = _select_uccl_ep_variant(network)
    print(
        f"[node {node_rank}] network_fingerprint={json.dumps(network, sort_keys=True)}"
    )

    dataset_path = "/tmp/dsv4_60k_synthetic/training.jsonl"
    # Over-generate words; the tokenizer truncates to seq_length and pads to the
    # fixed length required by DSv4 CP.
    _write_synthetic_chat_jsonl(
        dataset_path,
        examples=SYNTHETIC_EXAMPLES,
        target_words=SEQ_LENGTH * 2,
    )

    checkpoint_dir = f"{CHECKPOINTS_DIR}/{run_id}"
    if Path(checkpoint_dir).exists():
        raise FileExistsError(f"Checkpoint directory already exists: {checkpoint_dir}")

    recipe_path = f"/tmp/{run_id}.yaml"
    Path(recipe_path).write_text(
        _recipe_yaml(
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            ep_size=ep_size,
        )
    )

    summary = {
        "run_id": run_id,
        "nodes": n_nodes,
        "total_gpus": total_gpus,
        "topology": topology,
        "sequence_length": SEQ_LENGTH,
        "optimizer_steps": MAX_STEPS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_targets": LORA_TARGET_MODULES,
        "checkpoint_dir": checkpoint_dir,
        "automodel_commit": AUTOMODEL_COMMIT,
        "uccl_provider": uccl_provider,
    }
    if node_rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True))

    cmd = [
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
        recipe_path,
    ]
    print(f"[node {node_rank}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        _print_cgroup_memory_status(node_rank)
        raise RuntimeError(f"AutoModel training failed with code {result.returncode}")
    checkpoints_vol.commit()
    return summary


def _validate_hf_peft_checkpoint(
    model_dir: Path,
    *,
    expected_layers: int,
    expected_tensor_count: int,
) -> dict[str, object]:
    """Load adapter tensors through PEFT while keeping the 284B base on meta."""
    import re

    import torch
    from accelerate import init_empty_weights
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import (
        load_peft_weights,
        set_peft_model_state_dict,
    )
    from transformers import AutoConfig, AutoModelForCausalLM

    base_config = AutoConfig.from_pretrained(HF_MODEL, trust_remote_code=False)
    adapter_config = PeftConfig.from_pretrained(str(model_dir))
    with init_empty_weights(include_buffers=True):
        base_model = AutoModelForCausalLM.from_config(
            base_config,
            trust_remote_code=False,
            dtype=torch.bfloat16,
        )
    model = get_peft_model(base_model, adapter_config, low_cpu_mem_usage=True)
    adapter_params = [
        (name, param) for name, param in model.named_parameters() if "lora_" in name
    ]
    if len(adapter_params) != expected_tensor_count:
        raise RuntimeError(
            f"PEFT injected {len(adapter_params)} adapter parameters, expected "
            f"{expected_tensor_count}"
        )

    for module in model.modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            module.lora_A["default"].to_empty(device="cpu")
            module.lora_B["default"].to_empty(device="cpu")

    state = load_peft_weights(str(model_dir), device="cpu")
    load_result = set_peft_model_state_dict(model, state, adapter_name="default")
    missing_adapter_keys = [key for key in load_result.missing_keys if "lora_" in key]
    if missing_adapter_keys:
        raise RuntimeError(
            f"PEFT load is missing {len(missing_adapter_keys)} adapter keys: "
            f"{missing_adapter_keys[:10]}"
        )
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"PEFT load has {len(load_result.unexpected_keys)} unexpected keys: "
            f"{load_result.unexpected_keys[:10]}"
        )

    adapter_params = [
        (name, param) for name, param in model.named_parameters() if "lora_" in name
    ]
    meta_params = [name for name, param in adapter_params if param.is_meta]
    nonfinite_params = [
        name
        for name, param in adapter_params
        if not param.is_meta and not torch.isfinite(param).all()
    ]
    zero_lora_b = [
        name
        for name, param in adapter_params
        if ".lora_B." in name
        and not param.is_meta
        and torch.count_nonzero(param).item() == 0
    ]
    lora_b_count = sum(".lora_B." in name for name, _ in adapter_params)
    expected_lora_b_count = expected_tensor_count // 2
    if lora_b_count != expected_lora_b_count:
        raise RuntimeError(
            f"PEFT injected {lora_b_count} LoRA-B parameters, expected "
            f"{expected_lora_b_count}"
        )
    if meta_params or nonfinite_params or zero_lora_b:
        raise RuntimeError(
            "PEFT load produced invalid adapter parameters: "
            f"meta={meta_params[:5]}, nonfinite={nonfinite_params[:5]}, "
            f"zero_lora_b={zero_lora_b[:5]}"
        )

    layers = sorted(
        {
            int(match.group(1))
            for name, _ in adapter_params
            if (match := re.search(r"\.layers\.(\d+)\.", name))
        }
    )
    targets = sorted(
        {
            match.group(1)
            for name, _ in adapter_params
            if (match := re.search(r"\.self_attn\.(q_a_proj|q_b_proj|kv_proj)\.", name))
        }
    )
    if layers != list(range(expected_layers)):
        raise RuntimeError(f"PEFT load covered unexpected layers: {layers}")
    expected_targets = ["kv_proj", "q_a_proj", "q_b_proj"]
    if targets != expected_targets:
        raise RuntimeError(f"PEFT load covered unexpected targets: {targets}")

    return {
        "adapter_parameters": len(adapter_params),
        "layers": len(layers),
        "lora_b_parameters": lora_b_count,
        "missing_adapter_keys": len(missing_adapter_keys),
        "targets": targets,
        "unexpected_keys": len(load_result.unexpected_keys),
    }


@app.function(
    image=automodel_image,
    volumes={HF_CACHE: hf_cache_vol, CHECKPOINTS_DIR: checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    memory=HOST_MEMORY,
)
def finalize_lora_checkpoint(run_id: str):
    """Merge PP-local LoRA shards after all training nodes commit the volume."""
    import hashlib
    import re

    import torch
    from safetensors.torch import load_file, save_file

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path component")

    expected_step = MAX_STEPS - 1
    expected_pp_size = PP_SIZE
    expected_layers = MODEL_LAYERS
    expected_lora_rank = LORA_RANK
    expected_lora_alpha = LORA_ALPHA

    checkpoints_vol.reload()
    run_dir = Path(CHECKPOINTS_DIR) / run_id
    candidates: list[tuple[tuple[int, int], Path]] = []
    pattern = re.compile(r"epoch_(\d+)_step_(\d+)")
    if run_dir.exists():
        for path in run_dir.iterdir():
            match = pattern.fullmatch(path.name)
            if path.is_dir() and match:
                candidates.append(((int(match.group(1)), int(match.group(2))), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint steps found under {run_dir}")

    (epoch, step), checkpoint_dir = max(candidates)
    if step != expected_step:
        raise RuntimeError(
            f"Latest checkpoint is step {step}, expected {expected_step}"
        )
    model_dir = checkpoint_dir / "model"
    stage_shards = sorted(model_dir.glob("adapter_model.pp-*-of-*.safetensors"))
    if len(stage_shards) != expected_pp_size:
        raise RuntimeError(
            f"Found {len(stage_shards)} PP adapter shards, expected {expected_pp_size}: "
            f"{[path.name for path in stage_shards]}"
        )

    merged: dict[str, torch.Tensor] = {}
    shard_key_counts: dict[str, int] = {}
    for shard_path in stage_shards:
        shard = load_file(str(shard_path), device="cpu")
        duplicate_keys = sorted(set(merged).intersection(shard))
        if duplicate_keys:
            raise RuntimeError(
                f"Duplicate adapter keys in {shard_path.name}: {duplicate_keys[:10]}"
            )
        for name, tensor in shard.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite adapter tensor: {name}")
        merged.update(shard)
        shard_key_counts[shard_path.name] = len(shard)

    automodel_to_peft_targets = {
        "wq_a": "q_a_proj",
        "wq_b": "q_b_proj",
        "wkv": "kv_proj",
    }
    peft_target_pattern = r"model\.layers\.\d+\.self_attn\.(q_a_proj|q_b_proj|kv_proj)"
    expected_adapters = ("lora_A", "lora_B")
    expected_shapes = {
        "wq_a": {
            "lora_A": (expected_lora_rank, 4096),
            "lora_B": (1024, expected_lora_rank),
        },
        "wq_b": {
            "lora_A": (expected_lora_rank, 1024),
            "lora_B": (32768, expected_lora_rank),
        },
        "wkv": {
            "lora_A": (expected_lora_rank, 4096),
            "lora_B": (512, expected_lora_rank),
        },
    }
    expected_tensor_count = (
        expected_layers * len(automodel_to_peft_targets) * len(expected_adapters)
    )
    if len(merged) != expected_tensor_count:
        raise RuntimeError(
            f"Merged adapter has {len(merged)} tensors, expected "
            f"{expected_tensor_count}"
        )

    missing: list[str] = []
    duplicate_matches: list[str] = []
    wrong_shapes: list[str] = []
    zero_lora_b: list[str] = []
    matched_names: set[str] = set()
    peft_state: dict[str, torch.Tensor] = {}
    for layer in range(expected_layers):
        for automodel_target, peft_target in automodel_to_peft_targets.items():
            for adapter in expected_adapters:
                needle = (
                    f".layers.{layer}.self_attn.{automodel_target}.{adapter}.weight"
                )
                matches = [key for key in merged if needle in key]
                if not matches:
                    missing.append(needle)
                    continue
                if len(matches) != 1:
                    duplicate_matches.append(needle)
                    continue

                name = matches[0]
                tensor = merged[name]
                matched_names.add(name)
                expected_shape = expected_shapes[automodel_target][adapter]
                if tuple(tensor.shape) != expected_shape:
                    wrong_shapes.append(
                        f"{name}: {tuple(tensor.shape)} != {expected_shape}"
                    )
                if adapter == "lora_B" and torch.count_nonzero(tensor).item() == 0:
                    zero_lora_b.append(name)

                peft_name = name.replace(
                    f".self_attn.{automodel_target}.",
                    f".self_attn.{peft_target}.",
                    1,
                )
                if peft_name in peft_state:
                    raise RuntimeError(f"Duplicate PEFT adapter key: {peft_name}")
                peft_state[peft_name] = tensor
    if missing:
        raise RuntimeError(
            f"Merged adapter is missing {len(missing)} expected tensors: {missing[:10]}"
        )
    if duplicate_matches:
        raise RuntimeError(
            f"Merged adapter has duplicate logical tensors: {duplicate_matches[:10]}"
        )
    unexpected = sorted(set(merged).difference(matched_names))
    if unexpected:
        raise RuntimeError(
            f"Merged adapter has {len(unexpected)} unexpected tensors: "
            f"{unexpected[:10]}"
        )
    if wrong_shapes:
        raise RuntimeError(
            f"Merged adapter has {len(wrong_shapes)} wrong tensor shapes: "
            f"{wrong_shapes[:10]}"
        )
    if zero_lora_b:
        raise RuntimeError(
            f"Merged adapter has {len(zero_lora_b)} untrained LoRA-B tensors: "
            f"{zero_lora_b[:10]}"
        )
    merged = peft_state

    adapter_config_path = model_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text())
    if adapter_config.get("r") != expected_lora_rank:
        raise RuntimeError(
            f"Adapter rank is {adapter_config.get('r')}, expected {expected_lora_rank}"
        )
    if adapter_config.get("lora_alpha") != expected_lora_alpha:
        raise RuntimeError(
            "Adapter alpha is "
            f"{adapter_config.get('lora_alpha')}, expected {expected_lora_alpha}"
        )
    adapter_config["target_modules"] = peft_target_pattern
    temporary_config_path = model_dir / "adapter_config.json.tmp"
    temporary_config_path.write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n"
    )
    temporary_config_path.replace(adapter_config_path)

    adapter_path = model_dir / "adapter_model.safetensors"
    temporary_path = model_dir / "adapter_model.safetensors.tmp"
    save_file(merged, str(temporary_path))
    temporary_path.replace(adapter_path)
    peft_validation = _validate_hf_peft_checkpoint(
        model_dir,
        expected_layers=expected_layers,
        expected_tensor_count=expected_tensor_count,
    )
    digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    manifest = {
        "run_id": run_id,
        "epoch": epoch,
        "step": step,
        "pipeline_stages": expected_pp_size,
        "peft_validation": peft_validation,
        "stage_shards": [path.name for path in stage_shards],
        "shard_key_counts": shard_key_counts,
        "adapter_tensor_count": len(merged),
        "adapter_bytes": adapter_path.stat().st_size,
        "adapter_sha256": digest,
        "layers": expected_layers,
        "lora_rank": expected_lora_rank,
        "lora_alpha": expected_lora_alpha,
        "automodel_targets": list(automodel_to_peft_targets),
        "targets": list(automodel_to_peft_targets.values()),
        "target_modules_pattern": peft_target_pattern,
        "updated_lora_b_tensors": expected_layers * len(automodel_to_peft_targets),
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    checkpoints_vol.commit()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


@app.function(
    image=vllm_image,
    gpu=VLLM_GPU,
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
def serve_lora():
    """Serve the finalized adapter selected by SERVE_RUN_ID."""
    import subprocess

    if not SERVE_RUN_ID:
        raise RuntimeError("Set SERVE_RUN_ID to a finalized checkpoint run ID")
    hf_cache_vol.reload()
    checkpoints_vol.reload()
    _prepare_vllm_hf_config()
    adapter_dir, preflight = _validate_finalized_adapter_for_vllm(
        SERVE_RUN_ID,
        SERVE_CHECKPOINT_STEP,
        SERVE_MAX_MODEL_LEN,
    )
    print(json.dumps({"vllm_preflight": preflight}, indent=2, sort_keys=True))
    subprocess.Popen(
        _vllm_server_command(
            adapter_dir,
            max_model_len=SERVE_MAX_MODEL_LEN,
        )
    )


@app.function(
    image=vllm_image,
    gpu=VLLM_GPU,
    volumes={HF_CACHE: hf_cache_vol, CHECKPOINTS_DIR: checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=21600,
    retries=0,
    memory=HOST_MEMORY,
    experimental_options={"efa_enabled": True},
)
def validate_lora_serving(run_id: str):
    """Compare base and adapter quality, including retrieval at 60k context."""
    import decimal
    import re
    import signal
    import subprocess
    import time
    import traceback
    import urllib.error
    import urllib.request

    import vllm
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    checkpoint_step = MAX_STEPS - 1
    max_model_len = 64 * 1024
    long_prompt_tokens = SEQ_LENGTH
    startup_timeout_seconds = 3600

    hf_cache_vol.reload()
    checkpoints_vol.reload()
    fresh_config_dir = _prepare_vllm_hf_config()
    print(f"vLLM HF config: {fresh_config_dir}")
    adapter_dir, preflight = _validate_finalized_adapter_for_vllm(
        run_id,
        checkpoint_step,
        max_model_len,
    )
    print(json.dumps({"vllm_preflight": preflight}, indent=2, sort_keys=True))

    snapshot_path = snapshot_download(
        HF_MODEL,
        token=os.environ.get("HF_TOKEN"),
    )
    hf_cache_vol.commit()
    print(f"Base model snapshot: {snapshot_path}")

    server_log_path = "/tmp/vllm-lora-server.log"
    server_log = open(server_log_path, "w")
    server_command = _vllm_server_command(
        adapter_dir,
        max_model_len=max_model_len,
    )
    server_proc: subprocess.Popen[bytes] | None = None

    def tail_log(limit: int = 50000) -> str:
        server_log.flush()
        return Path(server_log_path).read_text(errors="replace")[-limit:]

    def stop_server(proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)

    def request_json(
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}{path}",
            data=body,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"vLLM {path} returned HTTP {exc.code}: {error_body[:2000]}"
            ) from None

    def response_text(response: dict[str, Any]) -> str:
        if not response.get("choices"):
            raise RuntimeError(f"vLLM returned no choices: {response}")
        message = response["choices"][0].get("message", {})
        parts = [
            message.get("reasoning_content"),
            message.get("content"),
        ]
        text = "\n".join(part for part in parts if part)
        if not text:
            raise RuntimeError(f"vLLM returned no response text: {response}")
        return text

    def extract_number(text: str) -> str | None:
        final_matches = re.findall(
            r"FINAL\s*:\s*(-?\d[\d,]*(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        matches = final_matches or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return matches[-1].replace(",", "") if matches else None

    def numbers_equal(predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        try:
            return decimal.Decimal(predicted) == decimal.Decimal(expected)
        except decimal.InvalidOperation:
            return False

    try:
        max_starts = 4
        ready = False
        for start_attempt in range(1, max_starts + 1):
            print(
                f"Starting vLLM {vllm.__version__} "
                f"(attempt {start_attempt}/{max_starts})"
            )
            server_proc = subprocess.Popen(
                server_command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + startup_timeout_seconds
            while time.monotonic() < deadline:
                if server_proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{VLLM_PORT}/health",
                        timeout=5,
                    ):
                        ready = True
                        break
                except Exception:
                    time.sleep(5)
            if ready:
                break

            startup_log = tail_log(20000)
            stop_server(server_proc)
            transient_cuda_init = (
                "system not yet initialized" in startup_log
                or "Error 802" in startup_log
            )
            if transient_cuda_init and start_attempt < max_starts:
                print("Retrying transient CUDA error 802 after 15 seconds")
                time.sleep(15)
                continue
            raise RuntimeError("vLLM server failed to become healthy:\n" + startup_log)
        if not ready or server_proc is None:
            raise RuntimeError("vLLM server did not become healthy")

        models = request_json("/v1/models", timeout=30)
        model_ids = {item["id"] for item in models.get("data", [])}
        expected_model_ids = {HF_MODEL, VLLM_ADAPTER_NAME}
        if not expected_model_ids.issubset(model_ids):
            raise RuntimeError(
                f"Base or adapter is absent from /v1/models: {sorted(model_ids)}"
            )

        with urllib.request.urlopen(GSM8K_TEST_URL, timeout=60) as response:
            gsm8k_records = [
                json.loads(line) for line in response.read().decode().splitlines()
            ]
        eval_records = [gsm8k_records[index] for index in GSM8K_EVAL_INDICES]
        quality_results: dict[str, Any] = {}
        for model_id in (HF_MODEL, VLLM_ADAPTER_NAME):
            examples = []
            for index, record in zip(GSM8K_EVAL_INDICES, eval_records, strict=True):
                expected = (
                    record["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
                )
                chat = request_json(
                    "/v1/chat/completions",
                    {
                        "model": model_id,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Solve this math problem. End with a line in the "
                                    "form FINAL: <number>.\n\n" + record["question"]
                                ),
                            }
                        ],
                        "max_tokens": 512,
                        "temperature": 0,
                        "chat_template_kwargs": {"thinking": False},
                    },
                    timeout=900,
                )
                text = response_text(chat)
                predicted = extract_number(text)
                examples.append(
                    {
                        "index": index,
                        "expected": expected,
                        "predicted": predicted,
                        "correct": numbers_equal(predicted, expected),
                        "output_tail": text[-300:],
                    }
                )
                print(
                    f"GSM8K model={model_id} index={index} "
                    f"predicted={predicted} expected={expected}"
                )
            quality_results[model_id] = {
                "correct": sum(example["correct"] for example in examples),
                "total": len(examples),
                "examples": examples,
            }

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            trust_remote_code=True,
        )
        secret = "7391846250"
        prefix = (
            "A long archive document follows. Preserve its verified code.\n"
            f"VERIFIED ARCHIVE CODE: {secret}\n"
        )
        filler = (
            "This unrelated archive line records routine inventory notes and "
            "does not modify the verified code.\n"
        )
        suffix = (
            "\nEnd of archive. What is the VERIFIED ARCHIVE CODE stated near "
            "the beginning? Reply with only the ten digits.\nAnswer: "
        )
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        filler_ids = tokenizer.encode(filler, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        filler_count = long_prompt_tokens - len(prefix_ids) - len(suffix_ids)
        if filler_count <= 0 or not filler_ids:
            raise RuntimeError("Could not construct the long-context prompt")
        repeated_filler = (
            filler_ids * ((filler_count + len(filler_ids) - 1) // len(filler_ids))
        )[:filler_count]
        prompt_ids = prefix_ids + repeated_filler + suffix_ids
        if len(prompt_ids) != long_prompt_tokens:
            raise RuntimeError(
                f"Constructed {len(prompt_ids)} prompt tokens, "
                f"expected {long_prompt_tokens}"
            )

        retrieval_results: dict[str, Any] = {}
        for model_id in (HF_MODEL, VLLM_ADAPTER_NAME):
            completion = request_json(
                "/v1/completions",
                {
                    "model": model_id,
                    "prompt": prompt_ids,
                    "max_tokens": 32,
                    "temperature": 0,
                },
                timeout=3600,
            )
            usage = completion.get("usage", {})
            if completion.get("model") != model_id:
                raise RuntimeError(
                    f"Long-context response used the wrong model: {completion}"
                )
            if usage.get("prompt_tokens") != long_prompt_tokens:
                raise RuntimeError(
                    f"Long-context request did not process 60k tokens: {usage}"
                )
            if usage.get("completion_tokens", 0) <= 0 or not completion.get("choices"):
                raise RuntimeError(f"Long-context generation failed: {completion}")
            text = completion["choices"][0].get("text", "")
            answer = text.strip().splitlines()[0] if text.strip() else ""
            retrieved = answer == secret
            if not retrieved:
                raise RuntimeError(
                    f"{model_id} failed 60k retrieval: output={text[:200]!r}"
                )
            retrieval_results[model_id] = {
                "answer": answer,
                "output": text[:200],
                "retrieved": retrieved,
                "usage": usage,
            }
            print(
                f"60k retrieval model={model_id} "
                f"retrieved={retrieved} output={text[:200]!r}"
            )

        base_examples = quality_results[HF_MODEL]["examples"]
        adapter_examples = quality_results[VLLM_ADAPTER_NAME]["examples"]
        prediction_agreement = sum(
            base["predicted"] == adapter["predicted"]
            for base, adapter in zip(base_examples, adapter_examples, strict=True)
        )
        result = {
            "adapter_name": VLLM_ADAPTER_NAME,
            "adapter_sha256": preflight["adapter_sha256"],
            "checkpoint_step": checkpoint_step,
            "gsm8k": quality_results,
            "gsm8k_prediction_agreement": {
                "matching": prediction_agreement,
                "total": len(base_examples),
            },
            "long_context_retrieval": retrieval_results,
            "max_model_len": max_model_len,
            "model_ids": sorted(model_ids),
            "run_id": run_id,
            "vllm": vllm.__version__,
            "vllm_log_tail": tail_log(4000),
            "vllm_preflight": preflight,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    except Exception:
        traceback.print_exc()
        print("\n=== vLLM server log ===")
        print(tail_log())
        raise
    finally:
        if server_proc is not None:
            stop_server(server_proc)
        server_log.close()


@app.local_entrypoint()
def h200_60k_lora_5step(
    run_id: str,
):
    _, topology = _training_topology(N_NODES)
    call = train_h200_60k_lora.spawn(
        run_id=run_id,
        expected_nodes=N_NODES,
    )
    call_id = (
        getattr(call, "object_id", None)
        or getattr(call, "function_call_id", None)
        or str(call)
    )
    result = {
        "function_call_id": call_id,
        "run_id": run_id,
        "max_steps": MAX_STEPS,
        "checkpoint_dir": f"{CHECKPOINTS_DIR}/{run_id}",
        "nodes": N_NODES,
        "topology": topology,
    }
    print(json.dumps(result, indent=2))
    return result
