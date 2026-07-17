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
VLLM_VERSION = "0.25.1"
VLLM_ADAPTER_NAME = "deepseek-v4-flash-60k-lora"
VLLM_PORT = 8000
VLLM_PIPELINE_PARALLEL_SIZE = 4
VLLM_GPU = f"H200:{VLLM_PIPELINE_PARALLEL_SIZE}"
VLLM_HF_CONFIG_DIR = Path("/tmp/deepseek-v4-flash-vllm-config")
GPUS_PER_NODE = 8
N_NODES = int(os.environ.get("N_NODES", "16"))
HOST_MEMORY_REQUEST_MB = int(os.environ.get("HOST_MEMORY_REQUEST_MB", "128"))
HOST_MEMORY_LIMIT_MB = int(os.environ.get("HOST_MEMORY_LIMIT_MB", str(256 * 1024)))
HOST_MEMORY = (
    HOST_MEMORY_REQUEST_MB
    if HOST_MEMORY_REQUEST_MB == HOST_MEMORY_LIMIT_MB
    else (HOST_MEMORY_REQUEST_MB, HOST_MEMORY_LIMIT_MB)
)
EPHEMERAL_DISK_MB = int(os.environ.get("EPHEMERAL_DISK_MB", "0"))
EPHEMERAL_DISK_OPTIONS: dict[str, Any] = (
    {"ephemeral_disk": EPHEMERAL_DISK_MB} if EPHEMERAL_DISK_MB > 0 else {}
)
ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "tilelang")
MOE_DISPATCHER = os.environ.get("MOE_DISPATCHER", "uccl_ep")
EFA_ENABLED = os.environ.get("EFA_ENABLED", "1").lower() not in {"0", "false", "no"}
MODAL_CLOUD = os.environ.get("MODAL_CLOUD") or None
NVSHMEM_BOOTSTRAP_IFNAME = os.environ.get("NVSHMEM_BOOTSTRAP_IFNAME", "eth1")
NVSHMEM_BOOTSTRAP_FAMILY = os.environ.get("NVSHMEM_BOOTSTRAP_FAMILY", "AF_INET6")
NVSHMEM_IBGDA_NIC_HANDLER = os.environ.get("NVSHMEM_IBGDA_NIC_HANDLER", "auto")
NVSHMEM_DISABLE_P2P = int(os.environ.get("NVSHMEM_DISABLE_P2P", "0"))
INSTALL_UCCL_EP = os.environ.get("INSTALL_UCCL_EP", "1").lower() in {"1", "true", "yes"}
UCCL_EP_USE_EFA = os.environ.get("UCCL_EP_USE_EFA", "1").lower() in {"1", "true", "yes"}
UCCL_EP_BUILD_FROM_SOURCE = os.environ.get(
    "UCCL_EP_BUILD_FROM_SOURCE", "1"
).lower() in {"1", "true", "yes"}
UCCL_EP_USE_DMABUF = os.environ.get("UCCL_EP_USE_DMABUF", "1").lower() in {
    "1",
    "true",
    "yes",
}
UCCL_SOCKET_IFNAME = os.environ.get("UCCL_SOCKET_IFNAME", "eth1")
UCCL_SOCKET_FAMILY = os.environ.get("UCCL_SOCKET_FAMILY", "AF_INET6")
UCCL_IB_GID_INDEX = int(os.environ.get("UCCL_IB_GID_INDEX", "-1"))
DIST_BACKEND = os.environ.get("DIST_BACKEND", "cpu:gloo,cuda:nccl")
UCCL_CUDA_ALLOC_CONF = os.environ.get(
    "UCCL_CUDA_ALLOC_CONF", "expandable_segments:False"
)
NCCL_MAX_NCHANNELS = os.environ.get("NCCL_MAX_NCHANNELS", "8")
UCCL_EP_CPU_TIMEOUT_SECS = os.environ.get("UCCL_EP_CPU_TIMEOUT_SECS", "600")
DEFAULT_LORA_TARGET_MODULES = ",".join(
    (
        "*.self_attn.wq_a",
        "*.self_attn.wq_b",
        "*.self_attn.wkv",
    )
)

HF_CACHE = "/root/.cache/huggingface"
CHECKPOINTS_DIR = "/checkpoints"

# Includes DSv4 model-owned CP support and the TileLang dependency fixes.
AUTOMODEL_COMMIT = os.environ.get(
    "AUTOMODEL_COMMIT", "1197b6281255957cc2c58e79d50796c1b256c57c"
)
AUTOMODEL_IMAGE = "nvcr.io/nvidia/nemo-automodel:26.06.00"
UCCL_EP_VARIANT = "cu13.efa" if UCCL_EP_USE_EFA else "cu13"
UCCL_EP_WHEEL = (
    "https://github.com/uccl-project/uccl/releases/download/v0.1.1/"
    f"uccl-0.1.1%2B{UCCL_EP_VARIANT}-cp312-abi3-manylinux_2_35_x86_64.whl"
)
# v0.1.1's EFA wheel predates USE_DMABUF support and requires the host's
# efa_nv_peermem module. This commit includes the public DMA-BUF build path.
UCCL_EP_COMMIT = os.environ.get(
    "UCCL_EP_COMMIT", "66170bc299205228f0170bc1638594a39af9ffd5"
)
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
SERVE_CHECKPOINT_STEP = int(os.environ.get("SERVE_CHECKPOINT_STEP", "4"))
SERVE_MAX_MODEL_LEN = int(os.environ.get("SERVE_MAX_MODEL_LEN", str(64 * 1024)))

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

if INSTALL_UCCL_EP:
    if UCCL_EP_BUILD_FROM_SOURCE:
        automodel_image = automodel_image.apt_install(
            "build-essential",
            "curl",
            "libibverbs-dev",
            "libnl-3-dev",
            "libnl-route-3-dev",
            "libnuma-dev",
            "ninja-build",
            "patch",
            "rdma-core",
        ).uv_pip_install(
            "intervaltree==3.1.0",
            "nanobind==2.13.0",
            "pybind11==3.0.1",
        )
        if UCCL_EP_USE_EFA:
            automodel_image = automodel_image.run_commands(
                "cd /tmp && "
                f"curl -fsSLO https://efa-installer.amazonaws.com/aws-efa-installer-{UCCL_EFA_INSTALLER_VERSION}.tar.gz && "
                f"tar -xzf aws-efa-installer-{UCCL_EFA_INSTALLER_VERSION}.tar.gz && "
                "cd aws-efa-installer && "
                "./efa_installer.sh -y --skip-kmod -g --no-verify && "
                "rm -rf /tmp/aws-efa-installer*"
            )
        automodel_image = automodel_image.add_local_file(
            str(UCCL_IPV6_PATCH), "/tmp/uccl_ipv6_oob.patch", copy=True
        ).run_commands(
            "rm -rf /opt/uccl && mkdir -p /opt/uccl && "
            f"curl -fsSL https://github.com/uccl-project/uccl/archive/{UCCL_EP_COMMIT}.tar.gz "
            "| tar -xz --strip-components=1 -C /opt/uccl && "
            "cd /opt/uccl && patch -p1 < /tmp/uccl_ipv6_oob.patch",
            "cd /opt/uccl/ep && rm -rf build ep*.so && "
            "EFA_HOME=/opt/amazon/efa "
            f"USE_DMABUF={int(UCCL_EP_USE_DMABUF)} "
            "TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 "
            "/opt/venv/bin/python setup.py build_ext --inplace && "
            "cp ep*.so /opt/uccl/uccl/ep.efa.abi3.so",
            "cd /opt/uccl/ep && rm -rf build ep*.so && "
            "EFA_HOME=/opt/uccl/no-efa "
            f"USE_DMABUF={int(UCCL_EP_USE_DMABUF)} "
            "TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 "
            "/opt/venv/bin/python setup.py build_ext --inplace && "
            "cp ep*.so /opt/uccl/uccl/ep.mellanox.abi3.so && "
            "cp ep*.so /opt/uccl/uccl/ep.abi3.so",
            "cd /opt/uccl && "
            "uv pip install --python /opt/venv/bin/python --no-build-isolation "
            "--no-deps --force-reinstall .",
        )
    else:
        automodel_image = automodel_image.uv_pip_install(UCCL_EP_WHEEL)

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
        "64",
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
    if adapter_config.get("r") != 64 or adapter_config.get("lora_alpha") != 64:
        raise RuntimeError(
            "Serving expects the validated rank-64/alpha-64 adapter, found "
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
        "q_a_proj": ((64, 4096), (1024, 64)),
        "kv_proj": ((64, 4096), (512, 64)),
        "wq_b": ((64, 1024), (32768, 64)),
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
    expected_target_counts = {target: 43 for target in expected_shapes}
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


UCCL_EP_PROBE_SOURCE = r"""import faulthandler
import os

import torch
import torch.distributed as dist

from uccl import ep as uccl_ep

from nemo_automodel.components.moe.megatron.fused_a2a import (
    free_uccl_buffer,
    uccl_fused_combine,
    uccl_fused_dispatch,
)


def main() -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=os.environ["UCCL_PROBE_DIST_BACKEND"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Composite process groups keep Python-object metadata on Gloo while CUDA
    # tensor collectives continue to use NCCL.
    tensor_probe = torch.tensor(rank + 1, device="cuda", dtype=torch.int64)
    dist.all_reduce(tensor_probe)
    expected_sum = world_size * (world_size + 1) // 2
    if tensor_probe.item() != expected_sum:
        raise RuntimeError(f"NCCL all-reduce returned {tensor_probe.item()}, expected {expected_sum}")
    object_probe = [None] * world_size
    dist.all_gather_object(object_probe, {"rank": rank})
    if [item["rank"] for item in object_probe] != list(range(world_size)):
        raise RuntimeError(f"Gloo object gather returned invalid metadata: {object_probe}")
    oob_ips = [None] * world_size
    dist.all_gather_object(oob_ips, uccl_ep.get_oob_ip())
    if rank == 0:
        print(
            "UCCL_EP_CONTROL_PLANE_OK "
            f"backend={os.environ['UCCL_PROBE_DIST_BACKEND']} world_size={world_size} "
            f"oob_ips={sorted(set(oob_ips))}",
            flush=True,
        )

    tokens = int(os.environ["UCCL_PROBE_TOKENS"])
    hidden_size = int(os.environ["UCCL_PROBE_HIDDEN_SIZE"])
    num_experts = int(os.environ["UCCL_PROBE_NUM_EXPERTS"])
    topk = int(os.environ["UCCL_PROBE_TOPK"])
    if num_experts % world_size != 0:
        raise ValueError(f"num_experts={num_experts} must divide world_size={world_size}")

    generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
    hidden = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    token_offsets = torch.arange(tokens, device="cuda", dtype=torch.int64)[:, None] * topk
    topk_offsets = torch.arange(topk, device="cuda", dtype=torch.int64)[None, :]
    token_indices = (token_offsets + topk_offsets + rank * topk) % num_experts
    token_probs = torch.full(
        (tokens, topk),
        1.0 / topk,
        device="cuda",
        dtype=torch.float32,
    )

    recv_hidden, _, recv_probs, _, handle = uccl_fused_dispatch(
        hidden,
        token_indices,
        token_probs,
        num_experts,
        dist.group.WORLD,
    )
    # Keep a gradient edge through the returned probabilities, as real MoE
    # expert weighting does, while leaving this transport probe numerically neutral.
    recv_hidden = recv_hidden * (1.0 + recv_probs.sum() * 0.0)
    combined, _ = uccl_fused_combine(recv_hidden, dist.group.WORLD, handle)
    if combined.shape != hidden.shape:
        raise RuntimeError(f"combine shape {combined.shape} != input shape {hidden.shape}")
    if not torch.isfinite(combined).all():
        raise RuntimeError("UCCL-EP combine produced non-finite values")

    combined.float().square().mean().backward()
    if hidden.grad is None or not torch.isfinite(hidden.grad).all():
        raise RuntimeError("UCCL-EP backward produced an invalid input gradient")

    torch.cuda.synchronize()
    completed = torch.ones((), device="cuda", dtype=torch.int32)
    dist.all_reduce(completed)
    if rank == 0:
        print(
            "UCCL_EP_PROBE_OK "
            f"world_size={world_size} tokens={tokens} hidden_size={hidden_size} "
            f"num_experts={num_experts} topk={topk} completed={completed.item()}"
        )
    dist.barrier()
    free_uccl_buffer()
    dist.barrier()
    if rank == 0:
        print("UCCL_EP_TEARDOWN_OK", flush=True)
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
"""


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


def _select_uccl_ep_variant(network: dict[str, object]) -> str | None:
    """Install the UCCL extension matching the runtime RDMA provider."""
    import importlib.util
    import shutil

    uccl_spec = importlib.util.find_spec("uccl")
    if uccl_spec is None or not uccl_spec.submodule_search_locations:
        return None
    package_dir = Path(next(iter(uccl_spec.submodule_search_locations)))
    variants = {
        "efa": package_dir / "ep.efa.abi3.so",
        "mellanox": package_dir / "ep.mellanox.abi3.so",
    }
    if not any(path.exists() for path in variants.values()):
        return None

    provider = str(network["provider"])
    if provider == "none":
        provider = "efa" if UCCL_EP_USE_EFA else "mellanox"
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


def _print_nccl_network_logs(node_rank: int) -> None:
    """Print a bounded NCCL transport summary after torchrun exits."""
    markers = (
        "Bootstrap : Using",
        "NET/Plugin",
        "NET/IB",
        "NET/OFI",
        "NET/Socket",
        "Using network",
        "NCCL_NET",
        "RAS",
    )
    matches: list[str] = []
    seen: set[str] = set()
    log_paths = sorted(Path("/tmp").glob("nccl.*.log"))
    for log_path in log_paths:
        try:
            lines = log_path.read_text(errors="replace").splitlines()
        except OSError as exc:
            print(f"[node {node_rank}] failed to read {log_path}: {exc}")
            continue
        for line in lines:
            if any(marker in line for marker in markers) and line not in seen:
                seen.add(line)
                matches.append(line)
                if len(matches) >= 120:
                    break
        if len(matches) >= 120:
            break

    print(
        f"[node {node_rank}] NCCL network diagnostics: "
        f"files={len(log_paths)}, matching_lines={len(matches)}"
    )
    for line in matches:
        print(f"[node {node_rank}] {line}")


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


def _recipe_yaml(
    *,
    dataset_path: str,
    seq_length: int,
    max_steps: int,
    global_batch_size: int,
    local_batch_size: int,
    cp_size: int,
    pp_size: int,
    ep_size: int,
    attn_backend: str,
    moe_dispatcher: str,
    dist_backend: str,
    lora_rank: int,
    lora_alpha: int,
    lora_target_modules: str,
    checkpoint_dir: str | None,
    save_optimizer: bool,
) -> str:
    peft_block = ""
    if lora_rank > 0:
        targets = [
            target.strip()
            for target in lora_target_modules.split(",")
            if target.strip()
        ]
        if not targets:
            raise ValueError(
                "lora_target_modules must contain at least one target when LoRA is enabled"
            )
        target_lines = "\n".join(f"  - {json.dumps(target)}" for target in targets)
        peft_block = f"""
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:
{target_lines}
  dim: {lora_rank}
  alpha: {lora_alpha}
  dropout: 0.0
  use_memory_efficient_lora: true
  use_triton: false
"""

    checkpoint_enabled = checkpoint_dir is not None
    checkpoint_path = checkpoint_dir or "/tmp/disabled-checkpoints"

    return f"""
recipe: TrainFinetuneRecipeForNextTokenPrediction

seed: 1234

step_scheduler:
  global_batch_size: {global_batch_size}
  local_batch_size: {local_batch_size}
  ckpt_every_steps: {max_steps}
  save_checkpoint_every_epoch: false
  val_every_steps: 100000
  gc_every_steps: 1
  num_epochs: 1
  max_steps: {max_steps}

distributed:
  strategy: fsdp2
  tp_size: 1
  cp_size: {cp_size}
  pp_size: {pp_size}
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
  backend: {dist_backend}
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
    attn: {attn_backend}
    linear: torch
    rms_norm: torch_fp32
    rope_fusion: false
    dispatcher: {moe_dispatcher}
    experts: torch_mm
    enable_hf_state_dict_adapter: true
    enable_fsdp_optimizations: true
{peft_block}

checkpoint:
  enabled: {str(checkpoint_enabled).lower()}
  checkpoint_dir: {json.dumps(checkpoint_path)}
  model_save_format: safetensors
  save_consolidated: final
  save_optimizer: {str(save_optimizer).lower()}
  dequantize_base_checkpoint: true

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: {dataset_path}
  split: train
  shuffle_seed: 42
  truncation: true
  seq_length: {seq_length}
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


@app.function(image=automodel_image, timeout=1800)
def smoke_test_automodel():
    import importlib.util
    import subprocess

    import nemo_automodel
    import torch

    try:
        import tilelang
    except ImportError as exc:
        raise RuntimeError(f"TileLang import failed: {exc}") from exc

    print(f"nemo_automodel={nemo_automodel.__file__}")
    automodel_source_commit = subprocess.check_output(
        ["git", "-C", "/opt/Automodel", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    print(f"automodel_source_commit={automodel_source_commit}")
    torch_version = str(torch.__version__)
    cuda_version = str(torch.version.cuda)
    print(f"torch={torch_version}, cuda={cuda_version}")
    print(f"tilelang={tilelang.__file__}")
    network = _network_fingerprint()
    _select_uccl_ep_variant(network)
    uccl_spec = importlib.util.find_spec("uccl")
    uccl_path = uccl_spec.origin if uccl_spec is not None else None
    if uccl_spec is not None:
        import uccl.ep as uccl_ep

        print(f"uccl_ep={uccl_ep.__file__}")
    return {
        "automodel": nemo_automodel.__file__,
        "automodel_source_commit": automodel_source_commit,
        "torch": torch_version,
        "cuda": cuda_version,
        "tilelang": tilelang.__file__,
        "uccl": uccl_path,
    }


@app.function(
    image=vllm_image,
    volumes={HF_CACHE: hf_cache_vol},
    timeout=1800,
)
def smoke_test_vllm_lora_support():
    import vllm
    from transformers import AutoConfig
    from vllm.config import ModelConfig
    from vllm.model_executor.models.interfaces import SupportsLoRA
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM

    hf_cache_vol.reload()
    if vllm.__version__ != VLLM_VERSION:
        raise RuntimeError(f"Expected vLLM {VLLM_VERSION}, found {vllm.__version__}")
    if SupportsLoRA not in DeepseekV4ForCausalLM.__mro__:
        raise RuntimeError("DeepseekV4ForCausalLM does not advertise LoRA support")

    expected_packed_mapping = {
        "fused_wqa_wkv": ["q_a_proj", "kv_proj"],
    }
    if DeepseekV4ForCausalLM.lora_packed_modules_mapping != expected_packed_mapping:
        raise RuntimeError(
            "Unexpected DSv4 packed LoRA mapping: "
            f"{DeepseekV4ForCausalLM.lora_packed_modules_mapping}"
        )
    if DeepseekV4ForCausalLM.packed_modules_mapping:
        raise RuntimeError(
            "DSv4 LoRA mapping leaked into class-level quantization setup"
        )

    cached_config = AutoConfig.from_pretrained(HF_MODEL, trust_remote_code=True)
    cached_quant_config = getattr(cached_config, "quantization_config", None)
    fresh_config_dir = _prepare_vllm_hf_config()
    hf_config = AutoConfig.from_pretrained(
        fresh_config_dir,
        trust_remote_code=True,
    )
    raw_quant_config = getattr(hf_config, "quantization_config", None)
    model_config = ModelConfig(
        model=HF_MODEL,
        hf_config_path=str(fresh_config_dir),
        trust_remote_code=True,
        max_model_len=SERVE_MAX_MODEL_LEN,
    )
    model_quantization = model_config.quantization
    resolved_quant_config = getattr(
        model_config.hf_config,
        "quantization_config",
        None,
    )
    if model_quantization != "deepseek_v4_fp8":
        raise RuntimeError(
            f"Expected deepseek_v4_fp8 quantization, found {model_quantization}"
        )

    mapper = DeepseekV4ForCausalLM.hf_to_vllm_mapper.get_unstacked_mapper()
    cases = {
        "model.layers.0.self_attn.q_a_proj": ("model.layers.0.attn.q_a_proj"),
        "model.layers.0.self_attn.kv_proj": "model.layers.0.attn.kv_proj",
        "model.layers.0.self_attn.q_b_proj": "model.layers.0.attn.wq_b",
        "layers.0.attn.wq_a.weight": "model.layers.0.attn.wq_a.weight",
    }
    mapped = {source: mapper._map_name(source) for source in cases}
    if mapped != cases:
        raise RuntimeError(f"Unexpected DSv4 PEFT name mapping: {mapped}")

    result = {
        "vllm": vllm.__version__,
        "supports_lora": True,
        "packed_modules_mapping": expected_packed_mapping,
        "mapped_names": mapped,
        "cached_quantization_config": cached_quant_config,
        "model_quantization": model_quantization,
        "raw_quantization_config": raw_quant_config,
        "resolved_quantization_config": resolved_quant_config,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.function(
    image=automodel_image,
    gpu="H200:8",
    timeout=1800,
    retries=0,
    memory=HOST_MEMORY,
    cloud=MODAL_CLOUD,
    experimental_options={"efa_enabled": EFA_ENABLED},
    **EPHEMERAL_DISK_OPTIONS,
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def probe_uccl_ep_transport(
    tokens: int = 128,
    hidden_size: int = 2048,
    num_experts: int = 256,
    topk: int = 8,
    uccl_ib_gid_index: int = UCCL_IB_GID_INDEX,
):
    import subprocess

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else 1
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )
    total_gpus = n_nodes * GPUS_PER_NODE

    if num_experts % total_gpus != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by total_gpus={total_gpus}"
        )
    if topk <= 0 or topk > num_experts:
        raise ValueError(f"topk={topk} must be in [1, {num_experts}]")

    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET6"
    os.environ["NCCL_SOCKET_IFNAME"] = UCCL_SOCKET_IFNAME
    os.environ["GLOO_SOCKET_IFNAME"] = UCCL_SOCKET_IFNAME
    os.environ["UCCL_SOCKET_IFNAME"] = UCCL_SOCKET_IFNAME
    os.environ["UCCL_SOCKET_FAMILY"] = UCCL_SOCKET_FAMILY
    if uccl_ib_gid_index >= 0:
        os.environ["NCCL_IB_GID_INDEX"] = str(uccl_ib_gid_index)
        os.environ["UCCL_IB_GID_INDEX"] = str(uccl_ib_gid_index)
    os.environ["UCCL_PROBE_TOKENS"] = str(tokens)
    os.environ["UCCL_PROBE_HIDDEN_SIZE"] = str(hidden_size)
    os.environ["UCCL_PROBE_NUM_EXPERTS"] = str(num_experts)
    os.environ["UCCL_PROBE_TOPK"] = str(topk)
    os.environ["UCCL_PROBE_DIST_BACKEND"] = DIST_BACKEND
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = UCCL_CUDA_ALLOC_CONF
    os.environ["NCCL_MAX_NCHANNELS"] = NCCL_MAX_NCHANNELS
    os.environ["UCCL_EP_CPU_TIMEOUT_SECS"] = UCCL_EP_CPU_TIMEOUT_SECS

    network = _network_fingerprint()
    uccl_provider = _select_uccl_ep_variant(network)
    print(
        f"[node {node_rank}] network_fingerprint={json.dumps(network, sort_keys=True)}"
    )
    if node_rank == 0:
        print(
            "UCCL-EP transport probe: "
            f"nodes={n_nodes}, total_gpus={total_gpus}, gid_index={uccl_ib_gid_index}, "
            f"socket_ifname={UCCL_SOCKET_IFNAME}, dist_backend={DIST_BACKEND}, "
            f"socket_family={UCCL_SOCKET_FAMILY}, "
            f"cuda_alloc_conf={UCCL_CUDA_ALLOC_CONF}, "
            f"nccl_max_nchannels={NCCL_MAX_NCHANNELS}, "
            f"cpu_timeout_secs={UCCL_EP_CPU_TIMEOUT_SECS}"
        )

    probe_path = "/tmp/uccl_ep_transport_probe.py"
    Path(probe_path).write_text(UCCL_EP_PROBE_SOURCE)
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
        "29502",
        probe_path,
    ]
    print(f"[node {node_rank}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return {
        "nodes": n_nodes,
        "total_gpus": total_gpus,
        "tokens": tokens,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "topk": topk,
        "uccl_ib_gid_index": uccl_ib_gid_index,
        "uccl_provider": uccl_provider,
    }


@app.function(
    image=automodel_image,
    gpu="H200:8",
    volumes={HF_CACHE: hf_cache_vol, CHECKPOINTS_DIR: checkpoints_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
    retries=0,
    memory=HOST_MEMORY,
    cloud=MODAL_CLOUD,
    experimental_options={"efa_enabled": EFA_ENABLED},
    **EPHEMERAL_DISK_OPTIONS,
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train_h200_60k_smoke(
    run_id: str = "dsv4-flash-automodel-h200-16n-cp16-60k-smoke",
    seq_length: int = 60000,
    max_steps: int = 1,
    cp_size: int = 16,
    pp_size: int = 4,
    ep_size: int = 32,
    attn_backend: str = ATTN_BACKEND,
    moe_dispatcher: str = MOE_DISPATCHER,
    global_batch_size: int = 8,
    local_batch_size: int = 4,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES,
    synthetic_examples: int = 8,
    save_checkpoint: bool = False,
    save_optimizer: bool = False,
    nvshmem_ibgda_nic_handler: str = NVSHMEM_IBGDA_NIC_HANDLER,
    nvshmem_disable_cuda_vmm: int = 0,
    nvshmem_disable_p2p: int = NVSHMEM_DISABLE_P2P,
    uccl_ib_gid_index: int = UCCL_IB_GID_INDEX,
):
    import shutil
    import subprocess

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else 1
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )

    if n_nodes != N_NODES:
        print(f"[warn] decorator N_NODES={N_NODES}, runtime n_nodes={n_nodes}")

    total_gpus = n_nodes * GPUS_PER_NODE
    if total_gpus % (pp_size * cp_size) != 0:
        raise ValueError(
            f"pp_size*cp_size={pp_size * cp_size} must divide {total_gpus}"
        )
    dp_size = total_gpus // (pp_size * cp_size)
    if ep_size > (total_gpus // pp_size):
        raise ValueError(
            f"ep_size={ep_size} cannot exceed non-PP group size {total_gpus // pp_size}"
        )
    if ep_size > dp_size * cp_size:
        raise ValueError(f"ep_size={ep_size} cannot exceed dp*cp={dp_size * cp_size}")
    if nvshmem_ibgda_nic_handler not in {
        "auto",
        "gpu",
        "cpu",
        "cpu_cuda_memory",
        "cpu_host_memory",
    }:
        raise ValueError(
            "nvshmem_ibgda_nic_handler must be one of "
            "auto, gpu, cpu, cpu_cuda_memory, or cpu_host_memory"
        )
    if nvshmem_disable_cuda_vmm not in {0, 1}:
        raise ValueError("nvshmem_disable_cuda_vmm must be 0 or 1")
    if nvshmem_disable_p2p not in {0, 1}:
        raise ValueError("nvshmem_disable_p2p must be 0 or 1")
    if lora_rank < 0:
        raise ValueError("lora_rank must be non-negative; use 0 for full finetuning")
    if lora_rank > 0 and lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive when LoRA is enabled")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if global_batch_size <= 0 or local_batch_size <= 0:
        raise ValueError("global_batch_size and local_batch_size must be positive")
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path component")

    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
    os.environ["NCCL_DEBUG_FILE"] = "/tmp/nccl.%h.%p.log"
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET6"
    os.environ["NCCL_SOCKET_IFNAME"] = UCCL_SOCKET_IFNAME
    os.environ["NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"] = NVSHMEM_BOOTSTRAP_IFNAME
    os.environ["NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY"] = NVSHMEM_BOOTSTRAP_FAMILY
    os.environ["NVSHMEM_IBGDA_NIC_HANDLER"] = nvshmem_ibgda_nic_handler
    os.environ["NVSHMEM_DISABLE_CUDA_VMM"] = str(nvshmem_disable_cuda_vmm)
    os.environ["NVSHMEM_DISABLE_P2P"] = str(nvshmem_disable_p2p)
    os.environ["GLOO_SOCKET_IFNAME"] = UCCL_SOCKET_IFNAME
    os.environ["UCCL_SOCKET_IFNAME"] = UCCL_SOCKET_IFNAME
    os.environ["UCCL_SOCKET_FAMILY"] = UCCL_SOCKET_FAMILY
    if moe_dispatcher == "uccl_ep":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = UCCL_CUDA_ALLOC_CONF
        os.environ["NCCL_MAX_NCHANNELS"] = NCCL_MAX_NCHANNELS
        os.environ["UCCL_EP_CPU_TIMEOUT_SECS"] = UCCL_EP_CPU_TIMEOUT_SECS
    if uccl_ib_gid_index >= 0:
        os.environ["NCCL_IB_GID_INDEX"] = str(uccl_ib_gid_index)
        os.environ["UCCL_IB_GID_INDEX"] = str(uccl_ib_gid_index)
    hf_cache_vol.reload()
    if save_checkpoint:
        checkpoints_vol.reload()

    network = _network_fingerprint()
    uccl_provider = _select_uccl_ep_variant(network)
    print(
        f"[node {node_rank}] network_fingerprint={json.dumps(network, sort_keys=True)}"
    )

    dataset_path = "/tmp/dsv4_60k_synthetic/training.jsonl"
    effective_synthetic_examples = max(
        synthetic_examples, max_steps * global_batch_size
    )
    # Over-generate words; the tokenizer truncates to seq_length and pads to the
    # fixed length required by DSv4 CP.
    _write_synthetic_chat_jsonl(
        dataset_path,
        examples=effective_synthetic_examples,
        target_words=max(seq_length * 2, 1000),
    )

    checkpoint_dir = f"{CHECKPOINTS_DIR}/{run_id}" if save_checkpoint else None
    if checkpoint_dir is not None and Path(checkpoint_dir).exists():
        raise FileExistsError(f"Checkpoint directory already exists: {checkpoint_dir}")

    recipe_path = f"/tmp/{run_id}.yaml"
    Path(recipe_path).write_text(
        _recipe_yaml(
            dataset_path=dataset_path,
            seq_length=seq_length,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            local_batch_size=local_batch_size,
            cp_size=cp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            attn_backend=attn_backend,
            moe_dispatcher=moe_dispatcher,
            dist_backend=DIST_BACKEND,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            checkpoint_dir=checkpoint_dir,
            save_optimizer=save_optimizer,
        )
    )

    if node_rank == 0:
        automodel_source_commit = subprocess.check_output(
            ["git", "-C", "/opt/Automodel", "rev-parse", "HEAD"],
            text=True,
        ).strip()
        print("=" * 80)
        print("DeepSeek-V4-Flash AutoModel H200 60k smoke")
        print(f"run_id={run_id}")
        print(f"nodes={n_nodes}, total_gpus={total_gpus}")
        print(f"seq_length={seq_length}, max_steps={max_steps}")
        print(f"tp=1, dp={dp_size}, pp={pp_size}, cp={cp_size}, ep={ep_size}")
        print(f"attn_backend={attn_backend}")
        print(f"moe_dispatcher={moe_dispatcher}")
        print(f"dist_backend={DIST_BACKEND}")
        print(f"pytorch_cuda_alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
        print(f"nccl_max_nchannels={os.environ.get('NCCL_MAX_NCHANNELS')}")
        print(f"uccl_ep_cpu_timeout_secs={os.environ.get('UCCL_EP_CPU_TIMEOUT_SECS')}")
        print(f"requested_efa_enabled={EFA_ENABLED}, requested_cloud={MODAL_CLOUD}")
        print(f"runtime_cloud_provider={os.environ.get('MODAL_CLOUD_PROVIDER')}")
        print(
            f"nvshmem_bootstrap={NVSHMEM_BOOTSTRAP_IFNAME}/{NVSHMEM_BOOTSTRAP_FAMILY}"
        )
        print(f"nvshmem_ibgda_nic_handler={nvshmem_ibgda_nic_handler}")
        print(f"nvshmem_disable_cuda_vmm={nvshmem_disable_cuda_vmm}")
        print(f"nvshmem_disable_p2p={nvshmem_disable_p2p}")
        print(f"uccl_socket_ifname={UCCL_SOCKET_IFNAME}")
        print(f"uccl_socket_family={UCCL_SOCKET_FAMILY}")
        print(f"uccl_ib_gid_index={uccl_ib_gid_index}")
        print(
            f"global_batch_size={global_batch_size}, local_batch_size={local_batch_size}"
        )
        print(
            f"lora_rank={lora_rank}, lora_alpha={lora_alpha}, "
            f"lora_target_modules={lora_target_modules if lora_rank > 0 else 'disabled'}"
        )
        print(f"synthetic_examples={effective_synthetic_examples}")
        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"save_optimizer={save_optimizer}")
        print(f"automodel_commit={AUTOMODEL_COMMIT}")
        print(f"automodel_source_commit={automodel_source_commit}")
        print(f"recipe={recipe_path}")
        print("=" * 80)
        print(Path(recipe_path).read_text())
        print("=" * 80)

    automodel_bin = shutil.which("automodel")
    if automodel_bin is None:
        raise RuntimeError("automodel CLI not found in image")

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
    result = subprocess.run(cmd, capture_output=False, text=True)
    if node_rank == 0:
        _print_nccl_network_logs(node_rank)
    if result.returncode != 0:
        _print_cgroup_memory_status(node_rank)
        raise RuntimeError(f"AutoModel smoke failed with code {result.returncode}")
    if save_checkpoint:
        checkpoints_vol.commit()

    return {
        "run_id": run_id,
        "seq_length": seq_length,
        "max_steps": max_steps,
        "nodes": n_nodes,
        "total_gpus": total_gpus,
        "cp_size": cp_size,
        "pp_size": pp_size,
        "ep_size": ep_size,
        "attn_backend": attn_backend,
        "moe_dispatcher": moe_dispatcher,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_modules": lora_target_modules if lora_rank > 0 else None,
        "checkpoint_dir": checkpoint_dir,
        "save_optimizer": save_optimizer,
        "nvshmem_disable_p2p": nvshmem_disable_p2p,
        "uccl_provider": uccl_provider,
    }


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
def finalize_lora_checkpoint(
    run_id: str,
    expected_step: int = 4,
    expected_pp_size: int = 4,
    expected_layers: int = 43,
    expected_lora_rank: int = 64,
    expected_lora_alpha: int = 64,
):
    """Merge PP-local LoRA shards after all training nodes commit the volume."""
    import hashlib
    import re

    import torch
    from safetensors.torch import load_file, save_file

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path component")

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
    cloud=MODAL_CLOUD,
    experimental_options={"efa_enabled": EFA_ENABLED},
    **EPHEMERAL_DISK_OPTIONS,
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
    cloud=MODAL_CLOUD,
    experimental_options={"efa_enabled": EFA_ENABLED},
    **EPHEMERAL_DISK_OPTIONS,
)
def validate_lora_serving(
    run_id: str,
    checkpoint_step: int = 4,
    max_model_len: int = 64 * 1024,
    long_prompt_tokens: int = 60000,
    startup_timeout_seconds: int = 3600,
):
    """Load the finalized adapter in vLLM and generate at 60k context."""
    import signal
    import subprocess
    import time
    import traceback
    import urllib.error
    import urllib.request

    import vllm
    from huggingface_hub import snapshot_download

    if long_prompt_tokens <= 0:
        raise ValueError("long_prompt_tokens must be positive")
    if long_prompt_tokens + 1 > max_model_len:
        raise ValueError(
            "max_model_len must leave room for one generated token: "
            f"{long_prompt_tokens + 1} > {max_model_len}"
        )

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
        if VLLM_ADAPTER_NAME not in model_ids:
            raise RuntimeError(
                f"Adapter is absent from /v1/models: {sorted(model_ids)}"
            )

        chat = request_json(
            "/v1/chat/completions",
            {
                "model": VLLM_ADAPTER_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2 + 2? Return only the integer.",
                    }
                ],
                "max_tokens": 16,
                "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            },
            timeout=600,
        )
        if chat.get("model") != VLLM_ADAPTER_NAME or not chat.get("choices"):
            raise RuntimeError(f"Invalid adapter chat response: {chat}")
        message = chat["choices"][0].get("message", {})
        chat_text = message.get("content") or message.get("reasoning_content")
        if not chat_text:
            raise RuntimeError(f"Adapter chat returned no text: {chat}")

        completion = request_json(
            "/v1/completions",
            {
                "model": VLLM_ADAPTER_NAME,
                "prompt": [0] * long_prompt_tokens,
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=3600,
        )
        usage = completion.get("usage", {})
        if completion.get("model") != VLLM_ADAPTER_NAME:
            raise RuntimeError(
                f"Long-context response used the wrong model: {completion}"
            )
        if usage.get("prompt_tokens") != long_prompt_tokens:
            raise RuntimeError(
                f"Long-context request did not process the requested prompt: {usage}"
            )
        if usage.get("completion_tokens") != 1 or not completion.get("choices"):
            raise RuntimeError(f"Long-context generation failed: {completion}")

        result = {
            "adapter_name": VLLM_ADAPTER_NAME,
            "adapter_sha256": preflight["adapter_sha256"],
            "chat_output": str(chat_text)[:200],
            "checkpoint_step": checkpoint_step,
            "long_context_finish_reason": completion["choices"][0].get("finish_reason"),
            "long_context_usage": usage,
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
def h200_16node_60k_lora_5step(
    run_id: str,
):
    if N_NODES != 16:
        raise ValueError(
            f"This validated entrypoint requires N_NODES=16, found {N_NODES}"
        )
    call = train_h200_60k_smoke.spawn(
        run_id=run_id,
        max_steps=5,
        lora_rank=64,
        lora_alpha=64,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        synthetic_examples=40,
        save_checkpoint=True,
        save_optimizer=False,
    )
    call_id = (
        getattr(call, "object_id", None)
        or getattr(call, "function_call_id", None)
        or str(call)
    )
    result = {
        "function_call_id": call_id,
        "run_id": run_id,
        "max_steps": 5,
        "checkpoint_dir": f"{CHECKPOINTS_DIR}/{run_id}",
        "nodes": N_NODES,
    }
    print(json.dumps(result, indent=2))
    return result
