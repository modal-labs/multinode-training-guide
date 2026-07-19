"""Container images for DeepSeek-V4-Flash training and serving."""

from pathlib import Path

import modal

AUTOMODEL_IMAGE = "nvcr.io/nvidia/nemo-automodel:26.06.00"
AUTOMODEL_COMMIT = "1197b6281255957cc2c58e79d50796c1b256c57c"
UCCL_EP_COMMIT = "66170bc299205228f0170bc1638594a39af9ffd5"
UCCL_EFA_INSTALLER_VERSION = "1.42.0"
VLLM_IMAGE = "vllm/vllm-openai:v0.25.1"

REMOTE_TRAIN_RECIPE = "/root/train_recipe.yaml"
REMOTE_CHECKPOINT_HELPER = "/root/dsv4_checkpoint.py"


def build_automodel_image(source_dir: Path) -> modal.Image:
    patches_dir = source_dir / "patches"
    uccl_ipv6_patch = patches_dir / "uccl_ipv6_oob.patch"
    automodel_patches = tuple(
        patches_dir / name
        for name in (
            "automodel_uccl_teardown.patch",
            "automodel_checkpoint_dequant.patch",
            "automodel_composite_backend.patch",
            "automodel_pp_peft_checkpoint.patch",
        )
    )

    # This commit provides the DeepSeek-V4 context-parallel implementation.
    image = (
        modal.Image.from_registry(AUTOMODEL_IMAGE)
        .entrypoint([])
        .run_commands(
            "cd / && rm -rf /opt/Automodel && "
            "git clone https://github.com/NVIDIA-NeMo/Automodel.git "
            "/opt/Automodel && "
            f"cd /opt/Automodel && git checkout {AUTOMODEL_COMMIT} && "
            "pip install --no-deps --force-reinstall -e /opt/Automodel"
        )
        .run_commands(
            "pip install --no-deps --force-reinstall "
            "'tilelang>=0.1.11' 'tile-kernels==1.0.0' "
            "'apache-tvm-ffi<=0.1.11'"
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

    # EFA-enabled scheduling can place a job on either EFA or Mellanox/RoCE.
    image = (
        image.apt_install(
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
        .add_local_file(
            str(uccl_ipv6_patch),
            "/tmp/uccl_ipv6_oob.patch",
            copy=True,
        )
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

    for patch_path in automodel_patches:
        image = image.add_local_file(
            str(patch_path),
            f"/tmp/{patch_path.name}",
            copy=True,
        )
    return (
        image.run_commands(
            "cd /opt/Automodel && "
            + " && ".join(
                f"git apply /tmp/{patch_path.name}" for patch_path in automodel_patches
            )
        )
        .add_local_file(
            str(source_dir / "train_recipe.yaml"),
            REMOTE_TRAIN_RECIPE,
            copy=True,
        )
        .add_local_file(
            str(source_dir / "dsv4_checkpoint.py"),
            REMOTE_CHECKPOINT_HELPER,
            copy=True,
        )
        .add_local_file(
            str(source_dir / "dsv4_images.py"),
            "/root/dsv4_images.py",
            copy=True,
        )
    )


def build_vllm_image(source_dir: Path) -> modal.Image:
    lora_patch = source_dir / "patches" / "vllm_deepseek_v4_lora.patch"
    return (
        modal.Image.from_registry(VLLM_IMAGE)
        .entrypoint([])
        .run_commands("ln -sf $(which python3) /usr/local/bin/python")
        .apt_install("patch")
        .add_local_file(
            str(lora_patch),
            f"/tmp/{lora_patch.name}",
            copy=True,
        )
        .run_commands(
            "cd $(python -c 'import pathlib, vllm; "
            "print(pathlib.Path(vllm.__file__).resolve().parent.parent)') && "
            f"patch --batch --forward -p1 < /tmp/{lora_patch.name}"
        )
        .add_local_file(
            str(source_dir / "dsv4_checkpoint.py"),
            REMOTE_CHECKPOINT_HELPER,
            copy=True,
        )
        .add_local_file(
            str(source_dir / "dsv4_images.py"),
            "/root/dsv4_images.py",
            copy=True,
        )
        .env(
            {
                "HF_XET_HIGH_PERFORMANCE": "1",
                "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
            }
        )
    )
