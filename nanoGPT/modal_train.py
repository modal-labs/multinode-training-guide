import os
import shutil
import subprocess

import modal
import modal.experimental

# Instructions for install flash-attn taken from this Modal guide doc:
# https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CODE_DIR = "/root/"
REMOTE_TRAIN_SCRIPT_PATH = "/root/train.py"
REMOTE_BENCH_SCRIPT_PATH = "/root/bench.py"
GPU_TYPE = "H100"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    # flash-attn has an undeclared dependency on PyPi packages 'torch' and 'packaging',
    # as well as git, requiring that we annoyingly install without it the first time.
    #
    # ref: https://github.com/astral-sh/uv/issues/6437#issuecomment-2535324784
    .apt_install("git")
    # https://github.com/karpathy/nanoGPT?tab=readme-ov-file#install
    # TODO: why doesn't karpathy pin these?
    .pip_install("torch", "transformers", "datasets", "tiktoken", "wandb", "tqdm")
    .add_local_dir(
        LOCAL_CODE_DIR,
        remote_path=REMOTE_CODE_DIR,
    )
)
app = modal.App("nanoGPT", image=image)
volume = modal.Volume.from_name("nanogpt-multinode-demo", create_if_missing=True)
volume_model_output = modal.Volume.from_name(
    "nanogpt-multinode-demo-model-output", create_if_missing=True
)


# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
n_nodes = 2
# Typically this matches the number of GPUs per container.
n_proc_per_node = 8

MOUNTS = []


@app.function(
    mounts=MOUNTS,
    timeout=3600,
    cpu=(0.2, 16),  # Set higher limit to avoid CPU bottleneck.
    volumes={
        "/vol": volume,
    },
)
def prepare_data():
    os.environ["TRUST_REMOTE_CODE"] = "true"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
    subprocess.run(["python", "/root/data/openwebtext/prepare.py"], check=True)
    print("Copying train.bin to modal.Volume for persistent storage...")
    shutil.copy("/root/data/openwebtext/train.bin", "/vol/train.bin")
    shutil.copy("/root/data/openwebtext/val.bin", "/vol/val.bin")


@app.function(
    gpu=f"{GPU_TYPE}:{n_proc_per_node}",
    mounts=MOUNTS,
    secrets=[
        # Required for connecting to Weights & Biases from within the Modal container.
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/vol": volume,
        # Mount a Volume where NanoGPT outputs checkpoints.
        "/root/out": volume_model_output,
    },
    timeout=60 * 60 * 24,
)
@modal.experimental.clustered(n_nodes)
def train_multi_node():
    """
    Train the model on a multi-node cluster with N GPUs per node (typically 8).
    Good cluster scale performance should result in a ~linear speedup as the number of nodes
    is increased.
    """
    from torch.distributed.run import parse_args, run

    cluster_info = modal.experimental.get_cluster_info()
    # which container am I?
    container_rank: int = cluster_info.rank
    # what's the leader/master/main container's address?
    main_ip_addr: str = cluster_info.container_ips[0]
    container_id = os.environ["MODAL_TASK_ID"]

    print(f"hello from {container_id}, rank {container_rank} of {n_nodes}")
    if container_rank == 0:
        print(f"main container's address: {main_ip_addr}")

    # "In particular, if you don't have Infiniband then also prepend ..."
    # As of Feb 2025 Modal does not (yet) support Infiniband.
    os.environ["NCCL_IB_DISABLE"] = "1"

    # Symlink the training data in our volume to the place that nanoGPT expects it.
    os.symlink("/vol/train.bin", "/root/data/openwebtext/train.bin")
    os.symlink("/vol/val.bin", "/root/data/openwebtext/val.bin")
    args = [
        f"--nnodes={n_nodes}",
        f"--nproc-per-node={n_proc_per_node}",
        f"--node-rank={cluster_info.rank}",
        f"--master-addr={main_ip_addr}",
        REMOTE_TRAIN_SCRIPT_PATH,
    ]
    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))


@app.function(
    gpu=f"A100:{n_proc_per_node}",
    mounts=MOUNTS,
    secrets=[
        # Required for connecting to Weights & Biases from within the Modal container.
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/vol": volume,
        # Mount a Volume where NanoGPT outputs checkpoints.
        "/root/out": volume_model_output,
    },
    timeout=60 * 60 * 24,
)
def train_single_node():
    """
    Train the model on a single node (a.k.a container) with N GPUs.
    Training on a single 8x A100 container is a useful baseline for performance comparison
    because it is the original training configuration of the karpathy/nanoGPT repository.
    """
    from torch.distributed.run import parse_args, run

    # Symlink the training data in our volume to the place that nanoGPT expects it.
    os.symlink("/vol/train.bin", "/root/data/openwebtext/train.bin")
    os.symlink("/vol/val.bin", "/root/data/openwebtext/val.bin")
    args = [
        f"--nproc-per-node={n_proc_per_node}",
        REMOTE_TRAIN_SCRIPT_PATH,
    ]
    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))


@app.function(
    gpu=GPU_TYPE,
    mounts=MOUNTS,
    volumes={
        "/vol": volume,
        # Mount a Volume where NanoGPT outputs checkpoints.
        "/root/out": volume_model_output,
    },
)
def bench(profile: bool = False):
    """
    Run the benchmark script, which profiles the performance of the model forward/backward pass
    on a single GPU.
    """
    from torch.distributed.run import parse_args, run

    if profile:
        os.environ["NANOGPT_PROFILE"] = "1"

    # Symlink the training data in our volume to the place that nanoGPT expects it.
    os.symlink("/vol/train.bin", "/root/data/openwebtext/train.bin")
    os.symlink("/vol/val.bin", "/root/data/openwebtext/val.bin")
    args = [
        "--nproc-per-node=1",
        REMOTE_BENCH_SCRIPT_PATH,
    ]
    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))
