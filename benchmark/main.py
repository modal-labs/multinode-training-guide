import os

import modal
import modal.experimental

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CODE_DIR = "/root/"
REMOTE_BENCH_SCRIPT_PATH = "/root/benchmark.py"

N_NODES = 2
N_PROC_PER_NODE = 8

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
    )
    .pip_install(
        "torch",
        "numpy",
        "importlib-metadata",
    )
    .add_local_dir(
        LOCAL_CODE_DIR,
        remote_path=REMOTE_CODE_DIR,
    )
)

app = modal.App("multinode-benchmark")


def export_rdma_env():
    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.environ.get('LD_LIBRARY_PATH', '')}:/usr/local/lib"
    )
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["LOGLEVEL"] = "DEBUG"
    os.environ["NCCL_IB_SPLIT_DATA_ON_QPS"] = "0"
    os.environ["NCCL_IB_QPS_PER_CONNECTION"] = "4"
    os.environ["NCCL_IB_TC"] = "41"
    os.environ["NCCL_IB_SL"] = "0"
    os.environ["NCCL_IB_TIMEOUT"] = "22"

    # Control‑plane (TCP) — stays on eth1, uses IPv6
    os.environ["NCCL_SOCKET_IFNAME"] = "eth1"
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET6"

    # Data‑plane (RDMA) — stays on the HCA ports, uses IPv4
    os.environ["NCCL_IB_ADDR_FAMILY"] = "AF_INET"
    os.environ["NCCL_IB_GID_INDEX"] = "3"  # OCI's IPv4‑mapped GID index
    os.environ["NCCL_IB_HCA"] = (
        "=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_9,mlx5_10,mlx5_11,mlx5_12"
    )
    os.environ["NCCL_IB_MERGE_NICS"] = "0"


@app.function(
    gpu="H100:8",
    cloud="oci",
    region="us-chicago-1",
    image=image,
    experimental_options={"rdma_enabled": "1"},
)
@modal.experimental.clustered(size=N_NODES)
def run_benchmark():
    """Run a simple benchmark script that passes around a tensor of size 500000x2000."""

    from torch.distributed.run import parse_args, run

    export_rdma_env()

    cluster_info = modal.experimental.get_cluster_info()
    # which container am I?
    container_rank: int = cluster_info.rank
    # what's the leader/master/main container's address?
    main_ip_addr: str = cluster_info.container_ips[0]
    container_id = os.environ["MODAL_TASK_ID"]

    print(f"hello from {container_id}, rank {container_rank} of {N_NODES}")
    if container_rank == 0:
        print(f"main container's address: {main_ip_addr}")

    args = [
        f"--nnodes={N_NODES}",
        f"--nproc-per-node={N_PROC_PER_NODE}",
        f"--node-rank={cluster_info.rank}",
        f"--master-addr={main_ip_addr}",
        REMOTE_BENCH_SCRIPT_PATH,
    ]
    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))


@app.local_entrypoint()
def main():
    run_benchmark.remote()
