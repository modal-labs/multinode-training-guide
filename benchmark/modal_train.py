import os

import modal
import modal.experimental

app = modal.App("multinode-benchmark-2-nodes")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CODE_DIR = "/root/"
REMOTE_BENCH_SCRIPT_PATH = "/root/train.py"

N_NODES = 2
N_PROC_PER_NODE = 8

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc15",
        "libnl-route-3-200",
    )
    .uv_pip_install(
        "torch==2.6.0",
        "numpy",
        "importlib-metadata",
    )
    .add_local_dir(
        LOCAL_CODE_DIR,
        remote_path=REMOTE_CODE_DIR,
    )
)

@app.function(
    gpu="H100:8",
    image=image,
    # Enabling EFA unlocks extra capacity of H100 clusters. This only requires that the container
    # has a CUDA runtime installed and packages libhwloc15 and libnl-route-3-200.
    experimental_options={
        "efa_enabled": True,
    },
    timeout=60 * 60, # 1 hour
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def run_benchmark():
    """Run a simple benchmark script that passes around a tensor of size 500000x2000."""
    from torch.distributed.run import parse_args, run

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
