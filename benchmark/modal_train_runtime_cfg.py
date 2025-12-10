import argparse
import dataclasses
import enum
import os
from typing import Union

import modal
import modal.experimental

cuda_version = "12.9.1"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu24.04"
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
    )
    .uv_pip_install(
        "torch==2.9.1", "numpy", "importlib-metadata", "nvidia-cudnn-cu12>=9.0.15"
    )
    .add_local_dir(
        LOCAL_CODE_DIR,
        remote_path=REMOTE_CODE_DIR,
    )
)

app = modal.App("multinode-benchmark")


class ModalGPU(enum.StrEnum):
    H100 = "H100"
    H200 = "H200"
    B200 = "B200"


@dataclasses.dataclass
class ModalClusterConfig:
    num_nodes: int
    gpus_per_node: int
    gpu_type: Union[str, ModalGPU] = ModalGPU.H100

    def __post_init__(self):
        if isinstance(self.gpu_type, str):
            try:
                self.gpu_type = ModalGPU(self.gpu_type)
            except ValueError:
                valid_gpu_types = ", ".join([f"'{g.value}'" for g in ModalGPU])
                raise ValueError(
                    f"Invalid GPU type '{self.gpu_type}'. Must be one of: {valid_gpu_types}"
                )

    def gpu_str(self):
        return f"{self.gpu_type}:{self.gpus_per_node}"


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


def build_benchmark(cfg: ModalClusterConfig):
    # additionally, could assign a different image build for hopper vs. blackwell
    # or perform other hardware-specific setup/configuration as needed

    wrapped_runner = app.function(
        gpu=cfg.gpu_str(),
        image=image,
    )(modal.experimental.clustered(size=cfg.num_nodes, rdma=True)(run_benchmark))

    return wrapped_runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multinode benchmark")
    parser.add_argument("num_nodes", type=int, help="Number of nodes in the cluster")
    parser.add_argument("gpus_per_node", type=int, help="Number of GPUs per node")
    parser.add_argument("--gpu-type", type=str, default=None, help="GPU type to use")

    args = parser.parse_args()

    gpu = ModalGPU(args.gpu_type) if args.gpu_type is not None else ModalGPU("H100")
    cluster_config = ModalClusterConfig(
        num_nodes=args.num_nodes, gpus_per_node=args.gpus_per_node, gpu_type=gpu
    )
    run_benchmark = build_benchmark(cluster_config)

    with modal.enable_output():
        with app.run(detach=True):
            run_benchmark.remote()
