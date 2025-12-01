import os

import modal
import modal.experimental

LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CODE_DIR = "/root/"
REMOTE_BENCH_SCRIPT_PATH = "/root/train.py"

N_NODES = 2
N_PROC_PER_NODE = 8

EFA_INSTALLER_VERSION = "1.44.0"
AWS_OFI_NCCL_VERSION = "1.17.2"
INSTALL_DIR = "/tmp"

image = (
    modal.Image.debian_slim()
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "wget",
        "ca-certificates",
        "curl",
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
    )
    .apt_install(
        "build-essential",
        "devscripts",
        "debhelper",
        "check",
        "libsubunit-dev",
        "fakeroot",
        "pkg-config",
        "dkms",
        "libhwloc-dev",
        "cuda-toolkit-12-9",
    )
    .run_commands(
        # Install AWS EFA userspace libraries.
        f"""cd {INSTALL_DIR} && \
        curl -O https://efa-installer.amazonaws.com/aws-efa-installer-{EFA_INSTALLER_VERSION}.tar.gz && \
        tar -xf {INSTALL_DIR}/aws-efa-installer-{EFA_INSTALLER_VERSION}.tar.gz && \
        cd aws-efa-installer && \
        ./efa_installer.sh -y -d --skip-kmod --skip-limit-conf --no-verify
        """,
    )
    .run_commands(
        # Install AWS OFI NCCL libraries.
        f"""cd {INSTALL_DIR} && \
        wget https://github.com/aws/aws-ofi-nccl/releases/download/v{AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-{AWS_OFI_NCCL_VERSION}.tar.gz && \
        tar xf {INSTALL_DIR}/aws-ofi-nccl-{AWS_OFI_NCCL_VERSION}.tar.gz && \
        cd aws-ofi-nccl-{AWS_OFI_NCCL_VERSION} && \
        ./configure --with-libfabric=/opt/amazon/efa/ --with-cuda=/usr/local/cuda --prefix=/opt/amazon/ofi-nccl --disable-nccl-net-symlink && \
        make && \
        make install
        """,
    )
    .run_commands(
        # Remove EFA and OFI libraries from the default library path so they're not used on non-EFA hardware.
        "rm -f /etc/ld.so.conf.d/000_efa.conf",
        "rm -f /etc/ld.so.conf.d/100_ofinccl.conf",
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


app = modal.App("multinode-benchmark")

def _run_benchmark():
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


@app.function(
    gpu="H100:8",
    image=image,
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def run_benchmark():
    """Run a benchmark on Infiniband instances, testing that the EFA image works on non-EFA hardware."""
    _run_benchmark()


@app.function(
    gpu="H100:8",
    image=image,
    experimental_options={
        "efa_enabled": True,
    },
    timeout=60 * 60 * 6,
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def run_benchmark_efa():
    """Run a benchmark on EFA instances, testing that the EFA image works on EFA hardware."""

    if os.environ.get("MODAL_CLOUD_PROVIDER") == "CLOUD_PROVIDER_AWS":
        os.environ["LD_LIBRARY_PATH"] = "/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib"
        os.environ["NCCL_NET_PLUGIN"] = "ofi"

    _run_benchmark()
