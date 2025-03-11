import os
import subprocess

import modal
import modal.experimental

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CODE_DIR = "/root/"
REMOTE_TRAIN_SCRIPT_PATH = "/root/train.py"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .pip_install(
        "click", # Required by Lightning 'fabric' CLI
        "torch", 
        "lightning~=2.4.0", 
        "requests", # Required by Lightning demo code
    )
    .add_local_dir(
        LOCAL_CODE_DIR,
        remote_path=REMOTE_CODE_DIR,
    )
)
app = modal.App("lightning-demo", image=image)
volume = modal.Volume.from_name("lightning-multinode-demo", create_if_missing=True)
volume_model_output = modal.Volume.from_name(
    "lightning-multinode-demo-model-output", create_if_missing=True
)


# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
n_nodes = 2
# Typically this matches the number of GPUs per container.
n_proc_per_node = 8

def hours_in_seconds(hours: int) -> int:
    return hours * 60 * 60

@app.function(
    gpu=f"A100:{n_proc_per_node}",
    volumes={
        "/vol": volume,
        "/root/out": volume_model_output,
    },
    timeout=hours_in_seconds(1),
)
def train_single_node():
    """
    Train the model on a single node (a.k.a container) with N GPUs.
    """
    fabric_args = [
        "fabric",
        "run",
        "--accelerator=gpu",
        "--strategy=ddp",
        f"--devices={n_proc_per_node}",
        REMOTE_TRAIN_SCRIPT_PATH
    ]
    print(f"Running Lightning Fabric with args: {' '.join(fabric_args)}")
    subprocess.run(fabric_args, check=True)

@app.function(
    gpu=f"H100:{n_proc_per_node}",
    volumes={
        "/root/data": volume,  # 
        "/root/out": volume_model_output,
    },
    timeout=hours_in_seconds(1),
)
@modal.experimental.clustered(n_nodes)
def train_multi_node():
    """
    Train the model on a multi-node cluster with N GPUs per node (typically 8).
    Good cluster scale performance should result in a ~linear speedup as the number of nodes
    is increased.
    """

    cluster_info = modal.experimental.get_cluster_info()
    # which container am I?
    container_rank: int = cluster_info.rank
    # what's the leader/master/main container's address?
    main_ip_addr: str = cluster_info.container_ips[0]
    container_id = os.environ["MODAL_TASK_ID"]

    print(f"hello from {container_id}, rank {container_rank} of {n_nodes}")
    if container_rank == 0:
        print(f"main container's address: {main_ip_addr}")

    fabric_args = [
        "fabric",
        "run",
        "--accelerator=gpu",
        "--strategy=ddp",
        f"--devices={n_proc_per_node}",
        f"--num-nodes={n_nodes}",
        f"--node-rank={cluster_info.rank}",
        f"--main-address={main_ip_addr}",
        REMOTE_TRAIN_SCRIPT_PATH
    ]
    print(f"Running Lightning Fabric with args: {' '.join(fabric_args)}")
    subprocess.run(fabric_args, check=True)
