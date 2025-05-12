import modal
import modal.experimental

import os

DATASET_ID = "bigcode/starcoderdata"

data_vol = modal.Volume.from_name(
    f"{DATASET_ID.replace('/', '-')}-dataset",
    create_if_missing=True,
)
DATASET_MOUNT_PATH = "/data"

model_vol = modal.Volume.from_name(
    "starcoder-model",
    create_if_missing=True,
)
MODEL_MOUNT_PATH = "/model"

hf_secret = modal.Secret.from_name("huggingface-token")

app = modal.App(
    f"{DATASET_ID}-train",
)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libibverbs-dev", "libibverbs1")
    .pip_install(
        "datasets>=2.19",
        "sympy",
        "transformers",
        "trl",
        "wandb",
        "huggingface_hub",
        "torch",
        "accelerate",
    )
)

LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CODE_DIR = "/root/"
REMOTE_TRAIN_SCRIPT_PATH = "/root/train.py"

image = base_image.add_local_dir(
    LOCAL_CODE_DIR,
    remote_path=REMOTE_CODE_DIR,
)

# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
n_nodes = 2
# Typically this matches the number of GPUs per container.
n_proc_per_node = 8

main_port = 29500


def _train_multi_node() -> None:
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

    global_batch_size = 256
    per_device_batch_size = 4
    grad_accum = global_batch_size // (
        n_proc_per_node * n_nodes * per_device_batch_size
    )

    if container_rank == 0:
        wandb_project = f"{DATASET_ID.replace('/', '-')}-training"
        wandb_run_name = f"starcoder-nodes_{n_nodes}-gpus_{n_proc_per_node}"
        wandb_args = [
            "--wandb_project",
            wandb_project,
            "--wandb_run_name",
            wandb_run_name,
        ]
    else:
        wandb_args = []

    args = [
        f"--nnodes={n_nodes}",
        f"--nproc-per-node={n_proc_per_node}",
        f"--node-rank={cluster_info.rank}",
        f"--master-addr={main_ip_addr}",
        f"--master-port={main_port}",
        REMOTE_TRAIN_SCRIPT_PATH,
        "--data_dir",
        DATASET_MOUNT_PATH,
        "--output_dir",
        MODEL_MOUNT_PATH,
        "--epochs",
        "2",
        "--batch_per_device",
        str(per_device_batch_size),
        "--grad_accum",
        str(grad_accum),
        "--buffer_size",
        "20000",
        *wandb_args,
    ]
    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))


@app.function(
    image=image,
    volumes={
        DATASET_MOUNT_PATH: data_vol,
        MODEL_MOUNT_PATH: model_vol,
    },
    secrets=[
        # Required for connecting to Weights & Biases from within the Modal container.
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-token"),
    ],
    cloud="oci",
    gpu="H100:8",
    timeout=60 * 60 * 24,
)
@modal.experimental.clustered(n_nodes, rdma=True)
def train_multi_node():
    _train_multi_node()


@app.local_entrypoint()
def main():
    train_multi_node.remote()
