import modal
import modal.experimental
import os
import subprocess
from enum import Enum

from common import (
    DATASET_ID,
    DATASET_VOLUME_NAME,
    DATASET_MOUNT_PATH,
    MODEL_MOUNT_PATH,
    MODEL_CACHE_DIR,
    train_image,
    REMOTE_TRAIN_SCRIPT_PATH,
    MODEL_VOLUME_NAME,
)

data_vol = modal.Volume.from_name(
    DATASET_VOLUME_NAME,
    create_if_missing=True,
)

model_vol = modal.Volume.from_name(
    MODEL_VOLUME_NAME,
    create_if_missing=True,
)

hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")

app = modal.App(
    f"{DATASET_ID}-train",
)


# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 4.
n_nodes = 2
# Typically this matches the number of GPUs per container.
n_proc_per_node = 8
# Port used for inter-container control communication.
main_port = 29500


class LaunchType(Enum):
    TORCHRUN = "torchrun"
    ACCELERATE = "accelerate"


@app.function(
    image=train_image,
    volumes={
        DATASET_MOUNT_PATH: data_vol,
        MODEL_MOUNT_PATH: model_vol,
    },
    secrets=[
        wandb_secret,
        hf_secret,
    ],
    gpu="H100:8",
    timeout=60 * 60 * 24,
)
@modal.experimental.clustered(n_nodes, rdma=True)
def train_multi_node(launch_type: str = "torchrun", profile: bool = False):
    """
    Performs multi-node training using either torchrun or Hugging Face Accelerate.
    Launch type can be either 'torchrun' or 'accelerate'.
    """
    # Parse the launch_type string into the LaunchType enum
    parsed_launch_type: LaunchType
    if launch_type.lower() == "torchrun":
        parsed_launch_type = LaunchType.TORCHRUN
    elif launch_type.lower() == "accelerate":
        parsed_launch_type = LaunchType.ACCELERATE
    else:
        raise ValueError(
            f"Invalid launch_type: '{launch_type}'. Must be 'torchrun' or 'accelerate'."
        )

    # Get Modal cluster info for inter-container communication
    cluster_info = modal.experimental.get_cluster_info()
    container_rank: int = cluster_info.rank
    main_ip_addr: str = cluster_info.container_ips[0]
    container_id = os.environ["MODAL_TASK_ID"]

    # Configuration for batch sizes and gradient accumulation. Target a constant
    # global batch size so we can do apples-to-apples comparisons between runs.
    global_batch_size_config = 1024
    per_device_batch_size_config = 16
    grad_accum_config = global_batch_size_config // (
        n_proc_per_node * n_nodes * per_device_batch_size_config
    )

    # Unified run name for output directories and W&B
    current_run_name = (
        f"starcoder-nodes_{n_nodes}-gpus_{n_proc_per_node}"
        f"-batch_{global_batch_size_config}-per_device_{per_device_batch_size_config}"
        f"-grad_accum_{grad_accum_config}"
    )

    print(
        f"Hello from {container_id}, rank {container_rank} of {n_nodes} "
        f"using {parsed_launch_type.value}. Run ID: {current_run_name}"
    )

    if container_rank == 0:
        print(f"Main container's IP address: {main_ip_addr}")
        # Setup W&B environment variables on the main node
        wandb_project_name = f"{DATASET_ID.replace('/', '-')}-training"
        os.environ["WANDB_PROJECT"] = wandb_project_name
        os.environ["WANDB_RUN_NAME"] = current_run_name
        print(
            f"Weights & Biases: Project='{wandb_project_name}', Run='{current_run_name}'"
        )

    script_args = [
        "--data_dir",
        DATASET_MOUNT_PATH,
        "--output_dir",
        f"{MODEL_MOUNT_PATH}/{current_run_name}",
        "--epochs",
        "2",
        "--batch_per_device",
        str(per_device_batch_size_config),
        "--grad_accum",
        str(grad_accum_config),
        "--model_cache_dir",
        MODEL_CACHE_DIR,
    ]
    if profile:
        script_args.append("--profile")

    def _train_torchrun() -> None:
        from torch.distributed.run import parse_args, run

        args = [
            f"--nnodes={n_nodes}",
            f"--nproc-per-node={n_proc_per_node}",
            f"--node-rank={container_rank}",
            f"--master-addr={main_ip_addr}",
            f"--master-port={main_port}",
            REMOTE_TRAIN_SCRIPT_PATH,
            *script_args,
        ]
        print(f"Executing torchrun with args: {' '.join(args)}")
        run(parse_args(args))

    def _train_accelerate() -> None:
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(n_proc_per_node),
            "--num_machines",
            str(n_nodes),
            "--machine_rank",
            str(container_rank),
            "--main_process_ip",
            main_ip_addr,
            "--main_process_port",
            str(main_port),
            "--mixed_precision",
            "bf16",
            REMOTE_TRAIN_SCRIPT_PATH,
            *script_args,
        ]
        print(f"Executing accelerate launch with: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # Dispatch to the correct training function based on launch_type
    if parsed_launch_type == LaunchType.TORCHRUN:
        _train_torchrun()
    elif parsed_launch_type == LaunchType.ACCELERATE:
        _train_accelerate()
    else:
        raise ValueError(
            f"Invalid launch_type: '{launch_type}'. Must be 'torchrun' or 'accelerate'."
        )
