from constants import NUM_NODES

import modal
import modal.experimental
import subprocess
import requests
import pathlib
import time

app = modal.App(name="multinode-rl")
cuda_version = "12.6.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libibverbs-dev", "libibverbs1")
    .run_commands(
        "uv pip install --system huggingface_hub[hf_xet]==0.32.4 vllm==0.8.5.post1 trl[vllm,deepspeed]==0.18.1 requests",
        "uv pip install --system flashinfer-python==0.2.2.post1 -i https://flashinfer.ai/whl/cu124/torch2.6",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        }
    )
    .run_commands("uv pip install --system wandb")
    .add_local_file("constants.py", remote_path="/root/constants.py")
)

# ## Caching HuggingFace, vLLM, and storing model weights
# We create Modal Volumes to persist:
# - HuggingFace downloads
# - vLLM cache
# - Model weights

HF_CACHE_DIR = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

VLLM_CACHE_DIR = "/root/.cache/vllm"
VLLM_CACHE_VOL = modal.Volume.from_name("vllm-cache", create_if_missing=True)

WEIGHTS_DIR = "/root/multinode_rl_weights"
WEIGHTS_VOL = modal.Volume.from_name("multinode-rl-weights", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ae659febe817e4b3ebd7355f47792725801204c9"


def run_vllm_server():
    cmd = [
        "trl",
        "vllm-serve",
        "--model",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--port",
        "8000",
        "--log-level",
        "info",
        "--tensor-parallel-size",
        "8",
        "--host",
        "0.0.0.0",
    ]
    vllm_process = subprocess.Popen(
        ["/bin/bash", "-c", " ".join(cmd)],
        stderr=subprocess.STDOUT,
    )
    vllm_process.wait()

def _generate_node_config(
    machine_rank, leader_ip, leader_port, template_path, target_path
):
    try:
        accelerate_config_template = template_path.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: Accelerate template file not found at {template_path.as_posix()}"
        ) from None

    formatted_config_content = accelerate_config_template.format(
        rank=machine_rank, leader_ip=leader_ip, leader_port=leader_port, num_nodes=NUM_NODES - 1
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(formatted_config_content)


def run_trainer(cluster_info):
    """Runs the training process on rank 0, waiting for the vLLM server."""

    from requests.exceptions import ConnectionError

    parent_dir = pathlib.Path(__file__).parent
    accelerate_config_path = parent_dir / "config_grpo_multinode.yaml"
    generated_accelerate_config_path = parent_dir / "zero3.yaml"
    trainer_script = parent_dir / "trainer_script_grpo.py"

    vllm_server_host_ip = f"10.100.0.{NUM_NODES}" # 10.100.0.{rank + 1}, we run the vllm server on the last node.
    vllm_server_url = f"http://{vllm_server_host_ip}:8000"

    accelerate_leader_ip = "10.100.0.1" # we run the accelerate leader on the first node.
    accelerate_leader_port = 29500

    rank = cluster_info.rank
    _generate_node_config(
        rank,
        accelerate_leader_ip,
        accelerate_leader_port,
        accelerate_config_path,
        generated_accelerate_config_path,
    )
    print(
        f"Container Modal rank {rank} (Trainer): Generated accelerate "
        f"config at {generated_accelerate_config_path.as_posix()}"
    )

    print(
        f"Container rank {rank} (Trainer) waiting for vLLM server at {vllm_server_url}..."
    )

    while True:
        try:
            response = requests.get(f"{vllm_server_url}/health")
            if response.status_code:  # Any status code means it's up
                print(f"Container rank {rank} (Trainer): vLLM server is ready!")
                break
        except ConnectionError:
            pass
        time.sleep(5)

    train_cmd = [
        "accelerate",
        "launch",
        "--config-file",
        f"{generated_accelerate_config_path.as_posix()}",
        f"{trainer_script.as_posix()}",
        "--vllm_server_host",
        f"{vllm_server_host_ip}",
    ]
  
    print(
        f"Container Modal rank {rank} (Trainer): "
        f"Launching train command: {' '.join(train_cmd)}"
    )

    # Launch the training process
    # Using Popen directly with a list of arguments is generally safer than shell=True
    train_process = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout pipe
        text=True,  # Decode output as text
        bufsize=1,  # Line buffered
        universal_newlines=True,  # Ensure cross-platform newline handling
    )
    # Stream output from the subprocess
    if train_process.stdout:
        for line in iter(train_process.stdout.readline, ""):
            print(f"[Trainer Rank {rank} Output]: {line.strip()}", flush=True)
        train_process.stdout.close()

    train_process.wait()
    if train_process.returncode == 0:
        print(f"Container Modal rank {rank} (Trainer) process exited successfully")
    else:
        print(
            f"Container Modal rank {rank} (Trainer) process exited with code {train_process.returncode}."
        )           

@app.function(
    gpu="H100:8",
    image=image,
    volumes={
        HF_CACHE_DIR: HF_CACHE_VOL,
        VLLM_CACHE_DIR: VLLM_CACHE_VOL,
        WEIGHTS_DIR: WEIGHTS_VOL,
    },
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret", environment_name="main")],
)
@modal.experimental.clustered(NUM_NODES, rdma=True)
def train_fn(trainer_script: str, config_file: str):    
    import wandb

    with open("/root/trainer_script_grpo.py", "w") as f:
        f.write(trainer_script)
    with open("/root/config_grpo_multinode.yaml", "w") as f:
        f.write(config_file)

    wandb.init(project="multinode-rl")

    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank

    if rank == NUM_NODES - 1:
        run_vllm_server()
    elif rank < NUM_NODES - 1:
        run_trainer(cluster_info)
    else:
        print(f"Container rank {rank} has no specific role in this setup.") 


@app.local_entrypoint()
def main(
    trainer_script: str = "trainer_script_grpo.py",
    config_file: str = "config_grpo.yaml",
):
    print(
        f"Training with trainer script: {trainer_script} and config file: {config_file}"
    )
    with open(trainer_script, "r") as f:
        trainer_content = f.read()
    with open(config_file, "r") as f:
        config_content = f.read()

    train_fn.remote(trainer_content, config_content)

