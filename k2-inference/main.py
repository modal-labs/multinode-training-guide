import os
import subprocess
import time

import modal
import modal.experimental


tag = "12.8.1-devel-ubuntu22.04"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
    )
    .run_commands(
        "uv pip install --system -U uv",
        "uv pip install --system blobfile==3.0.0",
        # using nightly until they cut a new release (current stable is v0.9.2)
        # we need this vllm commit to use pipeline parallelism with kimi:
        # https://github.com/vllm-project/vllm/commit/ad6c2e1a0b56c29065c7d70ff2e736e4f2fb03af
        "uv pip install --system --pre vllm --extra-index-url https://wheels.vllm.ai/nightly",
        "uv pip install --system --pre -U torch==2.7.1",
        # known bug for H100s when using NCCL 2.26.x with CUDA>12.6, when NVLS enabled (default),
        # and when 2+ processes participate in collective ops. upgrading to 2.27+ fixes this
        "uv pip install --system -U nvidia-nccl-cu12==2.27.6",
    )
    .env(
        {
            "RAY_DISABLE_DOCKER_CPU_WARNING": "1",
        }
    )
)

app = modal.App("k2-multinode-inference", image=image)

# Volume for Hugging Face cache
hf_cache_volume = modal.Volume.from_name("big-model-hfcache")
vllm_cache_volume = modal.Volume.from_name(
    "k2-multinode-vllmcache", create_if_missing=True, version=2
)

# Number of nodes and GPUs configuration
N_NODES = 4
GPUS_PER_NODE = 8
GPU_TYPE = "H100"

# Ray configuration
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
VLLM_PORT = 8000

# vllm
MODEL = "moonshotai/Kimi-K2-Instruct"
TP_SIZE = 16
PP_SIZE = 2
MAX_MODEL_LEN = 16384
MAX_SEQS = 32


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPUS_PER_NODE}",
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    timeout=60 * 60 * 1,
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def run_vllm_inference():
    """
    Run vLLM inference across multiple nodes with Ray orchestration.
    """
    # Get cluster information
    cluster_info = modal.experimental.get_cluster_info()
    container_rank = cluster_info.rank
    container_v4_ips = [
        f"10.100.0.{rank + 1}" for rank in range(len(cluster_info.container_ips))
    ]
    main_addr_v4 = container_v4_ips[0]
    this_v4 = container_v4_ips[container_rank]

    # Set up Ray environment variables
    os.environ["RAY_ADDRESS"] = f"{main_addr_v4}:{RAY_PORT}"
    os.environ["VLLM_HOST_IP"] = this_v4

    # Start Ray on each node
    if container_rank == 0:
        # Head node
        print("Starting Ray head node...")
        ray_cmd = [
            "ray",
            "start",
            "--head",
            f"--node-ip-address={main_addr_v4}",
            f"--port={RAY_PORT}",  # 6379
            "--dashboard-host=0.0.0.0",
            f"--dashboard-port={RAY_DASHBOARD_PORT}",  # 8265
            "--block",
        ]
    else:
        # Worker nodes
        print(f"Starting Ray worker node {container_rank}...")
        # Give head node time to start
        time.sleep(10)
        ray_cmd = [
            "ray",
            "start",
            f"--node-ip-address={this_v4}",
            f"--address={main_addr_v4}:{RAY_PORT}",
            "--block",
        ]

    # Start Ray in the background
    ray_process = subprocess.Popen(ray_cmd)

    # Only the head node runs vLLM server
    if container_rank != 0:
        # Worker nodes just keep Ray running
        print(f"Worker node {container_rank} keeping Ray alive...")
        ray_process.wait()
    else:
        # Wait for all Ray nodes to connect
        print("Waiting for Ray cluster to be ready...")
        time.sleep(30)

        # Check Ray cluster status
        subprocess.run(["ray", "status"], check=True)

        # Start vLLM with distributed configuration
        print("Starting vLLM server on head node...")
        vllm_cmd = [
            "vllm",
            "serve",
            MODEL,
            "--download-dir",
            "/root/.cache/huggingface",
            "--trust-remote-code",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--distributed-executor-backend",
            "ray",
            "--tensor-parallel-size",
            str(TP_SIZE),
            "--pipeline-parallel-size",
            str(PP_SIZE),
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--max-num-seqs",
            str(MAX_SEQS),
        ]

        print(f"vLLM command: {' '.join(vllm_cmd)}")

        # Run vLLM server
        try:
            subprocess.run(vllm_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"vLLM server failed: {e}")
            raise
