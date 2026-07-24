import os

import subprocess
import time

import modal
import modal.experimental


tag = "12.8.1-devel-ubuntu22.04"
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("libibverbs-dev", "libibverbs1")
    .uv_pip_install(
        "uv",
        "blobfile==3.0.0",
        "requests==2.32.4",
        "psutil",
        "ray[default]",
    )
    .uv_pip_install("vllm==0.10.0", extra_options="--torch-backend cu128")
    .uv_pip_install(
        # avoiding cursed torch==2.7.0
        "torch==2.7.1",
        pre=True,
        extra_options="--torch-backend cu128 -U",
    )
    .uv_pip_install(
        # known bug for H100s when using NCCL 2.26.x with CUDA>12.6, when NVLS enabled (default),
        # and when 2+ processes participate in collective ops. upgrading to 2.27+ fixes this
        "nvidia-nccl-cu12==2.27.6",
        extra_options="-U",
    )
    .apt_install("git", "build-essential", "g++", "wget")
    .uv_pip_install("cuda-bindings")
    .run_commands(
        # recursive bc DeepGEMM vendors CUTLASS for the build
        "git clone https://github.com/deepseek-ai/DeepGEMM.git",
        # latest commit on main broke authless recursive clone, thus:
        "cd DeepGEMM && git checkout 03d0be3d2d03b6eed3c99d683c0620949a13a826",
        "cd DeepGEMM && git submodule update --init --recursive",
        "uv pip install --system ./DeepGEMM",
    )
    .uv_pip_install("nvidia-nvshmem-cu12==3.3.9")
    .run_commands(
        "git clone https://github.com/deepseek-ai/DeepEP.git",
        # nvidia-nvshmem-cu12 ships with versioned binaries, but the DeepEP build process expects unversioned. sigh...
        "cd $(python -c 'import nvidia.nvshmem; import os; print(nvidia.nvshmem.__path__[0])') && cp lib/libnvshmem_host.so.3 lib/libnvshmem_host.so",
        "NVSHMEM_DIR=$(python -c 'import nvidia.nvshmem; import os; print(nvidia.nvshmem.__path__[0])') CXX=g++ uv pip install --system ./DeepEP --no-build-isolation",
    )
    .env({"RAY_DISABLE_DOCKER_CPU_WARNING": "1", "VLLM_USE_DEEPGEMM": "1"})
)

app = modal.App("k2-multinode-inference-dp2", image=image)

# Volume for Hugging Face cache
hf_cache_volume = modal.Volume.from_name("big-model-hfcache")
vllm_cache_volume = modal.Volume.from_name(
    "k2-multinode-vllmcache",
    create_if_missing=True,
)

# Ray configuration
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
VLLM_PORT = 8000

MODEL = "moonshotai/Kimi-K2-Instruct"

with image.imports():
    import requests


class K2Inference:
    """
    Run vLLM inference across multiple nodes with Ray orchestration.
    """

    tp_size: int
    pp_size: int
    dp_size: int
    nodes: int
    max_model_len: int = 128000
    max_seqs: int | None = None
    enable_expert_parallel: bool = True

    @modal.enter()
    def setup(self):
        cluster_info = _spawn_ray_nodes()
        container_rank = cluster_info.rank
        vllm_cmd = _build_vllm_cmd(
            self.tp_size,
            self.pp_size,
            self.dp_size,
            len(cluster_info.container_ips),
            self.max_seqs,
            self.max_model_len,
            self.enable_expert_parallel,
        )
        if container_rank == 0:
            # Run vLLM server and open a Flash tunnel for it
            try:
                vllm_process = subprocess.Popen(vllm_cmd)
                self.flash_handle = modal.experimental.flash_forward(8000)
                vllm_process.wait()
                if vllm_process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        vllm_process.returncode,
                        cmd=vllm_cmd,
                        output=vllm_process.stdout,
                        stderr=vllm_process.stderr,
                    )
            except subprocess.CalledProcessError as e:
                print(f"vLLM server failed: {e}")
                raise

    @modal.method()
    def server(self):
        pass

    @modal.exit()
    def cleanup(self):
        cluster_info = modal.experimental.get_cluster_info()

        if cluster_info.rank == 0:
            self.flash_handle.stop()

            deadline = time.time() + 60  # 1 minute deadline
            while time.time() < deadline:
                try:
                    response = requests.get("http://localhost:8000/load")
                    if response.status_code == 200:
                        load_metrics = response.json()
                        server_load = load_metrics.get("server_load")
                        print(f"Server load: {server_load} requests")
                        if server_load is None:
                            raise RuntimeError(
                                f"Server load expected from /load response, but found None: {load_metrics}"
                            )
                        if server_load == 0:
                            print("Server load is 0, continuing...")
                            break
                    else:
                        print(f"Failed to get load metrics: {response.status_code}")
                except Exception as e:
                    print(f"Error getting load metrics: {e}")

                time.sleep(1)  # Wait 1 second before next check
            else:
                print("Deadline reached, continuing regardless of server load...")

            self.flash_handle.close()


def _spawn_ray_nodes():
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
            "--include-dashboard=True",
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

    return cluster_info


def _build_vllm_cmd(
    tp_size: int,
    pp_size: int,
    dp_size: int,
    num_nodes: int,
    max_seqs: None | int,
    max_model_len: int,
    enable_expert_parallel: bool,
):
    # Start vLLM with distributed configuration
    print("Starting vLLM server on head node...")
    vllm_cmd = [
        "vllm",
        "serve",
        MODEL,
        "--download-dir",
        "/root/.cache/huggingface",
        "--served-model-name",
        "kimi-k2",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "kimi_k2",
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        "0.95",
        "--distributed-executor-backend",
        "ray",
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
    ]

    if max_seqs is not None:
        vllm_cmd.extend(["--max-num-seqs", str(max_seqs)])
    if dp_size > 1:
        dp_size_local = dp_size // num_nodes if dp_size >= num_nodes else 1

        vllm_cmd.extend(
            [
                "--data-parallel-backend",
                "ray",
                "--data-parallel-size",
                str(dp_size),
                "--data-parallel-size-local",
                str(dp_size_local),
            ]
        )
    if enable_expert_parallel:
        vllm_cmd.append("--enable-expert-parallel")

    print(f"vLLM command: {' '.join(vllm_cmd)}")
    return vllm_cmd
