"""
Standalone multi-node Ray + Slime pattern.

Design:
- Ray cluster bootstrap happens in modal.enter() inside a modal.Cls.
- Job submission happens in a separate modal.method().
- The submitting method is multi-node via @modal.experimental.clustered.
- Local entrypoint accepts a config name, generates the Slime command, and submits it.
"""

import os
import pathlib
import subprocess
import time

import modal
import modal.experimental

from slime_configs import get_config, list_configs


app = modal.App(os.environ.get("RAY_APP_NAME", "ray-slime-standalone"))

N_NODES = 2
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
HF_CACHE_PATH = "/root/.cache/huggingface"
DATA_PATH = "/data"
CHECKPOINTS_PATH = "/checkpoints"

here = pathlib.Path(__file__).parent

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("grpo-slime-example-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(
    "grpo-slime-checkpoints", create_if_missing=True
)


image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260126a")
    .entrypoint([])
    .add_local_file(here / "slime_configs.py", "/root/slime_configs.py")
)

with image.imports():
    import ray
    from huggingface_hub import snapshot_download
    from ray.job_submission import JobSubmissionClient


def _build_slime_cmd(config_name: str) -> tuple[str, str]:
    cfg = get_config(config_name)
    train_args = cfg.generate_train_args(
        hf_model_path="$MODEL_PATH",
        checkpoints_path=CHECKPOINTS_PATH,
        data_path=DATA_PATH,
    )
    train_script = "slime/train.py" if cfg.sync else "slime/train_async.py"
    cmd = f"python3 {train_script} {train_args}"
    return cmd, cfg.model_id


@app.cls(
    image=image,
    gpu="H100:8",
    volumes={
        HF_CACHE_PATH: hf_cache_volume,
        DATA_PATH: data_volume,
        CHECKPOINTS_PATH: checkpoints_volume,
    },
    timeout=24 * 60 * 60,
    scaledown_window=1 * 60 * 60,
    retries=10,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
class RayCluster:
    @modal.enter()
    def bootstrap_ray(self):
        hf_cache_volume.reload()
        data_volume.reload()
        checkpoints_volume.reload()

        cluster_info = modal.experimental.get_cluster_info()
        self.rank = cluster_info.rank
        self.node_ips = cluster_info.container_ipv4_ips
        self.main_addr = self.node_ips[0]
        self.node_addr = self.node_ips[self.rank]

        os.environ["SLIME_HOST_IP"] = self.node_addr

        if self.rank == 0:
            print(f"Starting Ray head at {self.node_addr}")
            subprocess.Popen(
                [
                    "ray",
                    "start",
                    "--head",
                    f"--node-ip-address={self.node_addr}",
                    "--dashboard-host=0.0.0.0",
                ]
            )

            for _ in range(30):
                try:
                    ray.init(address="auto")
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("Failed to connect to Ray head")

            for _ in range(60):
                alive_nodes = [n for n in ray.nodes() if n["Alive"]]
                print(f"Alive nodes: {len(alive_nodes)}/{len(self.node_ips)}")
                if len(alive_nodes) == len(self.node_ips):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Not all worker nodes connected to Ray head")

            self.client = JobSubmissionClient(f"http://127.0.0.1:{RAY_DASHBOARD_PORT}")
            print("Ray cluster is ready and idle.")
        else:
            print(f"Starting Ray worker at {self.node_addr}, head={self.main_addr}")
            subprocess.Popen(
                [
                    "ray",
                    "start",
                    f"--node-ip-address={self.node_addr}",
                    "--address",
                    f"{self.main_addr}:{RAY_PORT}",
                ]
            )

    @modal.method()
    async def submit_slime_job(self, slime_cmd: str, hf_model: str) -> dict:
        if self.rank == 0:
            model_path = snapshot_download(repo_id=hf_model, local_files_only=True)
            entrypoint = slime_cmd.replace("$MODEL_PATH", model_path)

            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            megatron_path = "/root/Megatron-LM/"
            pythonpath = (
                f"{megatron_path}:{existing_pythonpath}"
                if existing_pythonpath
                else megatron_path
            )

            runtime_env = {
                "env_vars": {
                    "MASTER_ADDR": self.main_addr,
                    "no_proxy": self.main_addr,
                    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                    "NCCL_NVLS_ENABLE": "1",
                    "PYTHONPATH": pythonpath,
                }
            }

            job_id = self.client.submit_job(
                entrypoint=entrypoint, runtime_env=runtime_env
            )
            print(f"Submitted Ray job: {job_id}")
            print(f"Entrypoint: {entrypoint}")

            async for line in self.client.tail_job_logs(job_id):
                print(line, end="", flush=True)

            status = self.client.get_job_status(job_id).value
            print(f"\nFinal status: {status}")
            return {"job_id": job_id, "status": status}

        while True:
            time.sleep(10)


@app.local_entrypoint()
def main(config: str = "qwen-8b-multi"):
    if config not in list_configs():
        available = ", ".join(list_configs())
        raise ValueError(f"Unknown config: {config}. Available configs: {available}")

    slime_cmd, hf_model = _build_slime_cmd(config)
    print(f"Using config: {config}")
    print(f"Model: {hf_model}")

    cluster = modal.Cls.from_name("ray-slime-standalone", "RayCluster")()
    result = cluster.submit_slime_job.remote(slime_cmd=slime_cmd, hf_model=hf_model)
    print(result)
