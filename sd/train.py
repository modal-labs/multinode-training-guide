import subprocess
import threading
import modal
import os
import secrets
import time
import socket

from modal import Queue, forward

import modal.experimental

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/mosaicml/diffusion.git",
        "cd diffusion && pip install -e .",
    )
    .pip_install("jupyterlab")
    .apt_install(
        # Needed for RDMA.
        "libibverbs-dev",
        "libibverbs1",
    )
    .add_local_dir("diffusion", remote_path="/root/diffusion")
)
# This is in nathan-dev in production.
latents_vol = modal.Volume.from_name("laion-aesthetics_v2_4.5-latents", version=2)
# This is in nathan-dev in dev_cluster.
# latents_vol = modal.Volume.from_name("laion-aesthetics_v2_4.5-latents")
# The Weights & Biases secret 'wandb-secret-modal-labs' is in the nathan-dev namespace
# and points at @thundergolfer's personal Weights & Biases account. 'wandb-secret' in the
# nathan-dev namespace points at @thecodingwizard's personal Weights & Biases account.
wandb_secret = modal.Secret.from_name("wandb-secret-modal-labs")
app = modal.App(
    "sd-train",
    image=image,
    volumes={"/latents": latents_vol},
    secrets=[wandb_secret],
)

n_gpus_per_node = 8
n_nodes = 4

def export_rdma_env():
    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.environ.get('LD_LIBRARY_PATH', '')}:/usr/local/lib"
    )
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["LOGLEVEL"] = "DEBUG"
    os.environ["NCCL_IB_SPLIT_DATA_ON_QPS"] = "0"
    os.environ["NCCL_IB_QPS_PER_CONNECTION"] = "4"
    os.environ["NCCL_IB_TC"] = "41"
    os.environ["NCCL_IB_SL"] = "0"
    os.environ["NCCL_IB_TIMEOUT"] = "22"

    # Control‑plane (TCP) — stays on eth1, uses IPv6
    os.environ["NCCL_SOCKET_IFNAME"] = "eth1"
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET6"

    # Data‑plane (RDMA) — stays on the HCA ports, uses IPv4
    os.environ["NCCL_IB_ADDR_FAMILY"] = "AF_INET"
    os.environ["NCCL_IB_GID_INDEX"] = "3"  # OCI's IPv4‑mapped GID index
    os.environ["NCCL_IB_HCA"] = (
        "mlx5_0,mlx5_1,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17"
    )

@app.function(
    timeout=60 * 60 * 24,
    gpu=f"H100:{n_gpus_per_node}",
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    # RDMA is currently only supported on OCI in us-chicago-1
    cloud="oci",
    region="us-chicago-1",
    image=image,
    experimental_options={"rdma_enabled": "1"},
)
@modal.experimental.clustered(size=n_nodes)
def train(rdma: bool = False):
    multinode_flags = ""
    os.environ["HYDRA_FULL_ERROR"] = "1"

    if n_nodes > 1:
        if os.environ["MODAL_CLOUD_PROVIDER"] not in (
            "CLOUD_PROVIDER_GCP",
            "CLOUD_PROVIDER_OCI",
        ):
            raise ValueError("Only GCP and OCI are supported")

        if rdma:
            print("Exporting RDMA environment variables")
            export_rdma_env()

        # os.environ["NCCL_TOPO_DUMP_FILE"] = "/latents/nccl_topo.txt"
        # os.environ["NCCL_GRAPH_DUMP_FILE"] = "/latents/nccl_graph.txt"
        # os.environ["NCCL_DEBUG"] = "TRACE"
        # os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET,GRAPH,TUNING"
        # os.environ["NCCL_DEBUG_FILE"] = "/latents/debug.%h.%p"
        # os.environ["NCCL_SOCKET_NTHREADS"] = "4"
        # os.environ["NCCL_NSOCKS_PERTHREAD"] = "1"

        cluster_info = modal.experimental.get_cluster_info()
        multinode_flags = f"--world_size {len(cluster_info.container_ips) * n_gpus_per_node} --node_rank {cluster_info.rank} --master_addr {cluster_info.container_ips[0]} --master_port 29500"

    os.system(
        f"cd diffusion && composer {multinode_flags} run.py --config-path yamls/hydra-yamls --config-name SD-2-base-256.yaml"
    )


def wait_for_port(url: str, q: Queue):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 8888), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError(
                    "Waited too long for port 8888 to accept connections"
                ) from exc
    q.put(url)


@app.function(
    timeout=60 * 60 * 24,
    gpu="A100",
    cpu=8,
)
def run_jupyter(q: Queue):
    token = secrets.token_urlsafe(13)
    with forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        threading.Thread(target=wait_for_port, args=(url, q)).start()
        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                "--port=8888",
                "--notebook-dir=/root",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )
    q.put("done")


@app.function(timeout=6000)
def verify_shard(shard_path: str):
    import json

    index_filename = os.path.join(shard_path, "index.json")
    try:
        obj = json.load(open(index_filename))
    except Exception as e:
        print(f"Error reading {index_filename}: {e}")
        return False

    for info in obj["shards"]:
        basename = info["raw_data"]["basename"]

        filename = os.path.join(shard_path, basename)

        if not os.path.exists(filename):
            print(f"ERROR: {filename} does not exist")
            return False
        else:
            filesize = os.path.getsize(filename)
            if filesize != int(info["raw_data"]["bytes"]):
                print(
                    f"ERROR: {filename} has size {filesize} but {info['raw_data']['bytes']} in index.json"
                )
                return False
    return True


@app.function(
    timeout=60 * 20,
    volumes={"/vol": modal.Volume.from_name(
        "sd-coco-volumefs1",
        version=1,
    ), "/models": modal.Volume.from_name("sd-checkpoints", version=1)},
    gpu="H100",
)
def offline_eval():
    # TODO: gave up on getting this working. Code has so many issues I think you'd have to understand almost all 
    # of it to get it working. Just kept stepping into rakes.
    subprocess.run(
        "composer run_eval.py --config-path yamls/hydra-yamls --config-name eval-clean-fid",
        shell=True,
        check=True,
        cwd="/root/diffusion/",
    )


@app.local_entrypoint()
def main(rdma: bool = False):
    # To start the training job:
    train.remote(rdma)

    # To verify MDS shard correctness:
    # verify_shard.remote("/latents/data")

    # To run Jupyter:
    # with Queue.ephemeral() as q:
    #     run_jupyter.spawn(q)
    #     url = q.get()
    #     time.sleep(1)  # Give Jupyter a chance to start up
    #     print("\nJupyter on Modal, opening in browser...")
    #     print(f"   -> {url}\n")
    #     assert q.get() == "done"
