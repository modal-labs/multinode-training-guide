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
)
# This is in nathan-dev in production.
latents_vol = modal.Volume.from_name("laion-aesthetics_v2_4.5-latents", version=2)
# This is in nathan-dev in dev_cluster.
# latents_vol = modal.Volume.from_name("laion-aesthetics_v2_4.5-latents")
# The Weigths & Biases secret 'wandb-secret-modal-labs' is in the nathan-dev namespace
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


@app.function(
    mounts=[modal.Mount.from_local_dir("diffusion", remote_path="/root/diffusion")],
    timeout=60 * 60 * 24,
    gpu=f"H100:{n_gpus_per_node}",
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
)
@modal.experimental.clustered(size=n_nodes)
def train():
    multinode_flags = ""
    os.environ["HYDRA_FULL_ERROR"] = "1"

    if n_nodes > 1:
        if os.environ["MODAL_CLOUD_PROVIDER"] not in (
            "CLOUD_PROVIDER_GCP",
            "CLOUD_PROVIDER_OCI",
        ):
            raise ValueError("Only GCP and OCI are supported")

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
    mounts=[modal.Mount.from_local_dir("diffusion", remote_path="/root/diffusion")],
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


@app.local_entrypoint()
def main():
    # To start the training job:
    train.remote()

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
