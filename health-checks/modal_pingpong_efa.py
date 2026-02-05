import os
import subprocess

import modal
import modal.experimental

import time

cuda_version = "12.4.0"  # Should be no greater than host CUDA version
flavor = "devel"  #  Includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc15",
        "libnl-route-3-200",
    )
)
app = modal.App("rdma-pingpong-efa", image=image)

# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
N_NODES = 2

server_ip_dict = modal.Dict.from_name("server-ip-dict", create_if_missing=True)

@app.function(
    gpu=f"H100:8",
    experimental_options={"efa_enabled": True},
    cloud="aws",
    timeout=60 * 60, # 1 hour
)
@modal.experimental.clustered(N_NODES, rdma=True)
def fi_pingpong():
    """Runs a ping-pong test between two nodes using AWS's EFA provider. Node rank 0 first adds its ipv4 address to the dict
    then starts the server for fi_pingpong, which is a libfabric utility. Node rank 1 waits until the server has added its ip address to the dict 
    and then runs the fi_pingpong command with the ip as an argument. You should see a successful output from both nodes showing sent/received bytes."""

    # Get current node rank
    cluster_info = modal.experimental.get_cluster_info()
    container_rank: int = cluster_info.rank

    # Update the dict with server ip address
    cmd_args = ["-p", "efa"]
    if container_rank == 0:
        server_ip_dict["server_ip"] = cluster_info.container_ipv4_ips[0]
    else:
        # Wait until server has added its ip address to the dict
        while server_ip_dict.get("server_ip") is None:
            time.sleep(1)
        # Add server ip address to fi_pingpong args
        cmd_args.append(server_ip_dict.get("server_ip"))

    # Run fi_pingpong command
    env = os.environ.copy()
    cmd = ["/opt/amazon/efa/bin/fi_pingpong", *cmd_args]
    print(f"[rank {container_rank}] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Print output with sent/received bytes to console
    for line in proc.stdout:
        print(f"[rank {container_rank}] {line}", end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

@app.local_entrypoint()
def main():
    fi_pingpong.remote()