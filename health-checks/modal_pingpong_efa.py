import subprocess

import modal
import modal.experimental

import time

cuda_version = "12.4.0"  # Should be no greater than host CUDA version
flavor = "devel"  #  Includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10").apt_install(
    "libibverbs-dev",
    "libibverbs1",
    "libhwloc15",
    "libnl-route-3-200",
)
app = modal.App("rdma-pingpong-efa", image=image)

# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
N_NODES = 2

@app.function(
    gpu="H100:8",
    experimental_options={"efa_enabled": True},
    cloud="aws",
    timeout=60 * 60,  # 1 hour
)
@modal.experimental.clustered(N_NODES, rdma=True)
def fi_pingpong(server_ip_dict: modal.Dict):
    """
    Runs a ping-pong test between two nodes on EFA.
    Node rank 0 adds its IPv4 address to the dict, then starts the
    fi_pingpong server. Node rank 1 waits until
    the server's IP has been added to the dict, then runs
    fi_pingpong command with the IP as an argument.
    """

    # Get current node rank
    cluster_info = modal.experimental.get_cluster_info()
    container_rank: int = cluster_info.rank
    print(f"[rank {container_rank}] cluster {cluster_info.cluster_id}", flush=True)

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
    cmd = ["/opt/amazon/efa/bin/fi_pingpong", *cmd_args]
    print(f"[rank {container_rank}] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
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
    with modal.Dict.ephemeral() as server_ip_dict:
        fi_pingpong.remote(server_ip_dict)
