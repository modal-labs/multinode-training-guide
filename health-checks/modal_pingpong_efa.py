import os
import subprocess

import modal
import modal.experimental

import time

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("libnl-route-3-200")  # required by fi_pingpong
)
app = modal.App("rdma-pingpong-efa", image=image)

# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
n_nodes = 2
# Typically this matches the number of GPUs per container.
n_proc_per_node = 8

pingpong_dict = modal.Dict.from_name("pingpong-dict", create_if_missing=True)
pingpong_dict.clear()

@app.function(
    gpu=f"H100:{n_proc_per_node}",
    experimental_options={"efa_enabled": True},
    cloud="aws",
    timeout=60 * 60, # 1 hour
)
@modal.experimental.clustered(n_nodes, rdma=True)
def fi_pingpong():
    cluster_info = modal.experimental.get_cluster_info()
    print(cluster_info)
    container_rank: int = cluster_info.rank
    
    local_ip = cluster_info.container_ipv4_ips[container_rank]
    print(f"[rank {container_rank}] IPv4 address: {local_ip}", flush=True)

    cmd_args = ["-p", "efa"]

    if container_rank == 0:
        pingpong_dict["server_ip"] = local_ip
    else:
        while pingpong_dict.get("server_ip") is None:
            time.sleep(1)
        server_ip = pingpong_dict.get("server_ip")
        cmd_args.append(server_ip)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
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
    
    for line in proc.stdout:
        print(f"[rank {container_rank}] {line}", end="", flush=True)
    
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

@app.local_entrypoint()
def main():
    fi_pingpong.remote()