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
    .apt_install(
        "build-essential",
        "autoconf",
        "automake",
        "libtool",
        "libibverbs-dev",
        "ibverbs-utils",
        "librdmacm-dev",
        "libibumad-dev",
        "libpci-dev",
        "libfabric-dev",
        "libfabric-bin",
        "iproute2",
        "git",
    )
    .run_commands(
        "git clone https://github.com/linux-rdma/perftest.git /opt/perftest",
        "cd /opt/perftest && ./autogen.sh && ./configure --prefix=/usr/local/ && make -j && make install",
    )
)
app = modal.App("rdma-bandwidth-ib", image=image)

# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
n_nodes = 2
# Typically this matches the number of GPUs per container.
n_proc_per_node = 8

bandwidth_dict = modal.Dict.from_name("bandwidth-dict", create_if_missing=True)
bandwidth_dict.clear()

@app.function(
    gpu=f"H200:{n_proc_per_node}",
    experimental_options={"efa_enabled": False},
    cloud="gcp",
    timeout=60 * 60, # 1 hour
)
@modal.experimental.clustered(n_nodes, rdma=True)
def rdma_bandwidth_test():
    import re
    
    cluster_info = modal.experimental.get_cluster_info()
    print(cluster_info)
    container_rank: int = cluster_info.rank
    
    rdma_device = "mlx5_0"
    gid_index = 3  # Using GID index 3 for RoCE v2
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    # Get GID table from ibv_devinfo to extract the IPv4 address
    devinfo_result = subprocess.run(
        ["/usr/bin/ibv_devinfo", "-v", "-d", rdma_device],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    print(f"[rank {container_rank}] [ibv_devinfo output]:", flush=True)
    print(devinfo_result.stdout, flush=True)
    
    # Parse the IP address from GID[3] in the output
    # Looking for: GID[  3]:		::ffff:10.200.0.10, RoCE v2
    gid_pattern = rf'GID\[\s*{gid_index}\]:\s+::ffff:(\d+\.\d+\.\d+\.\d+)'
    ip_match = re.search(gid_pattern, devinfo_result.stdout)
    
    if not ip_match:
        raise RuntimeError(f"Could not find IPv4 address in GID[{gid_index}] for device {rdma_device}")
    
    local_ip = ip_match.group(1)
    print(f"[rank {container_rank}] Extracted RDMA IPv4 address from GID[{gid_index}]: {local_ip}", flush=True)

    cmd_args = ["-d", rdma_device, "-x", str(gid_index), "-R", "--report_gbits", "--bidirectional"]

    if container_rank == 0:
        bandwidth_dict["server_ip"] = local_ip
    else:
        while bandwidth_dict.get("server_ip") is None:
            time.sleep(1)
        server_ip = bandwidth_dict.get("server_ip")
        cmd_args.append(server_ip)

    cmd = ["/usr/local/bin/ib_write_bw", *cmd_args]
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
    rdma_bandwidth_test.remote()
