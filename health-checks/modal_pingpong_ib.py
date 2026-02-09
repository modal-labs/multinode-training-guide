import subprocess

import modal
import modal.experimental

import time
import json
import re

cuda_version = "12.4.0"  # Should be no greater than host CUDA version
flavor = "devel"  #  Includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "build-essential",
        "autoconf",
        "automake",
        "ca-certificates",
        "curl",
        "libtool",
        "libibverbs-dev",
        "ibverbs-utils",
        "librdmacm-dev",
        "libibumad-dev",
        "libpci-dev",
        "iproute2",
    )
    .run_commands(
        "curl -fsSL -o /tmp/perftest-25.10.0-0.128.tar.gz https://github.com/linux-rdma/perftest/archive/refs/tags/25.10.0-0.128.tar.gz",
        "rm -rf /opt/perftest && mkdir -p /opt && tar -xzf /tmp/perftest-25.10.0-0.128.tar.gz -C /opt && mv /opt/perftest-25.10.0-0.128 /opt/perftest",
        "rm -f /tmp/perftest-25.10.0-0.128.tar.gz",
        "cd /opt/perftest && ./autogen.sh && ./configure --prefix=/usr/local/ && make -j && make install",
    )
)
app = modal.App("rdma-pingpong-ib", image=image)

# The number of containers (i.e. nodes) in the cluster.
N_NODES = 2
# Default port for the ib_send_lat command
BASE_PORT = 18515

server_ip_dict = modal.Dict.from_name(
    "rdma-pingpong-ib-server-ips", create_if_missing=True
)
if modal.is_local():
    server_ip_dict.clear()


@app.function(
    gpu="H200:8",
    experimental_options={"efa_enabled": False},
    cloud="gcp",
    timeout=60 * 60,  # 1 hour
)
@modal.experimental.clustered(N_NODES, rdma=True)
def infiniband_pingpong_test():
    """Runs a ping-pong latency test over InfiniBand using ib_send_lat from the perftest suite.
    Tests only the first RDMA device (mlx5_0) between two nodes. Node rank 0 acts as the server
    and node rank 1 acts as the client."""

    # Get current node rank
    cluster_info = modal.experimental.get_cluster_info()
    container_rank: int = cluster_info.rank
    print(
        f"[rank {container_rank}] Starting InfiniBand_pingpong_test",
        flush=True,
    )

    # Get local ib devices (sorted by device index from 0 to 7)
    local_ib_devices = get_local_ib_devices()
    print(
        f"[rank {container_rank}] Found {len(local_ib_devices)} local IB devices: {local_ib_devices}",
        flush=True,
    )

    # Only use the first device
    device = local_ib_devices[0]
    print(f"[rank {container_rank}] Using first device: {device}", flush=True)

    if container_rank == 0:
        # Get RDMA network interfaces and find the IP for the first gpu device
        ip_addr = subprocess.run(
            ["ip", "-j", "addr", "show"],
            capture_output=True,
            text=True,
            check=True,
        )
        ib_stat_json = json.loads(ip_addr.stdout)
        for iface in ib_stat_json:
            if iface["ifname"].startswith("gpu"):
                idx = re.search(r"\d", iface["ifname"]).group(0)
                if idx == "0":
                    ip_address = iface["addr_info"][0]["local"]
                    server_ip_dict["0"] = ip_address
                    print(
                        f"[rank {container_rank}] Published server IP for device 0: {ip_address}",
                        flush=True,
                    )
                    break

        # Start ib_send_lat server on the first device
        print(
            f"[rank {container_rank}] Starting ib_send_lat server on device {device} port {BASE_PORT}",
            flush=True,
        )
        cmd = [
            "/usr/local/bin/ib_send_lat",
            "-d",
            device,
            "-p",
            str(BASE_PORT),
            "-R",
        ]
        print(f"[rank {container_rank}] Running: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
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

    else:
        # Client: wait for the server to publish its IP, then connect
        print(
            f"[rank {container_rank}] Waiting for server to publish IP...",
            flush=True,
        )
        while server_ip_dict.get("0") is None:
            time.sleep(1)
        server_ip = server_ip_dict["0"]
        print(f"[rank {container_rank}] Server IP: {server_ip}", flush=True)

        # Start ib_send_lat client connecting to server
        cmd = [
            "/usr/local/bin/ib_send_lat",
            "-d",
            device,
            "-p",
            str(BASE_PORT),
            "-R",
            server_ip,
        ]
        print(f"[rank {container_rank}] Running: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
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


# Get a sorted list of InfiniBand device names: "mlx5_0", "mlx5_1", etc.
def get_local_ib_devices() -> list[str]:
    ib_devices = subprocess.run(
        ["ls", "/sys/class/infiniband/"],
        capture_output=True,
        text=True,
        check=True,
    )
    devices = ib_devices.stdout.split("\n")[:-1]  # Trim the final new line character
    return sorted(devices)


@app.local_entrypoint()
def main():
    infiniband_pingpong_test.remote()
