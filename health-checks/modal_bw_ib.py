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
app = modal.App("rdma-bandwidth-ib", image=image)

# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
N_NODES = 2
# This is the default base port for the ib_write_bw command
BASE_PORT = 18515

server_ip_dict = modal.Dict.from_name(
    "rdma-bw-ib-server-ips", create_if_missing=True
)
if modal.is_local():
    server_ip_dict.clear()

LOGGING_DEBUG = False


@app.function(
    gpu="H200:8",
    experimental_options={"efa_enabled": False},
    cloud="gcp",
    timeout=60 * 60,  # 1 hour
)
@modal.experimental.clustered(N_NODES, rdma=True)
def infiniband_bandwidth_test():
    """Runs a bidirectional RDMA bandwidth test using perftest.
    Node rank 0 acts as server and node rank 1 acts as client. The client
    waits until the Modal dict has 8 IPs, then runs ib_write_bw.
    """

    # Get current node rank
    cluster_info = modal.experimental.get_cluster_info()
    container_rank: int = cluster_info.rank
    print(f"[rank {container_rank}] Starting rdma_bandwidth_test", flush=True)

    # Get local ib devices (sorted by device index from 0 to 7)
    local_ib_devices = get_local_ib_devices()
    print(
        f"[rank {container_rank}] Found {len(local_ib_devices)} local IB devices: {local_ib_devices}",
        flush=True,
    )

    if container_rank == 0:
        # Get all RDMA network interfaces and their ipv4 addresses
        print(f"[rank {container_rank}] Getting IP addresses...", flush=True)
        ip_addr = subprocess.run(
            ["ip", "-j", "addr", "show"],
            capture_output=True,
            text=True,
            check=True,
        )
        ib_stat_json = json.loads(ip_addr.stdout)
        for device in ib_stat_json:
            # Only get devices that start with "gpu" such as "gpu0rdma0"
            if device["ifname"].startswith("gpu"):
                # Get the device index (0->7) using regex exp to extract the first integer from the string
                idx = re.search(r"\d", device["ifname"]).group(0)
                # Extract the ipv4 address from the addr_info
                ip_address = device["addr_info"][0]["local"]
                # Update dict with new key-value pair
                server_ip_dict[idx] = ip_address
        print(
            f"[rank {container_rank}] Dict now has {server_ip_dict.len()} entries",
            flush=True,
        )

        # Spawn 8 background processes to listen for connections on each device
        processes = []
        for idx, device in enumerate(local_ib_devices):
            port = BASE_PORT + idx
            print(
                f"[rank {container_rank}] Starting ib_write_bw server on device {device} port {port}",
                flush=True,
            )
            processes.append(run_ib_write_server(device, port))
        print(
            f"[rank {container_rank}] Started {len(processes)} server processes, waiting for them to complete...",
            flush=True,
        )

        # Collect output from all server processes
        results = []
        for i, process in enumerate(processes):
            print(
                f"[rank {container_rank}] Waiting for process {i} to complete...",
                flush=True,
            )
            process.wait()
            for line in process.stdout:
                results.append(line)

        # Optionally print the output from all ib_write_bw commands
        if LOGGING_DEBUG:
            print(f"[rank {container_rank}] {''.join(results)}", flush=True)

        # Aggregate statistics from all server processes
        mean_bw_peak, mean_bw_avg = aggregate_statistics(results)
        print(
            f"[rank {container_rank}] Mean BW peak: {mean_bw_peak:.2f} Gb/s, Mean BW avg: {mean_bw_avg:.2f} Gb/s",
            flush=True,
        )

    else:
        # Wait until server has finished populating the dict with 8 items
        print(
            f"[rank {container_rank}] Waiting for server to populate dict (need 8 entries)...",
            flush=True,
        )
        while server_ip_dict.len() < 8:
            print(
                f"[rank {container_rank}] Dict has {server_ip_dict.len()} entries, waiting...",
                flush=True,
            )
            time.sleep(2)
        print(
            f"[rank {container_rank}] Server dict ready, starting client connections...",
            flush=True,
        )

        # Spawn 8 background processes to write to the server's devices
        processes = []
        for idx, device in enumerate(local_ib_devices):
            server_ip = server_ip_dict[str(idx)]
            # Assign each sender a unique ib_write_bw port to avoid conflicts
            port = BASE_PORT + idx
            print(
                f"[rank {container_rank}] Connecting device {device} to server {server_ip}:{port}",
                flush=True,
            )
            processes.append(run_ib_write_client(device, port, server_ip))

        # Wait for client processes to complete
        results = []
        for i, process in enumerate(processes):
            print(
                f"[rank {container_rank}] Waiting for client process {i} to complete...",
                flush=True,
            )
            process.wait()


# Run ib_write_bw command for server
def run_ib_write_server(device: str, port: int) -> subprocess.Popen:
    cmd = [
        "/usr/local/bin/ib_write_bw",
        "-d",
        device,
        "-p",
        str(port),
        "-R",
        "--report_gbits",
        "--bidirectional",
    ]  # Bidirectional to test both direction
    print(f"Running command: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


# Run ib_write_bw command for client with rank 0 IP as the server IP
def run_ib_write_client(device: str, port: int, server_ip: str) -> subprocess.Popen:
    cmd = [
        "/usr/local/bin/ib_write_bw",
        "-d",
        device,
        "-p",
        str(port),
        "-R",
        "--report_gbits",
        "--bidirectional",
        server_ip,
    ]
    print(f"Running command: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


# Get a sorted list of InfiniBand device names: "mlx5_0", "mlx5_1", etc.
def get_local_ib_devices() -> list[str]:
    ib_devices = subprocess.run(
        ["ls", "/sys/class/infiniband/"],
        capture_output=True,
        text=True,
        check=True,
    )
    devices = ib_devices.stdout.split("\n")[
        :-1
    ]  # Trim the final new line character from the output
    return sorted(devices)


# Returns the mean peak bandwidth and the mean bandwidth average across all devices in Gb/s.
def aggregate_statistics(results: list[str]) -> tuple[float, float]:
    bw_peaks = []
    bw_averages = []

    for line in results:
        parts = line.split()
        # Results line has 5 columns: bytes, iterations, bw_peak, bw_avg, msg_rate
        if len(parts) == 5:
            try:
                bw_peak = float(parts[2])
                bw_avg = float(parts[3])
                bw_peaks.append(bw_peak)
                bw_averages.append(bw_avg)
            except ValueError:
                continue

    if not bw_peaks:
        return 0.0, 0.0

    mean_bw_peak = sum(bw_peaks) / len(bw_peaks)
    mean_bw_avg = sum(bw_averages) / len(bw_averages)
    return mean_bw_peak, mean_bw_avg


@app.local_entrypoint()
def main():
    infiniband_bandwidth_test.remote()
