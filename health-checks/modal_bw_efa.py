import math
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
        "curl",
        "pciutils",
    )
    .run_commands(
        # Download fabtests 2.3.1 to match the EFA libfabric (v2.3.1amzn1.0).
        "curl -fsSL -o /tmp/fabtests-2.3.1.tar.bz2 https://github.com/ofiwg/libfabric/releases/download/v2.3.1/fabtests-2.3.1.tar.bz2",
        "tar -xjf /tmp/fabtests-2.3.1.tar.bz2 -C /opt && rm -f /tmp/fabtests-2.3.1.tar.bz2",
    )
)
app = modal.App("rdma-bandwidth-efa", image=image)

# Number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
N_NODES = 2
# Base port for the fi_rma_bw command
BASE_PORT = 30000
# RDMA Message Size. IMPORTANT: Keep < 1 or libfabric will use the read message protocol, which triggers a slow, multi-turn trip.
MB_PER_WRITE = 1

LOGGING_DEBUG = True


@app.function(
    gpu="H200:8",
    experimental_options={"efa_enabled": True},
    cloud="aws",
    timeout=60 * 60,  # 1 hour
)
@modal.experimental.clustered(N_NODES, rdma=True)
def efa_bandwidth_test(server_ip_dict: modal.Dict):
    """Runs a bidirectional RDMA bandwidth test using the fabtests library. Creates a Modal dict for storing the server's container IP address.
    The client waits until this value has been added and then runs a fi_rma_bw command to benchmark bandwidth. On AWS EFA, every device has 2 NICs
    and each NIC has its own RDM (reliable datagram) domain. 16 fi_rma_bw processes (one per domain) are spun up in parrallel.
    """

    # Build and install fabtests at runtime because /opt/amazon/efa is only mounted at runtime
    build_fabtests()

    # Get current node rank
    cluster_info = modal.experimental.get_cluster_info()
    container_rank: int = cluster_info.rank
    print(f"[rank {container_rank}] Starting rdma_bandwidth_test", flush=True)

    # Get local ib devices (sorted by device index from 0 to 7)
    local_efa_domains = get_local_efa_domains()
    print(
        f"[rank {container_rank}] Found {len(local_efa_domains)} local EFA domains: {local_efa_domains}",
        flush=True,
    )

    if container_rank == 0:
        # Add server ip address to dict
        server_ip_dict["server_ip"] = cluster_info.container_ipv4_ips[0]
        print(
            f"[rank {container_rank}] Server ip address added to dict",
            flush=True,
        )

        # Spawn 32 background processes to listen for connections on each EFA NIC
        processes = []
        for idx, domain in enumerate(local_efa_domains):
            print(
                f"[rank {container_rank}] Starting fi_rma_bw server on domain {domain}",
                flush=True,
            )
            processes.append(
                run_efa_write_server(
                    domain, port=BASE_PORT + idx, gpu_id=math.floor(idx / 4)
                )
            )
        print(
            f"[rank {container_rank}] Started {len(processes)} server processes, this could take a minute...",
            flush=True,
        )

        # Collect output from all server processes
        results = []
        for i, process in enumerate(processes):
            print(
                f"[rank {container_rank}] Waiting for process {i} to complete...",
                flush=True,
            )
            stdout, _ = process.communicate()
            results.append(stdout)

        # Optionally print the output from all fi_rma_bw commands
        if LOGGING_DEBUG:
            print(f"[rank {container_rank}] {''.join(results)}", flush=True)

        # Aggregate statistics from all server processes
        total_bw = aggregate_statistics(results)
        print(f"[rank {container_rank}] Total BW: {total_bw} GB/s", flush=True)

    else:
        # Wait until server has added its ip address to the dict
        while server_ip_dict.get("server_ip") is None:
            time.sleep(1)
        print(
            f"[rank {container_rank}] Server ip address added to dict, starting client connections...",
            flush=True,
        )

        # Launch all clients in parallel, retry any that get "Connection refused"
        server_ip = server_ip_dict["server_ip"]
        MAX_RETRIES = 10
        RETRY_DELAY = 2  # seconds

        # Build list of (idx, domain, gpu_id) for all clients
        client_args = [
            (idx, domain, math.floor(idx / 4))
            for idx, domain in enumerate(local_efa_domains)
        ]

        for attempt in range(MAX_RETRIES):
            # Launch all pending clients
            running = []
            for idx, domain, gpu_id in client_args:
                print(
                    f"[rank {container_rank}] Connecting domain {domain} to server {server_ip}",
                    flush=True,
                )
                proc = run_efa_write_client(
                    domain, server_ip, port=BASE_PORT + idx, gpu_id=gpu_id
                )
                running.append((idx, domain, gpu_id, proc))

            # Wait for all and collect failures
            failed = []
            for idx, domain, gpu_id, proc in running:
                stdout, _ = proc.communicate()
                if proc.returncode != 0 and "Connection refused" in stdout:
                    failed.append((idx, domain, gpu_id))
                    print(
                        f"[rank {container_rank}] Client {idx} ({domain}): connection refused",
                        flush=True,
                    )
                elif proc.returncode != 0:
                    print(
                        f"[rank {container_rank}] Client {idx} ({domain}) failed (exit {proc.returncode}):\n{stdout}",
                        flush=True,
                    )
                else:
                    print(
                        f"[rank {container_rank}] Client {idx} ({domain}) done",
                        flush=True,
                    )

            if not failed:
                break

            print(
                f"[rank {container_rank}] {len(failed)} clients got connection refused, retrying in {RETRY_DELAY}s (attempt {attempt + 1}/{MAX_RETRIES})",
                flush=True,
            )
            client_args = failed
            time.sleep(RETRY_DELAY)


# Run fi_rma_bw command for server
def run_efa_write_server(domain: str, port: int, gpu_id: int) -> subprocess.Popen:
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": "/opt/amazon/efa/lib",
        "FI_EFA_USE_DEVICE_RDMA": "1",  # Enable GPU Direct RDMA
    }
    cmd = [
        "/usr/local/bin/fi_rma_bw",
        "-d",
        domain,
        "-p",
        "efa",
        "-D",
        "cuda",  # We test GPU VRAM -> NIC instead of Host RAM -> NIC
        "-i",
        str(gpu_id),
        f"-b={port}",
        "-S",
        str((MB_PER_WRITE << 20) - 1),  # To keep < 1MB for read message protocol
        "-W",
        "128",  # Large window size
        "-w",
        "100",  # Warmup iterations to get accurate bandwidth
        "-I",
        "1000",  # 5000 iterations for more accurate bandwidth
    ]
    print(f"Running command: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )


# Run fi_rma_bw command for client with server ip
def run_efa_write_client(
    domain: str, server_ip: str, port: int, gpu_id: int
) -> subprocess.Popen:
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": "/opt/amazon/efa/lib",
        "FI_EFA_USE_DEVICE_RDMA": "1",  # Enable GPU Direct RDMA
    }
    cmd = [
        "/usr/local/bin/fi_rma_bw",
        "-d",
        domain,
        "-p",
        "efa",
        "-D",
        "cuda",  # We test GPU VRAM -> NIC instead of Host RAM -> NIC
        "-i",
        str(gpu_id),
        f"-b={port}",
        "-S",
        str((MB_PER_WRITE << 20) - 1),  # To keep < 1MB for read message protocol
        "-W",
        "128",
        "-w",
        "100",  # Warmup iterations to get accurate bandwidth
        "-I",
        "1000",
        server_ip,
    ]
    print(f"Running command: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )


# Get a sorted list of EFA RDM domain names by parsing fi_info output.
def get_local_efa_domains() -> list[str]:
    env = {**os.environ, "LD_LIBRARY_PATH": "/opt/amazon/efa/lib"}
    result = subprocess.run(
        ["/opt/amazon/efa/bin/fi_info", "-p", "efa"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    # Remove duplicate domains
    domains = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        # Only grab RDM domains (reliable datagram), skip DGRM (unreliable datagram)
        if line.startswith("domain:") and line.endswith("-rdm"):
            domains.add(line.split(":", 1)[1].strip())
    return sorted(domains)


# Builds the fabtests library with CUDA support
def build_fabtests():
    fabtests_dir = "/opt/fabtests-2.3.1"
    subprocess.run(
        [
            "./configure",
            "--with-libfabric=/opt/amazon/efa",
            "--with-cuda=/usr/local/cuda",
        ],
        check=True,
        cwd=fabtests_dir,
    )
    subprocess.run(
        ["make", "-j"],
        check=True,
        cwd=fabtests_dir,
    )
    subprocess.run(
        ["make", "install"],
        check=True,
        cwd=fabtests_dir,
    )


# Returns the total bandwidth across 16 devices in Gb/s
def aggregate_statistics(results: list[str]) -> float:
    bw_mb = []

    for output in results:
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0][0] == str(MB_PER_WRITE):
                bw_mb.append(
                    float(parts[4]) * 2
                )  # Convert unidirectional to bidirectional

    total_bw_gbps = sum(bw_mb) / 1024  # MB/s -> GB/s
    return round(total_bw_gbps, 2)


@app.local_entrypoint()
def main():
    with modal.Dict.ephemeral() as server_ip_dict:
        efa_bandwidth_test.remote(server_ip_dict)
