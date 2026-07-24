"""
Load testing for K2 Inference deployment using Locust.

Run distributed load tests against the K2 inference server to measure performance
and identify bottlenecks under various load conditions.
"""

import os
import subprocess
from datetime import datetime
from time import sleep

import modal


# Image with locust and dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "locust==2.33.1",
        "requests==2.32.4",
    )
    .add_local_file("locustfile.py", "/root/locustfile.py")
)

# Volume for storing load test results
results_volume = modal.Volume.from_name("k2-loadtest-results", create_if_missing=True)

app = modal.App("k2-loadtest", image=image)

# Default test configuration
DEFAULT_SPAWN_RATE = "10"
DEFAULT_USERS = "100"
DEFAULT_TIME = "5m"


@app.function(
    volumes={"/results": results_volume},
    timeout=60 * 60,  # 1 hour timeout
)
def run_load_test(
    target_url: str,
    users: str = DEFAULT_USERS,
    spawn_rate: str = DEFAULT_SPAWN_RATE,
    time: str = DEFAULT_TIME,
    headless: bool = True,
):
    """
    Run locust load test against K2 inference deployment.

    Args:
        target_url: URL of the K2 inference deployment
        users: Number of concurrent users to simulate
        spawn_rate: Rate to spawn users (users per second)
        time: Duration to run the test (e.g., "5m", "30s", "1h")
        headless: Run in headless mode (no web UI)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/results/k2_loadtest_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Build locust command
    cmd = [
        "locust",
        "--locustfile",
        "/root/locustfile.py",
        "--host",
        target_url,
        "--users",
        str(users),
        "--spawn-rate",
        str(spawn_rate),
        "--run-time",
        str(time),
        "--csv",
        f"{results_dir}/results",
        "--html",
        f"{results_dir}/report.html",
        "--logfile",
        f"{results_dir}/locust.log",
        "--loglevel",
        "INFO",
    ]

    if headless:
        cmd.extend(["--headless", "--autostart", "--autoquit", "10"])

    # locustfile.py is already included in the image

    print(f"Running load test with command: {' '.join(cmd)}")
    print(f"Results will be saved to: {results_dir}")

    # Run the test
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Save command output
    with open(f"{results_dir}/output.log", "w") as f:
        f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")
        f.write(f"Return code: {result.returncode}\n")

    if result.returncode != 0:
        print(f"Load test failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
    else:
        print("Load test completed successfully")
        print(f"Results saved to volume at: {results_dir}")

    return {
        "success": result.returncode == 0,
        "results_path": results_dir,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(
    cpu=8,
    volumes={"/results": results_volume},
    timeout=60 * 60 * 2,  # 2 hour timeout
)
def run_distributed_load_test(
    target_url: str,
    users: str = "500",
    spawn_rate: str = "50",
    time: str = "10m",
    workers: int = 4,
):
    """
    Run distributed locust load test with multiple workers.

    Args:
        target_url: URL of the K2 inference deployment
        users: Total number of concurrent users across all workers
        spawn_rate: Rate to spawn users (users per second)
        time: Duration to run the test
        workers: Number of worker processes
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/results/k2_distributed_loadtest_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # locustfile.py is already included in the image

    # Start master process
    master_cmd = [
        "locust",
        "--locustfile",
        "/root/locustfile.py",
        "--host",
        target_url,
        "--users",
        str(users),
        "--spawn-rate",
        str(spawn_rate),
        "--run-time",
        str(time),
        "--csv",
        f"{results_dir}/results",
        "--html",
        f"{results_dir}/report.html",
        "--logfile",
        f"{results_dir}/master.log",
        "--loglevel",
        "INFO",
        "--master",
        "--headless",
        "--autostart",
        "--autoquit",
        "10",
        "--expect-workers",
        str(workers),
    ]

    print(f"Starting distributed load test with {workers} workers")
    print(f"Command: {' '.join(master_cmd)}")

    # Start worker processes
    worker_processes = []
    for i in range(workers):
        worker_cmd = [
            "locust",
            "--locustfile",
            "/root/locustfile.py",
            "--worker",
            "--master-host",
            "localhost",
            "--logfile",
            f"{results_dir}/worker_{i}.log",
        ]
        worker_proc = subprocess.Popen(worker_cmd)
        worker_processes.append(worker_proc)
        print(f"Started worker {i}")

    # Give workers time to connect
    sleep(5)

    # Start master
    result = subprocess.run(master_cmd, capture_output=True, text=True)

    # Wait for workers to finish and clean them up
    for i, worker_proc in enumerate(worker_processes):
        worker_proc.terminate()
        worker_proc.wait()
        print(f"Worker {i} terminated")

    # Save results
    with open(f"{results_dir}/output.log", "w") as f:
        f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")
        f.write(f"Return code: {result.returncode}\n")

    if result.returncode != 0:
        print(f"Distributed load test failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
    else:
        print("Distributed load test completed successfully")
        print(f"Results saved to volume at: {results_dir}")

    return {
        "success": result.returncode == 0,
        "results_path": results_dir,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "workers": workers,
    }


@app.local_entrypoint()
def main(
    target_url: str,
    users: str = DEFAULT_USERS,
    spawn_rate: str = DEFAULT_SPAWN_RATE,
    time: str = DEFAULT_TIME,
    distributed: bool = False,
    workers: int = 4,
):
    """
    Run load test against K2 inference deployment.

    Examples:
        # Basic load test
        modal run load_test.py --target-url https://your-k2-deployment.modal.run

        # Custom parameters
        modal run load_test.py --target-url https://your-k2-deployment.modal.run --users 200 --time 10m

        # Distributed test
        modal run load_test.py --target-url https://your-k2-deployment.modal.run --distributed --workers 8 --users 1000
    """
    if distributed:
        print(f"Running distributed load test with {workers} workers")
        result = run_distributed_load_test.remote(
            target_url=target_url,
            users=users,
            spawn_rate=spawn_rate,
            time=time,
            workers=workers,
        )
    else:
        print("Running single-process load test")
        result = run_load_test.remote(
            target_url=target_url,
            users=users,
            spawn_rate=spawn_rate,
            time=time,
        )

    print(f"Load test {'completed successfully' if result['success'] else 'failed'}")
    if not result["success"]:
        print(f"Error details: {result['stderr']}")
