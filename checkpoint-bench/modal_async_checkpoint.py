"""
Checkpoint I/O benchmark on a Modal cluster: blocking dcp.save() vs
async dcp.async_save() with and without DefaultStager (PyTorch 2.9).

Includes a "pipeline" mode where staging, disk write, and volume.commit()
all run in a background thread — zero main-thread blocking.

    modal run modal_async_checkpoint.py --ckpt-gib 60
"""

import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import modal
import modal.experimental

N_NODES = 2
N_GPUS = 8
CKPT_MOUNT = "/vol/checkpoints"
DIM = 16384  # Linear(DIM, DIM, bias=False) → DIM² params, DIM²×2 bytes in bf16

MODES = [
    ("blocking", "blocking (save + commit)"),
    ("async", "async (staging on main)"),
    ("fully_async", "fully_async (DefaultStager)"),
    ("pipeline", "pipeline (all background)"),
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libibverbs-dev", "libibverbs1", "libhwloc15", "libnl-route-3-200")
    .pip_install("torch==2.9.0", "numpy")
)
app = modal.App("async-checkpoint-bench", image=image)
volume = modal.Volume.from_name("async-ckpt-bench", create_if_missing=True)


def _bench(local_rank: int):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.nn as nn
    from torch.distributed.checkpoint.staging import DefaultStager
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from torch.distributed.checkpoint.stateful import Stateful
    from torch.distributed.fsdp import fully_shard

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank, world = dist.get_rank(), dist.get_world_size()

    ckpt_gib = float(os.environ["CKPT_GIB"])
    num_layers = max(1, math.ceil(ckpt_gib * 2**30 / 2 / DIM**2))
    total_bytes = num_layers * DIM**2 * 2

    if rank == 0:
        print(
            f"\n{total_bytes / 2 / 1e9:.2f}B params (bf16) | "
            f"{total_bytes / 2**30:.1f} GiB ckpt | "
            f"{total_bytes / world / 2**30:.2f} GiB/GPU | "
            f"{num_layers} layers | {world} GPUs",
            flush=True,
        )

    class WeightStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(DIM, DIM, bias=False) for _ in range(num_layers)]
            )

    class ModelState(Stateful):
        def __init__(self, m):
            self.model = m

        def state_dict(self):
            return {"model": get_model_state_dict(self.model)}

        def load_state_dict(self, _sd):
            pass

    with torch.device("meta"):
        model = WeightStack().to(torch.bfloat16)
    for layer in model.layers:
        layer.to_empty(device=f"cuda:{local_rank}")
        nn.init.normal_(layer.weight, std=0.01)
        fully_shard(layer)
    fully_shard(model)
    dist.barrier()

    if rank == 0:
        print(
            f"Model sharded (peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)\n",
            flush=True,
        )

    executor = ThreadPoolExecutor(max_workers=1)
    sd = {"app": ModelState(model)}

    def commit():
        if rank == 0:
            volume.commit()

    def _bg_commit(future):
        future.staging_completion.result()
        future.upload_completion.result()
        commit()

    def timed_save(ckpt_id, mode):
        """Returns (call_ms, total_ms).
        call_ms  = main-thread stall.  total_ms = wall time until fully committed.
        """
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if mode == "blocking":
            dcp.save(sd, checkpoint_id=ckpt_id)
            commit()
            dist.barrier()
            dt = (time.perf_counter() - t0) * 1000
            return dt, dt

        if mode == "async":
            f = dcp.async_save(sd, checkpoint_id=ckpt_id)
            call_ms = (time.perf_counter() - t0) * 1000
            f.result()
            commit()
            dist.barrier()
            return call_ms, (time.perf_counter() - t0) * 1000

        if mode == "fully_async":
            f = dcp.async_save(sd, checkpoint_id=ckpt_id, async_stager=DefaultStager())
            call_ms = (time.perf_counter() - t0) * 1000
            f.staging_completion.result()
            f.upload_completion.result()
            commit()
            dist.barrier()
            return call_ms, (time.perf_counter() - t0) * 1000

        # pipeline: staging + disk write + commit all in background thread
        f = dcp.async_save(sd, checkpoint_id=ckpt_id, async_stager=DefaultStager())
        bg = executor.submit(_bg_commit, f)
        call_ms = (time.perf_counter() - t0) * 1000
        bg.result()
        dist.barrier()
        return call_ms, (time.perf_counter() - t0) * 1000

    results: dict[str, tuple[float, float]] = {}

    for key, _ in MODES:
        ckpt_id = os.path.join(CKPT_MOUNT, key)
        results[key] = timed_save(ckpt_id, key)
        if rank == 0:
            shutil.rmtree(ckpt_id, ignore_errors=True)
            volume.commit()
        dist.barrier()

    executor.shutdown(wait=False)

    if rank == 0:
        W = 78
        print("=" * W)
        print(f"{'Mode':<36} {'Call (ms)':>12} {'Total (ms)':>12} {'GiB/s':>12}")
        print("-" * W)
        baseline = None
        for key, label in MODES:
            call_ms, total_ms = results[key]
            tput = (total_bytes / 2**30) / (total_ms / 1000)
            baseline = baseline or call_ms
            print(f"{label:<36} {call_ms:>10.1f}ms {total_ms:>10.1f}ms {tput:>10.1f}")
        pipe_call = results["pipeline"][0]
        print("-" * W)
        print(
            f"Blocking reduction: {baseline:.0f}ms -> {pipe_call:.0f}ms ({(1 - pipe_call / baseline) * 100:.1f}%)"
        )
        print("=" * W + "\n", flush=True)

    dist.barrier()
    dist.destroy_process_group()


@app.function(
    gpu=f"H100:{N_GPUS}",
    experimental_options={"efa_enabled": True},
    volumes={CKPT_MOUNT: volume},
    timeout=60 * 60,
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def run_checkpoint_bench(ckpt_gib: float):
    import subprocess, sys, textwrap

    os.environ["CKPT_GIB"] = str(ckpt_gib)

    info = modal.experimental.get_cluster_info()
    launcher = "/tmp/_launch.py"
    with open(launcher, "w") as f:
        f.write(
            textwrap.dedent("""\
            import importlib, os, sys
            sys.path.insert(0, "/root")
            mod = importlib.import_module("modal_async_checkpoint")
            mod._bench(int(os.environ["LOCAL_RANK"]))
        """)
        )

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={N_NODES}",
        f"--nproc-per-node={N_GPUS}",
        f"--node-rank={info.rank}",
        f"--master-addr={info.container_ips[0]}",
        launcher,
    ]
    print(f"[node {info.rank}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(ckpt_gib: float = 60.0):
    run_checkpoint_bench.remote(ckpt_gib)
