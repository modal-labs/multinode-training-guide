import os

import torch
import torch.distributed as dist

# Default settings from EleutherAI cookbook
WARMUP_ITERS, TRIALS = 5, 50

# these emulate the payload which will become a M * N * 4-sized tensor below
N = 500000
M = 2000


def sync_all():
    torch.cuda.synchronize()
    dist.barrier()


def timed_allreduce(mat, start_event, end_event, warmup_iters, iters):
    sync_all()
    for _ in range(warmup_iters):
        dist.all_reduce(mat)
    sync_all()

    start_event.record()
    for _ in range(iters):
        dist.all_reduce(mat)
    end_event.record()

    sync_all()
    duration = start_event.elapsed_time(end_event) / 1000
    avg_duration = duration / iters

    n = dist.get_world_size()
    size = M * N * 4  # 4 is 4 bytes in fp32
    # note that this is following the same math as NVIDIA/nccl-tests
    algbw = torch.tensor([size / avg_duration]).cuda(local_rank)

    # calculate mean across all ranks
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n

    return algbw.item()


def run(local_rank):
    is_global_rank_0 = dist.get_rank() == 0

    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    algbw = timed_allreduce(mat, start_event, end_event, warmup_iters=WARMUP_ITERS, iters=TRIALS)

    # the 2*(n-1)/n busbw correction factor specific to all-reduce is explained here:
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    # busbw reflects how optimally the hardware is used
    n = dist.get_world_size()
    busbw = algbw * (2 * (n - 1) / n)

    if is_global_rank_0:
        print(
            f"The average bandwidth of all_reduce with a {M * N * 4 / 1e9}GB payload ({TRIALS} trials, {n} ranks):\n",
            f"algbw: {algbw / 1e9:.3f} GBps ({algbw * 8 / 1e9:.1f} Gbps)\n",
            f"busbw: {busbw / 1e9:.3f} GBps ({busbw * 8 / 1e9:.1f} Gbps)\n",
        )


def init_processes(local_rank, fn, backend="nccl"):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend, device_id=torch.device(f"cuda:{local_rank}"))
    if dist.get_rank() == 0:
        print("Starting benchmark...")

    fn(local_rank)

    sync_all()
    dist.destroy_process_group()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_processes(local_rank=local_rank, fn=run)
