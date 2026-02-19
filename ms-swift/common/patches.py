"""
Megatron/Modal monkey-patches extracted from inline definitions.

These patches fix known issues with:
1. FileSystemWriterAsync on Modal volumes (race conditions, slow writes)
2. Checkpoint save sync and adapter export
3. Loss mask weighting for tool-call tokens

Previously inlined in train/megatron_glm_remote.py (lines 35-400+).
"""

import os
from functools import wraps

import torch


def patch_filesystem_writer_for_modal():
    """
    Patch FileSystemWriterAsync to be more robust on Modal volumes:
    1. Force thread_count=1 to avoid multiprocess file write race conditions
    2. Add explicit fsync and barriers after file writes
    3. Make the write process synchronous per rank (still parallel across ranks)
    4. Replace get_nowait() with get(timeout=...) to handle slow volume writes
    """
    import queue
    from itertools import chain
    from megatron.core.dist_checkpointing.strategies import filesystem_async
    from torch.distributed.checkpoint.api import _wrap_exception

    OriginalWriter = filesystem_async.FileSystemWriterAsync
    _original_init = OriginalWriter.__init__

    @wraps(_original_init)
    def _patched_init(self, path, *args, **kwargs):
        kwargs['thread_count'] = 1
        _original_init(self, path, *args, **kwargs)

    OriginalWriter.__init__ = _patched_init

    _original_write = OriginalWriter.write_preloaded_data

    @staticmethod
    def _patched_write(*args, **kwargs):
        kwargs['use_fsync'] = True
        result = _original_write(*args, **kwargs)
        return result

    OriginalWriter.write_preloaded_data = _patched_write

    def _patched_retrieve(self):
        """Patched version with timeout instead of get_nowait for slow volumes."""
        assert self.write_buckets is not None

        if self.results_queue is None:
            write_results_or_exc = {}
        else:
            try:
                write_results_or_exc = self.results_queue.get(timeout=60.0)
            except queue.Empty:
                return _wrap_exception(
                    RuntimeError("results_queue timeout after 60s - Modal volume may be slow")
                )

        if isinstance(write_results_or_exc, Exception):
            try:
                raise RuntimeError(
                    f"Worker failure: {write_results_or_exc}"
                ) from write_results_or_exc
            except Exception as e:
                return _wrap_exception(e)

        write_results: dict = write_results_or_exc
        if len(write_results) != len(self.write_buckets):
            return _wrap_exception(
                RuntimeError(
                    f"Incomplete worker results (expected {len(self.write_buckets)},"
                    f" got {len(write_results)}. This probably indicates a worker failure."
                )
            )
        return list(chain.from_iterable(write_results.values()))

    OriginalWriter.retrieve_write_results = _patched_retrieve

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print("[Checkpoint Patch] FileSystemWriterAsync patched for Modal (thread_count=1, 60s timeout, forced fsync)")


def save_adapters_to_pt(model, save_path: str):
    """
    Save all adapter weights to a simple .pt file for debugging.

    Gathers adapters from ALL ranks to rank 0, then saves a complete snapshot.
    This bypasses distributed checkpointing entirely. Keys include rank prefix
    ("rank{N}::{name}") to preserve TP shards.

    Args:
        model: The model (or list of model chunks for PP)
        save_path: Path to save the .pt file
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        print(f"\n[PT Export] Starting adapter export to {save_path}")

    try:
        models = model if isinstance(model, list) else [model]

        local_adapters = {}
        for model_chunk in models:
            for name, param in model_chunk.named_parameters():
                if ".adapter." in name:
                    key = f"rank{rank}::{name}"
                    local_adapters[key] = param.data.detach().cpu().clone()

        local_count = len(local_adapters)

        if torch.distributed.is_initialized() and world_size > 1:
            all_adapters_list = [None] * world_size
            torch.distributed.all_gather_object(all_adapters_list, local_adapters)

            if rank == 0:
                merged_adapters = {}
                for rank_adapters in all_adapters_list:
                    if rank_adapters:
                        merged_adapters.update(rank_adapters)
                local_adapters = merged_adapters

        if rank != 0:
            return

        dense_count = 0
        expert_count = 0
        total_norm = 0.0

        for key, tensor in local_adapters.items():
            name = key.split("::", 1)[1] if "::" in key else key
            norm = tensor.norm().item()
            total_norm += norm ** 2
            if ".experts." in name or "expert" in name.lower():
                expert_count += 1
            else:
                dense_count += 1

        checkpoint = {
            "adapter_state": local_adapters,
            "metadata": {
                "dense_count": dense_count,
                "expert_count": expert_count,
                "total_count": dense_count + expert_count,
                "total_l2_norm": (total_norm ** 0.5),
                "world_size": world_size,
                "local_count_rank0": local_count,
            }
        }

        torch.save(checkpoint, save_path)

        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"\n[PT Export] Saved {dense_count + expert_count} adapters to {save_path}")
            print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
            print(f"  Dense: {dense_count}, Expert: {expert_count}")
            print(f"  L2 norm: {(total_norm ** 0.5):.6f}")
            print(f"  Gathered from {world_size} ranks")
        else:
            print(f"\n[PT Export] ERROR: File was not created at {save_path}")

    except Exception as e:
        print(f"\n[PT Export] EXCEPTION on rank {rank}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def patch_checkpoint_save_with_sync():
    """
    Monkey-patch Megatron's checkpoint save with:
    1. Filesystem sync on ALL ranks before and after save
    2. Proper barriers to ensure all ranks are synchronized
    3. Print LoRA weight norms for debugging train/inference discrepancy
    4. Save adapters to simple .pt file for comparison
    5. Disable PEFT filtering during save to preserve all TP shards
       (workaround for Megatron-Bridge#2240)
    """
    from megatron.bridge.training import checkpointing

    _original_save = checkpointing.save_checkpoint

    @wraps(_original_save)
    def _save_with_sync(state, model, *args, **kwargs):
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Print LoRA weight norms BEFORE save (rank 0 only)
        if rank == 0 and model is not None:
            print("\n" + "=" * 70)
            print("LORA ADAPTER INSPECTION AT CHECKPOINT SAVE")
            print("=" * 70)
            m = model[0] if isinstance(model, list) else model
            total_norm = 0.0
            dense_count = 0
            expert_count = 0
            router_count = 0
            sample_names = []
            for name, param in m.named_parameters():
                if ".adapter." in name and param.numel() > 0:
                    norm = param.data.norm().item()
                    total_norm += norm ** 2
                    if len(sample_names) < 3:
                        sample_names.append(f"{name}: norm={norm:.4f}")
                    if ".experts." in name or "expert" in name.lower():
                        expert_count += 1
                    elif "router" in name.lower():
                        router_count += 1
                    else:
                        dense_count += 1
            total_count = dense_count + expert_count + router_count
            print(f"  Dense adapters: {dense_count}")
            print(f"  Expert adapters: {expert_count}")
            print(f"  Router adapters: {router_count}")
            print(f"  TOTAL: {total_count} adapters")
            print(f"  Combined L2 norm: {(total_norm ** 0.5):.6f}")
            if expert_count == 0:
                print("\n  WARNING: No expert adapters in memory!")
            print("\n  Sample adapter names:")
            for s in sample_names:
                print(f"    {s}")
            print("=" * 70 + "\n")

        if rank == 0:
            print(f"[Checkpoint] Pre-save sync on all {world_size} ranks...")

        os.sync()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Disable PEFT filtering during save to preserve all TP shards
        # (workaround for Megatron-Bridge#2240)
        peft_backup = None
        if hasattr(state, 'cfg') and hasattr(state.cfg, 'peft') and state.cfg.peft is not None:
            peft_backup = state.cfg.peft
            state.cfg.peft = None
            if rank == 0:
                print("[Checkpoint] PEFT filter DISABLED for save (preserves all TP shards)")

        try:
            result = _original_save(state, model, *args, **kwargs)
        finally:
            if peft_backup is not None:
                state.cfg.peft = peft_backup

        os.sync()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Save adapters to .pt file
        if model is not None and hasattr(state, 'train_state'):
            iteration = state.train_state.step
            cfg = state.cfg
            if hasattr(cfg, 'checkpoint') and cfg.checkpoint.save:
                pt_path = f"{cfg.checkpoint.save}/iter_{iteration:07d}_adapters.pt"
                save_adapters_to_pt(model, pt_path)
                os.sync()

        if rank == 0:
            print("[Checkpoint] Save completed and synced")

        return result

    checkpointing.save_checkpoint = _save_with_sync

    from megatron.bridge.training import train
    train.save_checkpoint = _save_with_sync

    print("[Checkpoint Patch] Patched checkpointing.save_checkpoint AND train.save_checkpoint")