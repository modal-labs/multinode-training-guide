"""
Fine-tunes Llama-2-7B on Go and Rust code using distributed training.

This script performs full-parameter fine-tuning of Meta's Llama-2-7B model on a dataset of Go and Rust code.
It uses Hugging Face's SFTTrainer for supervised fine-tuning and PyTorch's Fully Sharded Data
Parallel (FSDP) for efficient distributed training across multiple GPUs and nodes.

Key Points:
- Loads the Llama-2-7B model and tokenizer from Hugging Face.
- Builds a packed dataset from Arrow files containing Go and Rust code.
- Uses SFTTrainer from TRL library for supervised fine-tuning.
- Implements FSDP for memory-efficient weight distribution.
- Supports gradient accumulation and mixed precision training.
- Integrates with Weights & Biases for experiment tracking.

Requirements:
- PyTorch with CUDA support
- transformers library
- TRL library for supervised fine-tuning
- datasets library for data handling
- Weights & Biases (optional) for experiment tracking
- HuggingFace token with Llama 2 access

The script expects data in arrow format organized in go/ and rust/ subdirectories.
"""

import argparse
import os

from pathlib import Path
import datasets
import torch
from datasets import (
    load_dataset,
)

import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig
from torch.profiler import (
    profile,
    schedule,
    ProfilerActivity,
    tensorboard_trace_handler,
)
from transformers.trainer_callback import TrainerCallback

if os.environ.get("WANDB_PROJECT"):
    import wandb


MODEL_NAME = "meta-llama/Llama-2-7b-hf"


# New Profiler Callback Class
class TorchProfilerCallback(TrainerCallback):
    def __init__(
        self,
        wait,
        warmup,
        active,
        repeat,
        out_dir,
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
    ):
        self.prof = profile(
            activities=list(activities),
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=tensorboard_trace_handler(out_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=True,
            with_modules=False,
        )
        self.prof.__enter__()

    # 1 step == 1 optimizer update
    def on_step_end(self, *_, **__):
        self.prof.step()

    # make sure files are flushed even on KeyboardInterrupt
    def on_train_end(self, *_, **__):
        self.prof.__exit__(None, None, None)


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=os.environ["HF_TOKEN"],
        cache_dir=os.environ.get("MODEL_CACHE_DIR"),
    )
    tok.pad_token = tok.eos_token
    return tok


def download_llama2(cache_dir: str | None = None):
    """Fetch Llama‑2‑7B weights & tokenizer; returns (model, tokenizer)."""
    token = os.environ["HF_TOKEN"]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        token=token,
        cache_dir=cache_dir,
    )
    # KV caching breaks activation checkpointing
    model.config.use_cache = False

    return model


def build_packed_ds(data_dir: Path, buffer_size: int, block: int):
    """Builds a packed dataset for training by:
    1. Loading arrow files from Go and Rust directories
    2. Shuffling the data with a buffer
    3. Tokenizing the content and adding EOS tokens
    4. Packing sequences into fixed-length blocks
    5. Yielding packed sequences with attention masks

    Args:
        data_dir: Directory containing Go and Rust arrow files
        tokenizer: Tokenizer to use for encoding text
        buffer_size: Size of shuffle buffer
        block: Fixed sequence length for packed sequences
    """

    def gen():
        worker = torch.utils.data.get_worker_info()
        rank = worker.id if worker else 0
        n_ranks = worker.num_workers if worker else 1

        tokenizer = get_tokenizer()
        eos_id = tokenizer.eos_token_id

        data_files = []
        for lang_dir in ["go", "rust"]:
            data_files.extend(
                [str(f) for f in (data_dir / lang_dir).glob("**/*.arrow")]
            )
        print(f"Found {len(data_files)} files for training.")

        ds_iterable = load_dataset(
            "arrow",
            data_files=data_files,
            split="train",
            streaming=True,
        )

        # Shuffle the dataset so we don't overtrain on a single shard.
        ds_iterable = ds_iterable.shuffle(buffer_size=buffer_size, seed=44).shard(
            num_shards=n_ranks,
            index=rank,
        )

        buf = []
        consumed_samples = 0
        for rec in ds_iterable:
            # Log the number of consumed samples to Weights & Biases for
            # monitoring purposes.
            consumed_samples += 1
            if os.environ.get("WANDB_PROJECT") and consumed_samples % 100 == 0:
                wandb.log({"consumed_samples": consumed_samples}, commit=False)

            # Tokenize the content and add EOS tokens.
            buf.extend(
                tokenizer(rec["content"], add_special_tokens=False)["input_ids"]
                + [eos_id]
            )

            # Yield packed sequences with attention masks.
            while len(buf) >= block:
                yield {"input_ids": buf[:block], "attention_mask": [1] * block}
                del buf[:block]

    return datasets.IterableDataset.from_generator(gen)


def parse_args():
    p = argparse.ArgumentParser(description="Streamed SFT of Llama‑2‑7B on Go and Rust")

    # Volume path arguments
    p.add_argument("--data_dir", required=True, help="Folder of all datasets")
    p.add_argument("--output_dir", required=True, help="Where to write checkpoints")
    p.add_argument(
        "--model_cache_dir", type=str, default=None, help="Where to cache the model"
    )

    # Training arguments
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--block_size", type=int, default=4096, help="Context length tokens")
    p.add_argument("--profile", action="store_true", help="Enable profiling")

    return p.parse_args()


def main():
    # Disable tokenizer parallelism to avoid deadlocks. We're offloading
    # tokenization to the workers, so we don't need to parallelize.
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"

    args = parse_args()

    # Explicitly initialize the process group to avoid warnings about
    # NCCL not being able to pick a GPU.
    if not dist.is_initialized():
        print(f"Initializing process group for rank {os.environ.get('RANK')}")
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{os.environ.get('LOCAL_RANK')}"),
        )

    print(f"Rank {os.environ.get('RANK')} using GPU {torch.cuda.current_device()}")

    model = download_llama2(args.model_cache_dir)
    tokenizer = get_tokenizer()

    # Buffer size for shuffling data
    BUFFER_SIZE = 20_000
    train_ds = build_packed_ds(Path(args.data_dir), BUFFER_SIZE, args.block_size)

    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, pad_to_multiple_of=8
    )

    callbacks = []
    if args.profile:
        print("🔎 Profiling enabled")
        profiler_callback = TorchProfilerCallback(
            # Start profiling after 5 iterations. Profile 10 times then stop.
            warmup=5,
            active=3,
            repeat=3,
            wait=0,
            out_dir=f"{args.output_dir}/traces/{os.environ.get('MODAL_TASK_ID', 'local')}/{os.environ.get('RANK', '0')}",
        )
        callbacks.append(profiler_callback)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        seed=1234,
        # Hyperparameters
        per_device_train_batch_size=args.batch_per_device,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=8e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        max_seq_length=args.block_size,
        # Checkpointing and logging
        save_steps=125,
        logging_steps=1,
        report_to="wandb" if os.environ.get("WANDB_PROJECT") else "none",
        run_name=os.environ.get("WANDB_RUN_NAME"),
        # FSDP configuration
        # Note that this FSDP configuration is intentionally simple. To further
        # optimize performance, consider:
        # - disabling activation checkpointing
        # - enabling forward prefetching
        # - using `shard_grad_op` to localize weights to a single node
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "activation_checkpointing": True,
            "forward_prefetch": False,
            "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        },
        # Training loop exit conditions
        num_train_epochs=args.epochs,
        max_steps=10000,
        # Use dataloader workers so we don't block training on data loading.
        dataloader_num_workers=1,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        dataloader_multiprocessing_context="fork",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=cfg,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
