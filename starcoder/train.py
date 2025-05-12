#!/usr/bin/env python3
"""
train.py — Streamed SFT of Llama‑2‑7B on StarCoder jsonl.gz shards with FSDP.

Launch with torchrun on one or many nodes; see example commands in the chat.

Requirements (CUDA 12.x build):
  pip install "torch==2.3.0+cu121" --index-url https://download.pytorch.org/whl/cu121
  pip install transformers==4.40.0 datasets==2.20.0 trl==0.7.10 accelerate==0.29.2 \
              flash-attn==2.5.5 bitsandbytes==0.43.2

Each JSON‑lines row must contain a top‑level key "content" with raw source text.

The script keeps <1 GB of host RAM regardless of dataset size by streaming +
reservoir‑style buffered shuffling.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import (
    load_dataset,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────── helpers ────────────────────────────────

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def download_llama2(cache_dir: str | None = None):
    """Fetch Llama‑2‑7B weights & tokenizer; returns (model, tokenizer)."""
    token = os.environ["HF_TOKEN"]
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=token,
        cache_dir=cache_dir,
    )
    tok.pad_token = tok.eos_token  # ensure causal masking works for padding

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        torch_dtype="auto",
        token=token,
    )
    return model, tok


def build_streaming_dataset(
    data_dir: Path,
    tokenizer,
    buffer_size: int = 20_000,
    block_size: int = 4096,
):
    """
    Stream Arrow shards, shuffle, tokenize the 'content' field and yield
    *exactly one fixed-length record per chunk* (short docs are padded).
    """

    # 1. Stream & reservoir-shuffle whole documents
    ds = load_dataset(
        "arrow",
        data_files=str(data_dir / "**" / "*.arrow"),
        split="train",
        streaming=True,
    ).shuffle(buffer_size=buffer_size, seed=44)

    eos_id = tokenizer.eos_token_id

    def batch_tokenize_and_chunk(batch):
        input_ids, attention_masks = [], []

        for text in batch["content"]:
            ids = tokenizer(text, add_special_tokens=False, truncation=False)[
                "input_ids"
            ]

            # ensure EOS
            if ids[-1] != eos_id:
                ids.append(eos_id)

            # pad short docs so we always get ≥1 block
            if len(ids) < block_size:
                ids += [eos_id] * (block_size - len(ids))

            usable = (len(ids) // block_size) * block_size
            for i in range(0, usable, block_size):
                input_ids.append(ids[i : i + block_size])
                attention_masks.append([1] * block_size)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }

    # 2. explode into rows so each sample is already rectangular
    ds = ds.map(
        batch_tokenize_and_chunk,
        batched=True,
        batch_size=512,  # keeps RAM tiny; tune freely
        remove_columns=["content"],
    )

    return ds  # lists-of-ints; default TRL collator is happy


# ───────────────────────────── argparse / main ───────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Streamed SFT of Llama‑2‑7B on StarCoder")
    p.add_argument("--data_dir", required=True, help="Folder of all datasets")
    p.add_argument("--output_dir", required=True, help="Where to write checkpoints")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument(
        "--buffer_size", type=int, default=20_000, help="Shuffle reservoir size"
    )
    p.add_argument("--block_size", type=int, default=4096, help="Context length tokens")
    p.add_argument("--local_rank", type=int, default=-1, help="(set by torchrun)")
    p.add_argument(
        "--model_cache_dir", type=str, default=None, help="Where to cache the model"
    )
    p.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    p.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    return p.parse_args()


def main():
    args = parse_args()

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    model, tokenizer = download_llama2(args.model_cache_dir)
    train_ds = build_streaming_dataset(
        Path(args.data_dir), tokenizer, args.buffer_size, args.block_size
    )

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_per_device,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=2000,
        eval_steps=2000,
        bf16=True,
        fsdp="full_shard auto_wrap",
        gradient_checkpointing=True,
        max_seq_length=args.block_size,
        seed=1234,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.wandb_run_name,
        max_steps=10000,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
