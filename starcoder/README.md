# Fine-tuning LLaMA 2 on StarCoder

This directory contains code for fine-tuning LLaMA 2 on the StarCoder dataset using Modal's multi-node infrastructure. The goal is to enhance LLaMA 2's coding capabilities by training it on high-quality programming code from the StarCoder dataset.

## Why Fine-tune LLaMA 2 on Code?

While LLaMA 2 is a powerful general-purpose language model, it wasn't specifically trained on programming tasks. By fine-tuning it on the StarCoder dataset, we can:

- Improve code completion capabilities
- Enhance understanding of programming languages and patterns
- Better handle coding-specific tasks and documentation
- Learn from high-quality, diverse programming examples

## Quick Start

1. First, download and prepare the StarCoder dataset:
```bash
modal run download_dataset.py::orchestrate
```

2. Then start fine-tuning (add `-d` to detach and run in background):
```bash
modal run -d modal_train.py::train_multi_node
```

## Dataset

The training uses the [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) dataset from Hugging Face, which contains high-quality programming code from various sources. The `download_dataset.py` script:

- Downloads the dataset from Hugging Face
- Saves it to a Modal volume for persistence
- Validates the downloaded data

## Training Configuration

The training script (`modal_train.py`) implements supervised fine-tuning (SFT) with:

- Multi-node distributed training with RDMA
- Gradient accumulation for larger effective batch sizes
- Weights & Biases integration for experiment tracking
- Automatic mixed precision (AMP) training
- Checkpointing to Modal volumes

Key configuration parameters:

```python
# Number of nodes in the cluster (1-8)
n_nodes = 2
# GPUs per node (typically matches hardware)
n_proc_per_node = 8
# Global batch size
global_batch_size = 256
# Per-device batch size
per_device_batch_size = 4
```

## Hardware Requirements

- GPU: H100 (8 per node)
- Network: High-speed interconnect (RDMA) for multi-node

## Performance

The implementation is optimized for Modal's infrastructure with:

- RDMA support for fast inter-node communication
- Gradient accumulation for memory efficiency
- Automatic mixed precision training
- Efficient data loading and preprocessing

## Monitoring

Training progress can be monitored through:

- Command line output showing loss metrics
- Weights & Biases dashboard (if configured)
- Modal web UI showing GPU utilization and logs
