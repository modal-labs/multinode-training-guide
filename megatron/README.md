# Megatron: LoRA Fine-tuning GLM-4.7 (358B MoE)

This example demonstrates how to fine-tune the GLM-4.7 358B Mixture-of-Experts model using NVIDIA NeMo's Megatron framework with LoRA (Low-Rank Adaptation) on Modal's multi-node infrastructure.

## Overview

GLM-4.7 is a 358B parameter MoE model with 92 layers and 160 experts. Fine-tuning a model this large requires:
- Multi-node distributed training across 32 GPUs
- Advanced parallelism strategies (Tensor, Pipeline, Expert, Data parallelism)
- Memory-efficient techniques like LoRA and activation recomputation

The training pipeline consists of three main stages:
1. Model download and conversion to Megatron format
2. Dataset preparation (LongMIT-128K)
3. Multi-node distributed LoRA training

## Prerequisites

- Modal account with access to H100 GPUs
- Hugging Face account with access to GLM-4.7
- Weights & Biases account for experiment tracking

Configure Modal secrets:
```bash
modal secret create huggingface-secret HF_TOKEN=<your-token>
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

## Quick Start

### 1. Download and Convert Model

Download GLM-4.7 from Hugging Face and convert to Megatron checkpoint format:

```bash
modal run --detach modal_train.py::download_and_convert
```

This command:
- Downloads the 358B model weights (~700GB) on CPU
- Converts to Megatron's `torch_dist` checkpoint format on a B200 GPU (requires ~1TB memory)
- Stores in a Modal volume for training

**Note:** This step takes several hours due to model size. Run with `--detach` to avoid timeouts.

### 2. Prepare Dataset

Download and preprocess the LongMIT-128K dataset:

```bash
modal run modal_train.py::prep_dataset
```

This command:
- Downloads the dataset from Hugging Face
- Builds a long-context SFT prompt from passages + question
- Filters to <= 128K tokens and converts to JSONL for Megatron SFT
- Builds index files for efficient data loading

### 3. Training

Launch multi-node LoRA training:

```bash
modal run --detach modal_train.py::train_lora
```

This command:
- Launches a cluster of 4 nodes with 8 H100 GPUs each (32 GPUs total)
- Uses torchrun for distributed training coordination
- Enables RDMA for high-speed inter-node communication
- Saves checkpoints periodically to a Modal volume
- Logs metrics to Weights & Biases (run ID generated automatically)

**Memory Note:** The default configuration (mbs=1, full recompute) is chosen for long-context examples. Increase micro batch size only if you have headroom, or increase tensor/pipeline parallelism.

### 4. Evaluation

Run Megatron native evaluation on the latest LoRA checkpoint using the
existing LongMIT-128K dataset:

```bash
modal run modal_train.py::eval_lora
```

Optional overrides:

```bash
# Use a specific checkpoint subdirectory under /checkpoints
modal run modal_train.py::eval_lora --checkpoint-subdir glm47_lora

# Point directly to a checkpoint dir or use a run id suffix
modal run modal_train.py::eval_lora --checkpoint-dir /checkpoints/glm47_lora
modal run modal_train.py::eval_lora --run-id <run-id>

# Adjust evaluation settings
modal run modal_train.py::eval_lora --context 128k --lora-rank 128 --eval-iters 2
```

The evaluator will re-use the existing preprocessed dataset at
`/data/longmit-128k`. If `validation.jsonl` is missing, it creates a
lightweight link to `training.jsonl` for the duration of the run.

## Training Configuration

### Parallelism Strategy

The 358B MoE model uses a combination of parallelism strategies across 32 GPUs:

| Parallelism | Value | Description |
|-------------|-------|-------------|
| Tensor (TP) | 2 | Splits attention/FFN across GPUs |
| Pipeline (PP) | 4 | Distributes layers across pipeline stages |
| Expert (EP) | 4 | Distributes MoE experts |
| Data (DP) | 4 | Replicates for data parallelism |

**Total:** N_NODES × GPUs_per_node / (TP × PP) = 4 × 8 / (2 × 4) = 4 (DP)

### LoRA Configuration

```python
LoRA(
    dim=128,      # LoRA rank
    alpha=32,     # Scaling factor
    dropout=0.05, # Dropout rate
)
```

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Global batch size | 32 |
| Micro batch size | 1 |
| Sequence length | 131,072 |
| Learning rate | 1e-4 (cosine decay) |
| Warmup iterations | 50 |
| Training iterations | 650 |
| Checkpoint interval | 130 |

### Memory Optimizations

- **Activation recomputation:** Full recomputation with uniform method (configurable via `--recompute_num_layers`)
- **Mixed precision:** BF16 mixed precision training
- **Grouped GEMM:** Optimized MoE computation
- **Sequence parallel:** Enabled for memory efficiency
- **Operator fusion:** Masked softmax, bias activation, bias dropout, and RoPE fusion enabled
- **TransformerBlock patch:** Custom gradient flow fix for LoRA + recompute compatibility

## Monitoring

Training metrics are logged to Weights & Biases:
- Loss curves (LM loss, MTP loss, load balancing loss)
- Learning rate schedule
- Gradient norms
- Samples processed

You can also monitor GPU utilization via Modal's dashboard or by exec-ing into a running container:

```bash
modal container list
# This command can also be found in the Modal dashboard under the "Containers" tab.
modal shell <container-id>
nvidia-smi
```

## Customization

- Adjust parallelism in `train.py` (TP, PP, EP, CP sizes)
- Modify LoRA hyperparameters (dim, alpha, dropout)
- Change training parameters (batch size, learning rate, iterations)
- Use a different dataset by modifying `prep_dataset()`
- Tune memory usage with `--recompute_num_layers` (1=full recompute, higher=less recompute overhead)
- Run `memory_calc.py` to explore different parallelism configurations before training

## Files

| File | Description |
|------|-------------|
| `modal_train.py` | Modal app with download, convert, prep, and train functions |
| `train.py` | Training script executed via torchrun on each node |
| `eval.py` | Evaluation script executed via torchrun on each node |
| `memory_calc.py` | GPU memory calculator for exploring parallelism configurations |

