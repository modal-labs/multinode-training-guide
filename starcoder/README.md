# StarCoder: Fine-tuning Llama 2 for Go and Rust Generation

This example demonstrates how to fine-tune Meta's Llama-2-7B model to generate high-quality Go and Rust code. While Llama 2 is a powerful general-purpose language model, it can struggle with generating syntactically correct and idiomatic code in specific programming languages. This training setup aims to enhance its coding capabilities specifically for Go and Rust.

## Overview

The training pipeline consists of three main stages:
1. Dataset ingestion from the StarCoder dataset
2. Multi-node distributed training using PyTorch FSDP
3. Model evaluation on coding tasks

## Prerequisites

- Modal account with access to H100 GPUs
- Hugging Face account with access to Llama 2 model family
- Weights & Biases account (optional, for experiment tracking)

## Quick Start

### 1. Dataset Preparation

First, download and prepare the StarCoder dataset:

```bash
modal run download_dataset.py::ingest_dataset
```

This command:
- Downloads code samples from the StarCoder dataset
- Processes and validates the data
- Stores it in a Modal volume for training

### 2. Training

Launch multi-node training using torchrun (recommended):

```bash
modal run modal_train.py::train_multi_node --launch-type torchrun
```

Note that [Accelerate](https://github.com/huggingface/accelerate) is also supported through the `--launch-type accelerate` flag.

This command:
- Launches a cluster of 2 nodes with 8 H100 GPUs each
- Uses PyTorch FSDP (Fully Sharded Data Parallel) for efficient distributed training
- Automatically configures RDMA for high-speed inter-node communication
- Saves checkpoints periodically to a Modal volume

Key training parameters (configurable in `modal_train.py`):
- Global batch size: 2048
- Per-device batch size: 16
- Gradient accumulation steps: 2
- Learning rate: 8e-5 with cosine decay
- Context length: 4096 tokens

### 3. Evaluation

Evaluate the trained model on Go and Rust coding tasks:

```bash
modal run evaluate.py::eval_model --run-name 'starcoder-nodes_8-gpus_8-batch_2048-per_device_16-grad_accum_2'
```

This command:
- Loads checkpoints from the specified training run
- Evaluates the model on a set of coding prompts
- Compares performance against the base Llama 2 model
- Prints generation results

## Evaluation Prompts

The evaluation covers common coding tasks in both Go and Rust:
- Fibonacci number calculation (naive and efficient implementations)
- String manipulation (rotation checking, character frequency)
- And more...

## Performance Monitoring

If you've configured Weights & Biases:
- Training metrics are logged in real-time
- You can monitor loss, learning rate, and GPU utilization
- Compare different training runs and hyperparameters

## Customization

You can modify various aspects of the training:
- Adjust the number of nodes and GPUs in `modal_train.py`
- Change training hyperparameters in `train.py`
- Add new evaluation prompts in `evaluate.py`
- Configure data preprocessing in `download_dataset.py`

## Scaling

Sample consumption scales with the number of nodes and GPUs.
This scaling is sublinear but can be improved by increasing the global batch size and tuning FSDP configurations.

| Nodes | GPUs | Samples per minute |
|-------|------|--------------------|
| 2     | 8    | 7675               |
| 4     | 8    | 4981               |
| 8     | 8    | 2898               |


## Contributing

Feel free to:
- Add support for more programming languages
- Implement additional evaluation metrics
- Optimize the training configuration
