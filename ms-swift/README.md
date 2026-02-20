# Multi-Node GLM-4.7 Training with ms-swift on Modal

Fine-tune [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) (a Mixture-of-Experts model) across multiple nodes
using [ms-swift v4](https://swift.readthedocs.io/en/latest/) with Megatron-LM on [Modal](https://modal.com).

GLM-4.7 is a 60-expert MoE model. This setup uses Megatron-style distributed training with LoRA,
running on B200 GPUs with RDMA networking.

## Prerequisites

1. A Modal account with access to B200 GPUs
2. Two Modal secrets configured:
   - `huggingface-secret` — must contain `HF_TOKEN`
   - `wandb-secret` — must contain `WANDB_API_KEY`

Create them via the Modal dashboard or CLI:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
modal secret create wandb-secret WANDB_API_KEY=xxxxx
```

## Quick Start

### 1. Download the model

```bash
modal run modal_train.py::download_model
```

Downloads `zai-org/GLM-4.7` into the `glm-4-7-models` volume (~50 GB).

### 2. Prepare training data

```bash
modal run modal_train.py::prepare_dataset \
  --hf-dataset openai/gsm8k \
  --data-folder gsm8k \
  --split train
```

Converts a HuggingFace dataset into ms-swift's expected JSONL format (user/assistant message pairs)
and writes it to the `example-msswift-glm-4-7-data` volume.

For custom datasets, use `--input-col` and `--output-col` to specify which columns map to the
user prompt and assistant response.

### 3. Train

```bash
modal run --detach modal_train.py::train_model
```

This launches a 4-node distributed training job (32 B200 GPUs by default).

## Cluster Sizing

Control the number of nodes via the `N_NODES` environment variable (evaluated at deploy time):

```bash
N_NODES=2 modal run --detach modal_train.py::train_model
```

Each node has 8× B200 GPUs, so total GPU count = `N_NODES × 8`.

| N_NODES | GPUs | Notes |
|---------|------|-------|
| 1       | 8    | Single-node; set PP=1 or adjust sharding to fit in 8 GPUs |
| 2       | 16   | Minimum for the default TP=2 × EP=4 × PP=2 config |
| 4       | 32   | Default; accommodates TP=2 × EP=4 × PP=4 |
| 8       | 64   | For larger batch sizes or reduced recomputation |

The total number of model-parallel ranks must equal `TP × EP × PP × CP`, and this must
divide evenly into the total GPU count. The remaining factor becomes data parallelism (DP).

## Parallelism & Sharding Configuration

GLM-4.7 is a large MoE model that requires careful sharding across GPUs.
The script exposes four parallelism dimensions, plus sequence parallelism:

### Tensor Parallelism (TP)

```
--tp-size 2   (default)
```

Splits individual weight matrices across GPUs within a node. Higher TP reduces per-GPU memory
but increases all-reduce communication. TP should not exceed GPUs per node (8).

### Expert Parallelism (EP)

```
--ep-size 4   (default)
```

Distributes MoE experts across GPUs. GLM-4.7 has 60 experts; `EP=4` places 15 experts per
EP rank. EP must evenly divide the expert count.

### Pipeline Parallelism (PP)

```
--pp-size 4   (default)
```

Splits model layers across pipeline stages. Higher PP enables training larger models across more
nodes, at the cost of pipeline bubble overhead. Each PP stage holds a contiguous slice of layers.

### Context Parallelism (CP)

```
--cp-size 1   (default)
```

Splits the sequence dimension across GPUs. Useful for very long sequences. Leave at 1 unless
training with sequences >8k tokens.

### Sequence Parallelism (SP)

Always enabled (`--sequence_parallel true`). Complements TP by partitioning
activations along the sequence dimension in non-tensor-parallel regions (LayerNorm, dropout).
Reduces activation memory with no additional configuration.

### How the dimensions compose

Total model-parallel size = `TP × EP × PP × CP`. Data parallelism (DP) is implicit:

```
DP = (N_NODES × 8) / (TP × EP × PP × CP)
```

**Default config (4 nodes):**
- TP=2 × EP=4 × PP=4 × CP=1 = 32 model-parallel ranks
- 4 nodes × 8 GPUs = 32 GPUs → DP=1
- `global_batch_size` of 8 is handled entirely by gradient accumulation


## LoRA Configuration

Training uses LoRA (Low-Rank Adaptation) by default, targeting all linear layers:

| Parameter | Default | Flag |
|-----------|---------|------|
| Rank | 128 | `--lora-rank` |
| Alpha | 32 | `--lora-alpha` |
| Target modules | all-linear | (hardcoded) |
| Merge after training | false | `--merge-lora` |

A higher rank captures more of the weight update but uses more memory. The alpha/rank ratio
controls the effective learning rate scaling of the LoRA update.

To merge LoRA weights back into the base model after training:

```bash
modal run --detach modal_train.py::train_model --merge-lora
```

## Training Parameters

| Parameter | Default | Flag |
|-----------|---------|------|
| Epochs | 4 | `--max-epochs` |
| Max sequence length | 2048 | `--max-length` |
| Global batch size | 8 | `--global-batch-size` |
| Learning rate | 1e-4 | `--lr` |
| LR warmup | 5% of steps | (hardcoded) |
| Min LR | lr/10 | (hardcoded) |
| Attention backend | flash | (hardcoded) |
| Packing | disabled | `--disable-packing` |

## MoE-Specific Settings

| Parameter | Default | Flag |
|-----------|---------|------|
| Auxiliary loss coefficient | 1e-3 | `--moe-aux-loss-coeff` |
| Permute fusion | true | (hardcoded) |
| Grouped GEMM | true | (hardcoded) |
| Shared expert overlap | true | (hardcoded) |

The auxiliary loss coefficient controls load balancing across experts. Higher values encourage
more uniform expert utilization but may hurt task performance.

## Memory Optimization

Activation recomputation is enabled to reduce GPU memory usage:

| Parameter | Default | Flag |
|-----------|---------|------|
| Recompute granularity | full | (hardcoded) |
| Recompute method | uniform | (hardcoded) |
| Recompute num layers | 1 | `--recompute-num-layers` |
| Precision-aware optimizer | true | (hardcoded) |

`recompute_num_layers=1` means every layer recomputes activations during the backward pass.
Setting it higher (e.g., 2) recomputes fewer layers, trading memory for speed.

The training function also requests 1 TB of RAM and 2 TB of ephemeral disk per node.

## Checkpointing

Checkpoints are saved to the `example-msswift-glm-4-7-checkpoints` volume.

| Parameter | Default | Flag |
|-----------|---------|------|
| Save interval | every 50 steps | `--save-interval` |
| Save optimizer state | false | (hardcoded) |
| Save RNG state | false | (hardcoded) |
| Output format | HuggingFace | (hardcoded, `--use_hf 1`) |

Optimizer and RNG states are excluded to keep checkpoint sizes small (LoRA weights only).

## Monitoring

Training metrics are logged to [Weights & Biases](https://wandb.ai) under the project
`glm-4-7-sft`. Each run is identified by `--run-id` (auto-generated if not provided).

| Parameter | Default | Flag |
|-----------|---------|------|
| WandB project | glm-4-7-sft | (hardcoded) |
| Run name | auto-generated | `--run-id` |
| Log interval | every step | (hardcoded) |
| Eval interval | 50 steps | `--eval-interval` |
| Eval iterations | 10 | `--eval-iters` |

## Volumes

The script uses three Modal volumes (all v2):

| Volume | Mount | Purpose |
|--------|-------|---------|
| `glm-4-7-models` | `/models` | Base model weights |
| `example-msswift-glm-4-7-data` | `/data` | Training datasets |
| `example-msswift-glm-4-7-checkpoints` | `/checkpoints` | Output checkpoints |

## Networking

Multi-node training uses RDMA via `@modal.experimental.clustered(size=N_NODES, rdma=True)`
with EFA enabled (`efa_enabled: True`). The containers form a torchrun-compatible distributed
group with NCCL communicating over the RDMA fabric.

## Full Example

```bash
# Download model
modal run modal_train.py::download_model

# Prepare GSM8K
modal run modal_train.py::prepare_dataset \
  --hf-dataset openai/gsm8k \
  --data-folder gsm8k \
  --split train

# 4-node training with custom settings
modal run --detach modal_train.py::train_model \
  --data-folder gsm8k \
  --run-id my-gsm8k-run \
  --max-epochs 2 \
  --global-batch-size 16 \
  --lr 5e-5 \
  --lora-rank 64 \
  --lora-alpha 16 \
  --save-interval 100
```
