# ms-swift: SFT GLM-4.7 (same LongMIT dataset as Megatron example)

This example shows supervised fine-tuning (SFT) of `zai-org/GLM-4.7` with
[ms-swift](https://github.com/modelscope/ms-swift) on Modal.

It intentionally reuses the same dataset source and preprocessing strategy as
the Megatron example:

- Dataset source: `donmaclean/LongMIT-128K`
- Prompt construction: passages + question -> answer
- Token filter: `<= 131072`
- Output path/schema: `/data/longmit-128k/training.jsonl` with `input`/`output`

## Prerequisites

- Modal account with H100 access
- Hugging Face token with access to GLM-4.7

Create the required secret:

```bash
modal secret create huggingface-secret HF_TOKEN=<your-token>
```

Optional (only if you want W&B logging):

```bash
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

## Quick Start

### 1) Download model weights

```bash
modal run ms-swift/modal_train.py::download_model
```

### 2) Prepare dataset (same as Megatron example)

```bash
modal run ms-swift/modal_train.py::prep_dataset
```

### 3) Launch multi-node SFT with ms-swift

```bash
modal run --detach ms-swift/modal_train.py::train_sft
```

## What `train_sft` uses by default

- 4 nodes x 8 H100 GPUs
- `swift sft` + LoRA
- DeepSpeed ZeRO-3
- BF16
- `max_length=16384` with `truncation_strategy=left`
- same preprocessed LongMIT JSONL from `/data/longmit-128k/training.jsonl`

## Customize

You can override key args directly from `modal run`, for example:

```bash
modal run --detach ms-swift/modal_train.py::train_sft \
  --run-name glm47-ms-swift-longrun \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 32768
```

