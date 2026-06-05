# DeepSeek-V4-Flash SFT with ms-swift

Fine-tune [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
with LoRA SFT using [ms-swift](https://swift.readthedocs.io/en/latest/) Megatron on Modal.

DeepSeek-V4-Flash is a 284B-parameter MoE model with 13B activated parameters and a one-million-token
context window. This example keeps the default SFT run small enough to bring up the stack first
(`max_length=4096`, one epoch, LoRA) while using the same model and distributed training path as a
larger run.

## Prerequisites

1. A Modal account with access to B200 GPUs and the multi-node cluster preview.
2. A Modal secret named `huggingface-secret` with an `HF_TOKEN` value.

Create the Hugging Face secret with:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
```

Weights & Biases is optional. The default training command logs to stdout only.

## Quick start

Run commands from this directory:

```bash
cd deepseek-v4-flash-sft
```

### 1. Verify the image and ms-swift entrypoint

```bash
modal run modal_train.py::smoke_test
```

This does not download the model weights. It builds the ms-swift image, verifies that Transformers
can load the DeepSeek-V4-Flash config/tokenizer, checks the chat template, and confirms that
`megatron sft --help` is available.

### 2. Download the model

```bash
modal run --detach modal_train.py::download_model
```

This downloads `deepseek-ai/DeepSeek-V4-Flash` into the shared `huggingface-cache` Modal volume.
The model is large; keep this detached.

### 3. Prepare SFT data

```bash
modal run modal_train.py::prepare_dataset \
  --hf-dataset openai/gsm8k \
  --data-folder gsm8k \
  --split train \
  --max-examples 4096
```

The helper writes ms-swift chat JSONL to the
`example-deepseek-v4-flash-sft-data` volume:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

For a custom dataset, pass `--input-col` and `--output-col`.

### 4. Train

```bash
modal run --detach modal_train.py::train_model
```

By default this launches one 8×B200 node with:

| Dimension | Default | Notes |
| --- | ---: | --- |
| TP | 1 | DeepSeek-V4 Flash currently scales via EP/PP rather than TP. |
| EP | 8 | One expert-parallel group across the node. |
| PP | 1 | Increase this when scaling beyond the bring-up shape. |
| CP | 1 | Increase only for longer-context runs. |
| LoRA rank | 64 | Raise for quality once the run is stable. |
| Max length | 4096 | DeepSeek-V4 supports 1M context, but start small. |

## Scaling

`N_NODES` is evaluated when Modal defines the clustered function:

```bash
N_NODES=2 modal run --detach modal_train.py::train_model \
  --pp-size 2 \
  --ep-size 8 \
  --global-batch-size 16
```

The model-parallel product `TP × EP × PP × CP` must divide `N_NODES × 8`. Keep `TP=1` for
DeepSeek-V4-Flash until MLA tensor parallelism is supported for the DSv4 hybrid attention path.

## Optional W&B logging

If `WANDB_API_KEY` is available in the container environment, enable W&B logging with:

```bash
modal run --detach modal_train.py::train_model --report-to wandb
```

The default is `--report-to none`.

## Notes on the fallback Megatron path

Megatron Bridge added DeepSeek-V4 / V4-Flash support on `main`, including HF↔Megatron conversion and
FP8/MXFP4 handling. If ms-swift regresses, use the Megatron Bridge recipe as the fallback path:

- keep `TP=1` and scale with PP/EP;
- use `PP=4, EP=8` as the published Blackwell baseline for larger jobs;
- install `fast-hadamard-transform` for DSA attention;
- disable fused mHC on H100, but keep the default fused path for Blackwell.

This example starts with ms-swift because it exposes the shortest SFT command surface while still
using Megatron under the hood.
