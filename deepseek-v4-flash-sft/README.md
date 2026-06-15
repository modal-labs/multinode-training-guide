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
can load the DeepSeek-V4-Flash config/tokenizer, tokenizes a sample prompt, and confirms that
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
| LoRA rank / alpha | 64 / 64 | Increase rank for quality once the run is stable. |
| LoRA target modules | `linear_proj` | Default smoke target; broader DeepSeek MLA targets are safe when using the CP=4 long-context path. |
| Max length | 4096 | DeepSeek-V4 supports 1M context, but start small. |

For a short smoke run that should save after five training steps:

```bash
modal run --detach modal_train.py::train_model \
  --data-folder gsm8k \
  --train-iters 5 \
  --save-interval 5 \
  --run-id smoke-5steps
```

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
`modal_train.py` pins the Megatron-Core dev commit that provides the DSv4 hybrid attention module
required by `mcore-bridge`.

## Long-context (60k) SFT

For 60k-context SFT, use context parallelism instead of memory patches:

```bash
N_NODES=4 modal run --detach modal_train.py::long_context_loop \
  --target-modules linear_q_up_proj,linear_kv_proj,linear_proj
```

This runs baseline eval, prepares synthetic 60k summarization data, trains for five steps, exports the
checkpoint, and re-runs eval. The committed long-context path runs at CP=4 on 4×8 B200 nodes
(`TP*EP*PP*CP = 1*8*1*4 = 32` must divide `N_NODES*8`, so launch with `N_NODES=4`).

### Context parallelism instead of memory patches

An earlier revision fit 60k context on 2 nodes with detach-based memory patches that rewrote DSv4
attention to reduce peak memory. Those rewrites severed autograd through attention, so LoRA gradients
were only correct when targeting `linear_proj` (the attention *output* projection). Targeting any
module upstream of attention silently produced **zero gradients** on those adapters — no error raised.

Those patches were removed in favor of raising context parallelism. The allocations they guarded all
scale with the per-rank sequence length `S_rank` (the DSA indexer quadratically), so *in principle*
raising CP shrinks `S_rank` and absorbs the same memory while keeping gradients correct for **any**
`target_modules`:

| CP | `S_rank` @60k | DSA indexer peak (∝ `S_rank²`) | CSA gather peak (∝ `S_rank`) |
| ---: | ---: | ---: | ---: |
| 2 | 30k | ~43 GiB | ~19.5 GiB |
| 4 (default) | 15k | ~11 GiB | ~10 GiB |
| 8 | 7.5k | ~2.7 GiB | ~5 GiB |

With the memory patches gone, broader DeepSeek MLA LoRA targets (`linear_q_up_proj`, `linear_kv_proj`,
`linear_proj`) train with correct gradients.

> **Status — the patches-removed 60k path is not yet validated (open).** The ~81 GiB/GPU figure quoted
> in earlier revisions came from a run that still had the memory patches. With the patches removed,
> measured runs OOM on the first forward step: CP=4 on 4×8 B200 is ~17.7 GiB/GPU short, and CP=8 on
> 8×8 B200 closes most of the gap (~0.5 GiB/GPU short) but still OOMs — at CP=8 the per-rank cost is
> dominated by the EP-sharded frozen base weights, which CP does not reduce. CP=8 also exposes a RoPE
> correctness bug: the `megatron_rope_cp_shape_fix` patch only handles the single-tail CP-padding case
> (`repeats == 1`) and cyclically repeats the rotary cache at CP>4, corrupting positional encodings.
> This is a pre-existing property of the patches-removed design, independent of the `transformers`
> upgrade — an A/B at `transformers` 4.57.4 and 5.10.2 OOMs identically. See PR #84 for the full data
> and the open decision on how to make 60k fit (e.g. EP=16 on 16 nodes plus a DSv4 RoPE-CP fix, or
> restoring the gradient-safe memory patches with `linear_proj`-only LoRA). The validated path today is
> the 4k recipe (`train_model` / export).

> **Module names:** DeepSeek-V4 uses MLA, not fused QKV, so there is no `linear_qkv` module. The
> attention-input projections are `linear_q_up_proj` (RoPE query up-proj) and `linear_kv_proj` (KV
> latent); the output projection is `linear_proj`. Targeting a nonexistent module silently trains
> nothing there (peft matches 0 modules).

### Known limits

This does not change throughput: the unfused DSA/CSA reference kernels are slow regardless of memory.
The better long-term path for production long-context DSv4-Flash training is fused TileLang kernels with
real backward passes, but DSv4 attention LoRA is not wired up there today. Upstream NVIDIA Megatron-LM
has no fused DSA kernels and no CSA implementation at all as of 2026-06; the pinned Megatron-Core commit
comes from a fork.

## Optional W&B logging

If `WANDB_API_KEY` is available in the container environment, enable W&B logging with:

```bash
modal run --detach modal_train.py::train_model_wandb
```

The default `train_model` entrypoint does not require `wandb-secret`.

## Notes on the fallback Megatron path

Megatron Bridge added DeepSeek-V4 / V4-Flash support on `main`, including HF↔Megatron conversion and
FP8/MXFP4 handling. If ms-swift regresses, use the Megatron Bridge recipe as the fallback path:

- keep `TP=1` and scale with PP/EP;
- use `PP=4, EP=8` as the published Blackwell baseline for larger jobs;
- install `fast-hadamard-transform` for DSA attention;
- disable fused mHC on H100, but keep the default fused path for Blackwell.

This example starts with ms-swift because it exposes the shortest SFT command surface while still
using Megatron under the hood.
