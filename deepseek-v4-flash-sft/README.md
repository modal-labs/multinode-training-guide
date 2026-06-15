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
| LoRA target modules | `linear_proj` | Default smoke target; broader DeepSeek MLA targets require a gradient-safe CP path such as the validated AutoModel recipe below. |
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
`modal_train.py` pins a Megatron-Core PR-head commit from NVIDIA/Megatron-LM#5087 because released
Megatron-Core does not yet include DSv4 THD context parallelism.

## Validated 60k H200 LoRA training

`automodel_modal_train.py` is a separate NeMo AutoModel path for the public DSv4 context-parallel
implementation. This entrypoint runs five optimizer steps and writes a pipeline-sharded LoRA
checkpoint:

```bash
EFA_ENABLED=1 N_NODES=16 uv run --frozen modal run --detach \
  automodel_modal_train.py::h200_16node_60k_lora_5step \
  --run-id dsv4-flash-h200-16n-cp16-60k-lora-5step
```

| Dimension | Value |
| --- | ---: |
| GPUs | 16 nodes × 8 H200 (128 total) |
| TP / DP / PP / CP / EP | 1 / 2 / 4 / 16 / 32 |
| Sequence length | 60,000 |
| Attention / MoE dispatch | TileLang / UCCL-EP |
| LoRA | rank 64, alpha 64 |
| AutoModel LoRA targets | `wq_a`, `wq_b`, `wkv` |
| PEFT export targets | `q_a_proj`, `q_b_proj`, `kv_proj` |
| Host memory | 128 MiB scheduler request, 256 GiB cgroup limit |

The EFA-enabled request can run on either EFA or Mellanox capacity. The image builds both UCCL
variants and selects the matching extension at runtime. UCCL uses a 600-second CPU timeout because
PP stages initialize their EP groups sequentially; its old 100-second default is too short for this
topology. The scheduler request must remain separate from the cgroup limit: a 64 GiB host limit was
observed to OOM during model loading, while the 256 GiB limit completed without cgroup OOM events.

The 2026-07-16 EFA validation completed a supervised forward, backward, and optimizer update with
479,896 label tokens, finite loss (`0.0010`), and nonzero gradient norm (`0.0016`). Rank 0 reported
21.04 GiB allocated, and the highest sampled pipeline-stage allocation was 31.95 GiB per GPU. The
first step took 457 seconds and includes TileLang compilation plus sequential UCCL initialization;
it is not a steady-state throughput measurement.

The 2026-07-17 five-step checkpoint validation landed on Mellanox/RoCE capacity and completed with
finite loss (`0.0010` to `0.0008`) and gradient norm (`0.0017` to `0.0014`) at every step. Its
compile-heavy first step took about 413 seconds; the next four took approximately 28, 23, 22, and
22 seconds. This also validates runtime transport selection for both EFA and Mellanox clusters.

Only the upstream attention projections were trainable in that run. Their nonzero gradient validates
that context parallelism preserves autograd through attention, unlike the removed detach-based memory
patches. Set `--lora-target-modules` to a comma-separated wildcard list to change the adapters, or
set `--lora-rank 0` for full finetuning; a one-step full-finetune smoke also completed on the same
topology.

After the detached training app stops and all nodes commit the checkpoint volume, merge and validate
the four pipeline-stage shards:

```bash
uv run --frozen modal run \
  automodel_modal_train.py::finalize_lora_checkpoint \
  --run-id dsv4-flash-h200-16n-cp16-60k-lora-5step
```

The finalizer verifies all 258 expected LoRA tensors across 43 layers, rejects duplicate, non-finite,
wrong-shaped, or still-zero LoRA-B tensors, and maps AutoModel's internal projection names to the
official Transformers names. It writes a standard PEFT `adapter_model.safetensors` plus
`adapter_config.json`, then loads the result through PEFT against an official Transformers model
constructed on the meta device. Artifacts are stored in the
`example-deepseek-v4-flash-sft-checkpoints` volume under `<run-id>/epoch_0_step_4/model/`;
`checkpoint_manifest.json` in the parent directory records the tensor count, shard layout, PEFT load
result, and SHA-256 digest. The five-step entrypoint intentionally omits AdamW state, so this is an
adapter checkpoint for PEFT loading or export, not an exact optimizer-state resume checkpoint.

The entrypoint still creates synthetic local chat data; a customer recipe must replace that dataset.
The carried patches are narrow integration fixes for FP4 base-checkpoint dequantization, PP-stage PEFT
checkpointing, composite Gloo/NCCL shutdown handling, UCCL teardown, and IPv6 OOB transport. None
rewrites model forward functions or detaches tensors from autograd.

## Experimental Megatron/ms-swift 60k path

The intended 60k-context SFT path uses context parallelism and extra expert parallelism instead of
memory patches. To test only training, launch `train_model` directly:

```bash
N_NODES=16 modal run --detach modal_train.py::train_model \
  --run-id longctx-60k-summary \
  --data-folder meeting_summaries_60k_cp8_128 \
  --train-iters 1 \
  --save-interval 100 \
  --max-length 60000 \
  --cp-size 8 \
  --ep-size 16 \
  --target-modules linear_q_up_proj,linear_kv_proj,linear_proj \
  --packing \
  --padding-free
```

For the full demo loop, use:

```bash
N_NODES=16 modal run --detach modal_train.py::long_context_loop \
  --target-modules linear_q_up_proj,linear_kv_proj,linear_proj
```

This runs baseline eval, prepares synthetic 60k summarization data, trains, exports the checkpoint,
and re-runs eval. The long-context shape is CP=8 and EP=16 on 16x8 B200 nodes with
`max_length=60000` (`TP*EP*PP*CP = 1*16*1*8 = 128` must divide `N_NODES*8`). It also enables
packing/padding-free mode so Megatron uses the THD context-parallel path from NVIDIA/Megatron-LM#5087
instead of the older unpacked CSA path.

### Context parallelism instead of memory patches

An earlier revision fit 60k context on 2 nodes with detach-based memory patches that rewrote DSv4
attention to reduce peak memory. Those rewrites severed autograd through attention, so LoRA gradients
were only correct when targeting `linear_proj` (the attention *output* projection). Targeting any
module upstream of attention silently produced **zero gradients** on those adapters — no error raised.

Those patches were removed in favor of upstream THD context parallelism, then adding more model
partitioning when CP=8 alone was still short on the old unpacked path. The allocations they guarded all
scale with the per-rank sequence length `S_rank` (the DSA indexer quadratically), so *in principle*
raising CP shrinks `S_rank` and absorbs the same activation memory while keeping gradients correct for
**any** `target_modules`:

| CP | `S_rank` @60k | DSA indexer peak (proportional to `S_rank^2`) | CSA gather peak (proportional to `S_rank`) |
| ---: | ---: | ---: | ---: |
| 2 | 30k | ~43 GiB | ~19.5 GiB |
| 4 | 15k | ~11 GiB | ~10 GiB |
| 8 | 7.5k | ~2.7 GiB | ~5 GiB |

With the memory patches gone, broader DeepSeek MLA LoRA targets (`linear_q_up_proj`, `linear_kv_proj`,
`linear_proj`) train with correct gradients.

> **Current status.** This branch pins NVIDIA/Megatron-LM#5087, upgrades mcore-bridge to 1.5.2, and
> carries gradient-safe Bridge shims for DSv4 THD CP boundary tensors plus packed RoPE indexing. The
> Megatron/ms-swift path still OOMs in the unfused sparse-attention `kv_gathered.float()` allocation
> at 60k. FlashMLA's fused sparse-prefill path remains the missing optimization for that stack. Use
> the AutoModel H200 smoke above for the validated 60k execution path; the default 4k ms-swift recipe
> remains the production-oriented path with data preparation, export, and evaluation wired together.

> **Module names:** DeepSeek-V4 uses MLA, not fused QKV, so there is no `linear_qkv` module. The
> attention-input projections are `linear_q_up_proj` (RoPE query up-proj) and `linear_kv_proj` (KV
> latent); the output projection is `linear_proj`. Targeting a nonexistent module silently trains
> nothing there (peft matches 0 modules).

### Known limits

The long-context path is pinned to an open Megatron-Core PR, so treat it as a validation target rather
than a stable customer recipe. The image now installs the CuTeDSL stack needed by the PR's DSv4 CP
layout kernels. Keep the default 4k recipe unpacked; only the long-context loop opts into
packing/padding-free because NVIDIA's CP implementation is specifically the THD packed-sequence path.

Fused DSA can be requested with `--use-dsa-kernel-fusion`, but the current image intentionally leaves
it off by default because FlashMLA is not installed. Before handing efficient 60k training to users,
the open support should include Megatron-LM#5087 or equivalent DSv4 THD CP support, mcore-bridge or
ms-swift support for DSv4 boundary hidden/KV tensors, and a CUDA 12.9+ training image with FlashMLA's
SM100 sparse prefill kernels built and importable.

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
