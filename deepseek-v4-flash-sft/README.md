# DeepSeek-V4-Flash 60k LoRA SFT and serving

Train a rank-64 attention LoRA for
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
at a 60,000-token sequence length with NeMo AutoModel, finalize the
pipeline-parallel checkpoint as a PEFT adapter, and serve it with vLLM.

## Requirements

1. Modal multi-node cluster access and quota for 8 nodes of 8 H200 GPUs.
2. A Modal secret named `huggingface-secret` containing `HF_TOKEN`.
3. The repository's locked `uv` environment.

Create the Hugging Face secret once:

```bash
uv run --frozen modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
```

## Topology

| Dimension | 8-node default | 16-node option |
| --- | ---: | ---: |
| Nodes / GPUs | 8 / 64 H200 | 16 / 128 H200 |
| TP / DP / PP / CP / EP | 1 / 1 / 4 / 16 / 16 | 1 / 2 / 4 / 16 / 32 |
| Sequence length | 60,000 | 60,000 |
| Global / local batch | 8 / 4 | 8 / 4 |
| Gradient accumulation | 2 | 1 |
| Attention / MoE dispatch | TileLang / UCCL-EP | TileLang / UCCL-EP |
| LoRA | rank 64, alpha 64 | rank 64, alpha 64 |
| AutoModel targets | `wq_a`, `wq_b`, `wkv` | `wq_a`, `wq_b`, `wkv` |
| Exported PEFT targets | `q_a_proj`, `q_b_proj`, `kv_proj` | `q_a_proj`, `q_b_proj`, `kv_proj` |
| Optimizer steps | 5 | 5 |
| Checkpoint step | 4 (zero-based) | 4 (zero-based) |
| Scheduler / cgroup memory | 128 MiB / 256 GiB per node | 128 MiB / 256 GiB per node |

The request enables EFA so Modal can schedule from the larger RDMA-capable
pool. The image contains UCCL extensions for both EFA and Mellanox/RoCE and
activates the extension that matches the scheduled nodes.

## Train

Run from the repository root with a unique run ID:

```bash
uv run --frozen modal run --detach \
  deepseek-v4-flash-sft/modal_train.py::train \
  --run-id dsv4-flash-60k-lora-$(date -u +%Y%m%d-%H%M%S)
```

Eight nodes is the default. Set `N_NODES=16` before the command to select the
DP=2, EP=32 layout. The launcher derives the topology from `N_NODES` and
rejects incompatible layouts before scheduling.

The training function writes a fixed-length synthetic chat dataset on each
node and renders [`train_recipe.yaml`](train_recipe.yaml) with the selected EP
size and checkpoint path. Replace `_write_synthetic_dataset` and the recipe's
`dataset` section to use a production dataset.

Each pipeline stage writes its adapter shard to the
`example-deepseek-v4-flash-sft-checkpoints` volume. The run ID is immutable:
the trainer refuses to overwrite an existing checkpoint directory.

## Finalize

After the detached training app finishes, merge the four pipeline-stage shards:

```bash
uv run --frozen modal run \
  deepseek-v4-flash-sft/modal_train.py::finalize \
  --run-id <run-id>
```

Finalization requires all 258 tensors across all 43 layers, verifies their
shapes and values, rewrites AutoModel projection names to the Transformers
names used by PEFT and vLLM, and loads the complete adapter through PEFT.

The finalized files are:

```text
/checkpoints/<run-id>/epoch_0_step_4/
|-- checkpoint_manifest.json
`-- model/
    |-- adapter_config.json
    `-- adapter_model.safetensors
```

The manifest records the adapter SHA-256 digest. Optimizer state is not saved,
so this output is a serving adapter rather than a training-resume checkpoint.

## Serve

Deploy the finalized run:

```bash
SERVE_RUN_ID=<run-id> \
  uv run --frozen modal deploy \
  deepseek-v4-flash-sft/modal_train.py
```

The `serve` endpoint is an OpenAI-compatible vLLM server protected by Modal
proxy authentication. It exposes the base model as
`deepseek-ai/DeepSeek-V4-Flash` and the adapter as
`deepseek-v4-flash-60k-lora`, with a maximum model length of 65,536 tokens.

Serving uses pipeline parallelism across four H200 GPUs. The launcher verifies
the finalized manifest and adapter digest before starting vLLM.

## Integration patches

The pinned AutoModel, UCCL, and vLLM images use six integration patches:

| Patch | Scope | Purpose |
| --- | --- | --- |
| [`patches/automodel_checkpoint_dequant.patch`](patches/automodel_checkpoint_dequant.patch) | AutoModel loading | Pass FP4 checkpoint dequantization through the recipe API. |
| [`patches/automodel_pp_peft_checkpoint.patch`](patches/automodel_pp_peft_checkpoint.patch) | AutoModel checkpointing | Save one PEFT shard per pipeline stage without optimizer state. |
| [`patches/automodel_composite_backend.patch`](patches/automodel_composite_backend.patch) | AutoModel control plane | Handle the `cpu:gloo,cuda:nccl` backend in signal handling. |
| [`patches/automodel_uccl_teardown.patch`](patches/automodel_uccl_teardown.patch) | AutoModel shutdown | Release UCCL before distributed and CUDA teardown. |
| [`patches/uccl_ipv6_oob.patch`](patches/uccl_ipv6_oob.patch) | UCCL transport | Use Modal's IPv6 inter-node interface for OOB coordination. |
| [`patches/vllm_deepseek_v4_lora.patch`](patches/vllm_deepseek_v4_lora.patch) | vLLM adapter loading | Register DeepSeek-V4 LoRA and map PEFT attention names to fused modules. |

These patches cover model loading, checkpoint export, transport, shutdown, and
adapter registration. They do not modify the model forward pass, attention
math, or autograd graph.

vLLM 0.25.1 provides DeepSeek-V4 serving and pipeline parallelism, while the
patch supplies the LoRA registration and packed Q/KV mapping for
`DeepseekV4ForCausalLM`. The serving image downloads the official model config
to an isolated directory so the FP8 quantization metadata comes directly from
the model repository.
