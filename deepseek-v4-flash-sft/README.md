# DeepSeek-V4-Flash 60k LoRA SFT and serving

Train a rank-64 attention LoRA for
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
at a 60,000-token sequence length with NeMo AutoModel, then serve the resulting
PEFT adapter with vLLM.

This is a systems-validation recipe. The checked 64- and 128-GPU
configurations completed five forward, backward, and optimizer steps, exported
complete adapters, and loaded them in vLLM 0.25.1. The 64-GPU adapter was also
generated through and processed a 60,000-token serving prompt. The recipe uses
synthetic SFT data and does not demonstrate model quality.

## Requirements

1. Modal multi-node cluster access and quota for 8 nodes of 8 H200 GPUs.
2. A Modal secret named `huggingface-secret` containing `HF_TOKEN`.
3. The repository's locked `uv` environment.

Create the Hugging Face secret once:

```bash
uv run --frozen modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
```

## Validated topology

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

EFA is enabled on the Modal request so the job can use the larger EFA-capable
pool. The image contains UCCL extensions for both EFA and Mellanox/RoCE and
selects the matching extension after scheduling.

## Train

Run from the repository root. Choose a unique run ID because the checkpoint
writer refuses to overwrite an existing run:

```bash
uv run --frozen modal run --detach \
  deepseek-v4-flash-sft/automodel_modal_train.py::h200_60k_lora_5step \
  --run-id dsv4-flash-60k-lora-$(date -u +%Y%m%d-%H%M%S)
```

Eight nodes is the default. Set `N_NODES=16` before the command to use the
validated DP=2 topology; the recipe derives EP=16 or EP=32 from the selected
node count and rejects incompatible layouts before scheduling.

The detached app must finish successfully before finalization. Each pipeline
stage writes its own adapter shard to the
`example-deepseek-v4-flash-sft-checkpoints` volume.

The synthetic data generator creates at least 40 examples and supervises the
assistant tokens. Replace `_write_synthetic_chat_jsonl` and the `dataset`
section of `_recipe_yaml` before using customer data.

## Finalize

Merge the four pipeline-stage shards and validate the adapter:

```bash
uv run --frozen modal run \
  deepseek-v4-flash-sft/automodel_modal_train.py::finalize_lora_checkpoint \
  --run-id <run-id>
```

The finalizer requires all 258 expected tensors across all 43 layers. It
rejects duplicate, missing, non-finite, wrong-shaped, and unchanged LoRA-B
tensors, rewrites AutoModel projection names to the official Transformers
names, and loads the result through PEFT against a meta-device base model.

The finalized files are:

```text
/checkpoints/<run-id>/epoch_0_step_4/
|-- checkpoint_manifest.json
`-- model/
    |-- adapter_config.json
    `-- adapter_model.safetensors
```

The manifest records the adapter SHA-256 digest and validation result. This
recipe intentionally omits AdamW state, so the output is a serving/export
adapter rather than an exact training-resume checkpoint.

## Validate serving

The validator starts vLLM on four H200 GPUs with pipeline parallelism, verifies
the adapter manifest and tensor mapping, compares the base and adapter on a
fixed 12-example GSM8K slice, then runs a semantic retrieval prompt with exactly
60,000 input tokens through both model IDs:

```bash
uv run --frozen modal run \
  deepseek-v4-flash-sft/automodel_modal_train.py::validate_lora_serving \
  --run-id <run-id>
```

A result includes per-example base and adapter outputs, both model IDs from
`/v1/models`, and exact token-usage records for the two 60k retrieval requests.

The server uses vLLM's native DeepSeek-V4 FP8/FP4 kernels with PP=4. It does not
use tensor parallelism: vLLM's native FP4 MoE TP path is not yet the conservative
choice for this checkpoint layout.

## Deploy

Select the finalized run when deploying:

```bash
SERVE_RUN_ID=<run-id> \
  uv run --frozen modal deploy \
  deepseek-v4-flash-sft/automodel_modal_train.py
```

The `serve_lora` endpoint is an OpenAI-compatible vLLM server protected by
Modal proxy authentication. It exposes the adapter as
`deepseek-v4-flash-60k-lora`.

## Integration patches

The recipe carries six pinned integration patches:

| Patch | Scope | Why it remains |
| --- | --- | --- |
| `automodel_checkpoint_dequant.patch` | AutoModel loading | Passes FP4 checkpoint dequantization through the pinned recipe API. |
| `automodel_pp_peft_checkpoint.patch` | AutoModel checkpointing | Saves one PEFT shard per PP stage and permits omitting optimizer state. |
| `automodel_composite_backend.patch` | AutoModel control plane | Handles the `cpu:gloo,cuda:nccl` composite backend in signal handling. |
| `automodel_uccl_teardown.patch` | AutoModel shutdown | Frees UCCL before distributed/CUDA teardown. |
| `uccl_ipv6_oob.patch` | UCCL transport | Adds IPv6 OOB support for Modal's inter-node interface. |
| `vllm_deepseek_v4_lora.patch` | vLLM adapter loading | Advertises DSv4 LoRA support and maps PEFT attention names to vLLM's fused modules. |

None of these patches changes the model forward pass, attention math, or
autograd graph. The former Megatron/ms-swift experiment and its DSv4 boundary
rewrite are deliberately excluded because that path still OOMed at 60k and was
not part of the successful run.

vLLM 0.25.1 contains native DeepSeek-V4 serving and pipeline parallel support,
but its `DeepseekV4ForCausalLM` class does not yet expose the LoRA interface or
the packed Q/KV adapter mapping. The vLLM patch is limited to that registration
and name mapping. `_prepare_vllm_hf_config` also downloads a clean official
`config.json` outside the shared training cache so stale cache metadata cannot
silently select an unquantized loader.

## Validation

The checked 8-node DP=1 run completed all five steps with finite loss and
nonzero gradient norm. Its final step reported loss 0.0008, gradient norm
0.0014, and 25.35 GiB of trainer GPU memory. The end-to-end serving check used
that finalized adapter with vLLM 0.25.1 on four H200 GPUs. It registered 129
logical LoRA modules (three targets in each of 43 layers). On the fixed GSM8K
sanity slice, base and adapter each scored 11/12 and their numeric predictions
agreed on all 12 examples. Both model IDs then retrieved the expected code from
the beginning of an exact 60,000-token prompt.

## Limits

- The training evidence is a five-step synthetic-data smoke test, not a
  convergence or quality result.
- The 12-example GSM8K comparison is a regression sanity check, not a model
  quality benchmark.
- The checkpoint does not contain optimizer state.
- The first training step includes TileLang compilation and UCCL setup, so it
  is not representative of steady-state throughput.
- Serving was validated for correctness at 60k, not benchmarked for throughput
  or latency.
- The pinned upstream commits and local patches should be replaced as their
  corresponding fixes ship upstream.
