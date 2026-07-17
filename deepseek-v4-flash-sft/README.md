# DeepSeek-V4-Flash 60k LoRA SFT and serving

Train a rank-64 attention LoRA for
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
at a 60,000-token sequence length with NeMo AutoModel, then serve the resulting
PEFT adapter with vLLM.

This is a systems-validation recipe. The checked configuration completed five
forward, backward, and optimizer steps on 128 H200 GPUs, exported a complete
adapter, loaded that exact adapter in vLLM 0.25.1, generated through it, and
processed a 60,000-token prompt. It uses synthetic SFT data and does not
demonstrate model quality.

## Requirements

1. Modal multi-node cluster access and quota for 16 nodes of 8 H200 GPUs.
2. A Modal secret named `huggingface-secret` containing `HF_TOKEN`.
3. The repository's locked `uv` environment.

Create the Hugging Face secret once:

```bash
uv run --frozen modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
```

## Validated topology

| Dimension | Value |
| --- | ---: |
| Nodes / GPUs | 16 / 128 H200 |
| TP / DP / PP / CP / EP | 1 / 2 / 4 / 16 / 32 |
| Sequence length | 60,000 |
| Global / local batch | 8 / 4 |
| Attention / MoE dispatch | TileLang / UCCL-EP |
| LoRA | rank 64, alpha 64 |
| AutoModel targets | `wq_a`, `wq_b`, `wkv` |
| Exported PEFT targets | `q_a_proj`, `q_b_proj`, `kv_proj` |
| Optimizer steps | 5 |
| Checkpoint step | 4 (zero-based) |
| Scheduler / cgroup memory | 128 MiB / 256 GiB per node |

EFA is enabled on the Modal request so the job can use the larger EFA-capable
pool. The image contains UCCL extensions for both EFA and Mellanox/RoCE and
selects the matching extension after scheduling.

## Train

Run from the repository root. Choose a unique run ID because the checkpoint
writer refuses to overwrite an existing run:

```bash
EFA_ENABLED=1 N_NODES=16 uv run --frozen modal run --detach \
  deepseek-v4-flash-sft/automodel_modal_train.py::h200_16node_60k_lora_5step \
  --run-id dsv4-flash-60k-lora-$(date -u +%Y%m%d-%H%M%S)
```

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
the adapter manifest and tensor mapping, makes a chat request through the LoRA
model ID, then sends a 60,000-token prompt and requires one generated token:

```bash
EFA_ENABLED=1 uv run --frozen modal run \
  deepseek-v4-flash-sft/automodel_modal_train.py::validate_lora_serving \
  --run-id <run-id>
```

A passing result includes both the base and adapter IDs from `/v1/models`,
`prompt_tokens: 60000`, and `completion_tokens: 1`.

The server uses vLLM's native DeepSeek-V4 FP8/FP4 kernels with PP=4. It does not
use tensor parallelism: vLLM's native FP4 MoE TP path is not yet the conservative
choice for this checkpoint layout.

## Deploy

Select the finalized run when deploying:

```bash
SERVE_RUN_ID=<run-id> SERVE_CHECKPOINT_STEP=4 EFA_ENABLED=1 \
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

The checked five-step run had finite loss and nonzero gradient norm at every
step. The end-to-end serving check used the finalized adapter with vLLM 0.25.1
on four H200 GPUs. It registered 129 logical LoRA modules (three targets in each
of 43 layers), returned a chat completion from the adapter model ID, and
reported 60,000 prompt tokens plus one completion token.

## Limits

- The training evidence is a five-step synthetic-data smoke test, not a
  convergence or quality result.
- The checkpoint does not contain optimizer state.
- The first training step includes TileLang compilation and UCCL setup, so it
  is not representative of steady-state throughput.
- Serving was validated for correctness at 60k, not benchmarked for throughput
  or latency.
- The pinned upstream commits and local patches should be replaced as their
  corresponding fixes ship upstream.
