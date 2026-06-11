# 60k-context LoRA SFT without the gradient-unsafe memory patches

## Outcome (committed)

This example trains DeepSeek-V4-Flash at 60k context on **CP=4 (4×8 B200)** with
**no detach-based memory patches**. The four gradient-unsafe memory patches, the
padding-mask memory patch, and the RoPE-chunking sub-patch were removed from
`deepseek_patches.py`. Only the two gradient-safe Megatron correctness patches
(`mcore_bridge_rope_config`, `megatron_rope_cp_shape_fix`) remain on the training
side, alongside the model-registration and vLLM patches.

The payoff: LoRA gradients are correct for **any** `target_modules` (not just
`linear_proj`), because nothing severs autograd through attention anymore. The
memory the dropped patches used to save is absorbed by context parallelism
instead — CP=4 peaks at ~81 GiB/GPU at 60k, less than half of a B200.

The rest of this document records why those patches were droppable and how that
was validated.

## Background

The removed training-side patches rewrote attention internals with a `detach()`
+ in-place-write pattern to reduce peak memory at 60k tokens:

- `mcore_bridge_attention_memory`
- `mcore_bridge_long_context_padding_mask`
- `megatron_dsa_chunked_indexer` (the `no_grad` top-k is only valid when the
  indexer loss coefficient is 0)
- `megatron_csa_chunked_unfused_attention`
- the RoPE-chunking portion of the original `megatron_rope_cp_and_chunking`
  (split out as `megatron_rope_seq_chunking`)

The `detach()`-based rewrites sever the autograd graph through attention. That
was survivable **only** while the recipe targeted LoRA at `linear_proj` (the
attention output projection), whose adapter gradients do not require backprop
through the attention math. Targeting any module upstream of attention
(`linear_q_up_proj`, `linear_kv_proj`, `all-linear`, …) yielded **silently zero
gradients** on those adapters — no error raised.

The patches kept on the training side are gradient-safe at any `target_modules`:

| Kept patch | Why |
|---|---|
| `transformers_deepseek_v4_config` | `deepseek_v4` not registered in transformers 4.57 |
| `transformers_deepseek_v4_meta_model` | needed for Megatron→HF export |
| `mcore_bridge_rope_config` | YaRN rope fields missing from mcore-bridge 1.4.2; model init fails without it |
| `megatron_rope_cp_shape_fix` | CP>1 is structurally broken without the CP-padding + cos/sin shape fix |
| All 7 vLLM patches | inference-only; no training gradients involved |

## Strategy: replace code patches with context parallelism

The dropped patches all addressed allocations that scale with per-rank sequence
length S_rank — the dominant ones quadratically (the DSA indexer is
`S_rank × heads × S_rank` FP32). Instead of patching the kernels, shrink S_rank
by raising CP:

| CP | S_rank @60k | DSA indexer peak (≈43 GiB @30k, ∝ S_rank²) | CSA gather peak (≈19.5 GiB @30k, ∝ S_rank) |
|---:|---:|---:|---:|
| 2 | 30k | ~43 GiB | ~19.5 GiB |
| 4 (default) | 15k | ~11 GiB | ~10 GiB |
| 8 | 7.5k | ~2.7 GiB | ~5 GiB |

On B200 (180 GB), CP=4 comfortably absorbs every allocation the dropped patches
were avoiding, so it is the committed default. CP=8 is not required.

## Running 60k SFT

The `long_context_loop` entrypoint defaults to CP=4 (the general `train_model` /
`export` path stays CP=1 so it runs on a single node). A 60k run needs a 4-node
cluster (`N_NODES=4`, 32×B200) because `TP*EP*PP*CP = 1*8*1*4 = 32` must divide
`N_NODES*8`. Broad LoRA targets are now safe:

```bash
# full loop (eval → train → export → eval), CP=4 by default
N_NODES=4 modal run -d modal_train.py::long_context_loop

# or just a training run; train_model defaults to CP=1, so pass --cp-size 4
N_NODES=4 modal run -d modal_train.py::train_model \
  --max-length 61440 --cp-size 4 \
  --target-modules linear_q_up_proj,linear_kv_proj,linear_proj
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already set in the image.

> **Module names:** DeepSeek-V4 uses MLA, not fused QKV, so there is no
> `linear_qkv` module. The attention-input projections are `linear_q_up_proj`
> (RoPE query up-proj) and `linear_kv_proj` (KV latent); the output projection is
> `linear_proj`. Targeting a nonexistent module silently trains nothing there
> (peft matches 0 modules).

## Validation summary

The removal was validated end-to-end at 60k before committing (full per-run
numbers — grad_norms, peak memory, eval transcripts — are in the PR description):

- **Gradient correctness (4k A/B).** With the memory patches enabled, the
  attention-input LoRA adapters' `lora_B` stays *exactly* zero (they receive no
  gradient) while `linear_proj` trains normally; total grad_norm collapses from
  ~1.1 to ~0.15. With the patches removed, every attention-input adapter trains
  (nonzero `lora_B`). Verified with the `inspect_lora_adapter` Modal function,
  which reports per-module `lora_B` norms from the saved DCP checkpoint
  (`lora_B` is zero-initialized, so nonzero ⇒ gradient flowed).
- **60k memory (CP=4, 32×B200).** Trains 5 steps with no memory patches and
  broad MLA targets, peaking at ~81 GiB/GPU — no OOM at any of the allocation
  sites the patches used to guard (DSA indexer, CSA gather, attention concat,
  padding-mask reduction).
- **End-to-end parity.** A checkpoint trained under the new regime exports to
  merged HF, serves through the unchanged vLLM path, and scores 100%
  action-item recall on the 60k summarization eval.

## Known limits / alternatives

- This does not change throughput: the unfused DSA/CSA reference kernels are slow
  regardless of memory. For production-grade long-context DSv4-Flash training,
  the better long-term path is fused TileLang kernels with real backward passes,
  but DSv4 attention LoRA is not wired up there today (custom module names
  `wq_a/wq_b/wkv/wo_a/wo_b` are not in its LoRA target mapping, and `wo_a` is
  consumed via direct `.weight` access).
- Upstream NVIDIA Megatron-LM has no fused DSA kernels and no CSA implementation
  at all as of 2026-06; the pinned commit comes from a fork.
