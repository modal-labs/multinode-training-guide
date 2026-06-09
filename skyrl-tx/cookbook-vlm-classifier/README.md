# Tinker cookbook compatibility: `vlm_classifier`

Status: **not supported by this SkyRL-TX Modal example**

Source recipe: `tinker_cookbook.recipes.vlm_classifier.train`

## What the cookbook example does

`vlm_classifier` trains and evaluates a vision-language model on image
classification datasets. The datums include image chunks plus text prompts, and
the recommended model family is Qwen3-VL.

## Tinker API surface

- `ModelInput` values containing image chunks
- VLM renderers and image processors
- `forward_backward(..., loss_fn="cross_entropy")`
- sampler-based image classification evaluation
- state and sampler checkpoints

## SkyRL-TX support

This repo's Modal example is text-only:

- It launches `Qwen/Qwen3-8B`, not a Qwen3-VL checkpoint.
- The pinned JAX SkyRL-TX backend used here renders model inputs by concatenating
  token chunks; image chunks are not converted into VLM placeholder tokens and
  pixel tensors on this path.
- The example image does not install or configure the newer VLM/vLLM stack
  referenced by SkyRL's VLM documentation.

## Required adjustments

To support this recipe, the example would need a different launch path:

- load a VLM-capable base model such as Qwen3-VL
- use a backend renderer that turns image chunks into multimodal model inputs
- install the VLM-compatible vLLM/image-processing dependencies
- validate image forward/backward and sampler checkpoint evaluation on Modal

## Unsupported or risky pieces

Do not expect `vlm_classifier` to work by simply pointing it at the current
`skyrl-tx/modal_train.py` server. Text-only SFT/RL support does not imply VLM
support, even though the Tinker API type definitions include image chunks.
