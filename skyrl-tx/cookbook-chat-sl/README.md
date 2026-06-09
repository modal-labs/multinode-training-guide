# Tinker cookbook compatibility: `chat_sl`

Status: **supported for text-only chat SFT**

Source recipe: `tinker_cookbook.recipes.chat_sl.train`

## What the cookbook example does

`chat_sl` is the higher-level supervised chat-training recipe. It wraps the
shared supervised trainer with dataset builders for public chat datasets such as
Tulu3 and NoRobots, plus periodic evaluation and checkpointing.

## Tinker API surface

The recipe delegates to `tinker_cookbook.supervised.train`, so it uses:

- `create_lora_training_client(...)` or checkpoint-based training-client restore
- `forward_backward(..., loss_fn="cross_entropy")`
- `forward(...)` through NLL/evaluation helpers
- `optim_step(...)`
- periodic and final state/sampler checkpoint saves

## SkyRL-TX support

The backend-facing training path is supported:

- Cross-entropy chat SFT is the most direct SkyRL-TX use case.
- Text-only chat renderers produce `EncodedTextChunk` inputs, which the JAX
  SkyRL-TX backend consumes.
- Periodic and final checkpoints are compatible with the checkpoint path used by
  the Modal example.

## Required adjustments

- Pass `base_url=http://localhost:8000` and `model_name` matching the launched
  SkyRL-TX base model.
- Ensure the chosen cookbook renderer supports that model family. For the Modal
  example's default, use a Qwen-compatible renderer.
- If using the Modal launcher as-is, set the launcher/server `lora_rank` high
  enough for the recipe's requested rank. Cookbook defaults often use rank 32,
  while this repo's smoke defaults use rank 8.
- Reduce dataset/batch sizes for a quick validation run.

## Unsupported or risky pieces

No text-SFT backend feature is missing. The risk is mostly scale: full public
datasets plus frequent eval/checkpoint cadence will be much heavier than the
small Modal smoke in this repo.
