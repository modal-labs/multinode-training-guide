# Tinker cookbook compatibility: `sl_loop`

Status: **supported with normal SkyRL-TX launch parameters**

Source recipe: `tinker_cookbook.recipes.sl_loop`

## What the cookbook example does

`sl_loop` is the minimal supervised fine-tuning example. It builds chat datums
from a Hugging Face dataset, trains on assistant-token cross entropy, periodically
saves checkpoints, and can resume from a saved training state.

## Tinker API surface

- `ServiceClient.create_lora_training_client(...)`
- `ServiceClient.create_training_client_from_state_with_optimizer(...)` on resume
- `TrainingClient.forward_backward(batch, loss_fn="cross_entropy")`
- `TrainingClient.optim_step(...)`
- checkpoint helpers that save both training state and sampler weights

## SkyRL-TX support

This maps directly to the SkyRL-TX path used by the Modal example:

- `cross_entropy` is the best-supported Tinker loss.
- Text-only `ModelInput` values are supported by the JAX renderer.
- LoRA training, optimizer steps, state checkpoints, sampler checkpoints, and
  sampling-client creation from saved weights are supported.
- The Modal example's `sft_client.py` is a small hand-written version of this
  pattern and was validated on 2 x `H100:8`.

## Required adjustments

- Launch the SkyRL-TX server with the same `base_model` that the recipe passes as
  `model_name`; the server only has one loaded base model.
- Pass `base_url=http://localhost:8000` and `TINKER_API_KEY=tml-dummy` when
  running against the local SkyRL-TX API.
- Keep `lora_rank` at or below the server's configured `max_lora_rank`.
- Use a conservative LoRA learning rate for short smoke runs. The Modal example
  uses `1e-6` for stability on tiny batches.

## Unsupported or risky pieces

No backend feature blocks this recipe. The remaining risks are operational:
dataset download time, checkpoint volume capacity, and the need to use a model
name/renderer pair that matches the loaded SkyRL-TX model.
