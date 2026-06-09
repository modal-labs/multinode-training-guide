# Tinker cookbook compatibility: `preference/dpo`

Status: **partial: custom-loss smoke validated; full recipe still needs dataset adaptation**

Source recipe: `tinker_cookbook.recipes.preference.dpo.train`

## What the cookbook example does

The DPO recipe trains from chosen/rejected preference pairs. It computes
reference-model logprobs, runs a custom client-side DPO loss, then sends an
equivalent cross-entropy backward pass with synthetic weights.

## Tinker API surface

- `create_lora_training_client(...)` or checkpoint-based restore
- `save_weights_and_get_sampling_client()` for the frozen reference policy
- `SamplingClient.compute_logprobs_async(...)` for reference logprobs
- `TrainingClient.forward_backward_custom(...)`
- `TrainingClient.forward(...)` inside the custom-loss implementation
- `TrainingClient.forward_backward(..., loss_fn="cross_entropy")` for the
  surrogate backward pass
- `optim_step(...)` and checkpoints

## SkyRL-TX support

Most primitives exist in the pinned SkyRL-TX server:

- `forward(...)`, `forward_backward(..., "cross_entropy")`, optimizer steps, and
  checkpoints are supported.
- `forward_backward_custom(...)` is implemented by the Tinker SDK client using
  those standard API calls; it does not require a special server-side DPO loss.
- The JAX sampler has prompt-logprob plumbing needed by `compute_logprobs`.

## Required adjustments

- Run with `base_url=http://localhost:8000`.
- Use a `model_name` and renderer matching the loaded SkyRL-TX model.
- Start with small batches and a low LoRA learning rate.
- Keep the reference model as the initial policy/sampler checkpoint from the
  same loaded base model.

## Unsupported or risky pieces

The tiny custom-loss smoke passed, but full DPO should still be treated as
partial support until a real preference dataset run validates renderer choices,
batch sizing, and reference-logprob alignment at recipe scale. SkyRL's current
Tinker limitation docs still flag prompt logprobs as not ready, which is exactly
the API surface DPO depends on for reference logprobs.

## Executed SkyRL-TX smoke

Smoke code: `skyrl-tx/cookbook_smoke_client.py::CookbookSmokeRunner.preference_dpo`

Validated with:

```bash
modal run skyrl-tx/modal_train.py::run_cookbook --lora-rank 4
```

Recorded result on 2 x `H100:8`: **PASS**. The smoke created a frozen reference
sampler from the current policy, computed reference logprobs for one
chosen/rejected pair, ran `forward_backward_custom(...)` with a DPO-style
client-side Torch loss, and applied one optimizer step.

```json
{"example":"preference_dpo","status":"PASS","loss_sum":-69.27671873848885,"loss_values":44,"duration_seconds":52.007}
```
