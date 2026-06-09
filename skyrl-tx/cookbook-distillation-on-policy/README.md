# Tinker cookbook compatibility: `distillation/on_policy_distillation`

Status: **mostly unsupported as written; partial if converted to same-model no-KL RL**

Source recipe: `tinker_cookbook.recipes.distillation.on_policy_distillation`

## What the cookbook example does

The on-policy distillation recipe has a student sample trajectories, then uses a
teacher model's per-token logprobs to apply a reverse-KL training signal. Its
default `kl_penalty_coef` is `1.0`, so teacher KL is the actual supervision.

## Tinker API surface

- student `create_lora_training_client(...)`
- teacher `create_sampling_client(base_model=...)` or checkpoint-backed sampler
- student rollout `sample(...)`
- teacher `compute_logprobs_async(...)`
- `forward_backward(..., loss_fn="importance_sampling")`
- `optim_step(...)` and checkpoints

## SkyRL-TX support

Only the generic pieces are supported:

- Text rollout sampling, generated-token logprobs, `importance_sampling`,
  optimizer steps, and checkpointing are available.
- A same-base teacher represented by a saved sampler checkpoint can use the
  standard sampler path.

## Required adjustments

- Set `model_name` and `teacher_model` to the same model loaded by the SkyRL-TX
  server, or provide teacher weights as a sampler checkpoint for that same base.
- Pass `base_url=http://localhost:8000`.
- Reduce `lora_rank`; cookbook defaults use rank 128, much higher than this
  repo's smoke configuration.
- For a backend-only smoke, set `kl_penalty_coef=0.0`; that converts it into a
  standard no-reward/no-KL RL loop and no longer tests the recipe's core
  distillation objective.

## Unsupported or risky pieces

As written, the recipe depends on teacher prompt logprobs through
`compute_logprobs_async` and on using potentially different teacher/student base
models. The current SkyRL-TX Modal example loads one text model, and SkyRL's
Tinker limitation docs still mark prompt-logprob/KL workflows as not ready.
Therefore the actual distillation objective is not supported by this example
until a dedicated prompt-logprob teacher run is validated.
