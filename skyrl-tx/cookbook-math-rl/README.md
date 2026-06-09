# Tinker cookbook compatibility: `math_rl`

Status: **supported for default no-KL RL; partial for optional KL diagnostics**

Source recipe: `tinker_cookbook.recipes.math_rl.train`

## What the cookbook example does

`math_rl` is a full math-reasoning RL recipe. It builds math prompt environments,
samples multiple completions per prompt, grades answers, computes group-centered
advantages, and trains with the shared cookbook RL trainer.

## Tinker API surface

- `save_weights_and_get_sampling_client()` for rollout sampling
- `SamplingClient.sample(...)` with generated-token logprobs
- `forward_backward(..., loss_fn=<configured loss>)`
- `optim_step(...)`
- checkpoint saves and optional evaluation
- optional KL/post-KL code paths that call `compute_logprobs`

## SkyRL-TX support

The default model-side loop is compatible when configured like the simple
`rl_loop`:

- Default `loss_fn="importance_sampling"` is supported.
- The Modal example also validated the JAX `ppo` path on short rollouts, but the
  cookbook's documented safe default remains `importance_sampling`.
- Text prompts, rollout sampling, generated-token logprobs, optimizer steps, and
  checkpoints are supported.

## Required adjustments

- Keep `kl_penalty_coef=0.0` and `compute_post_kl=False` unless you are doing a
  separate prompt-logprob validation run.
- Match `model_name`, renderer, and LoRA rank to the launched SkyRL-TX server.
- Use smaller `groups_per_batch`, `group_size`, and `max_tokens` for smoke
  testing; cookbook defaults are intended for longer experiments.

## Unsupported or risky pieces

SkyRL's Tinker limitation docs mark prompt logprobs and KL penalty support as not
ready. The JAX code path contains prompt-logprob plumbing, but this Modal example
has not validated cookbook KL/post-KL. Treat `kl_penalty_coef > 0`,
`compute_post_kl=True`, and non-default loss experiments such as CISPO as
unvalidated.

## Executed SkyRL-TX smoke

Smoke code: `skyrl-tx/cookbook_smoke_client.py::CookbookSmokeRunner.math_rl`

Validated with:

```bash
modal run skyrl-tx/modal_train.py::run_cookbook --lora-rank 4
```

Recorded result on 2 x `H100:8`: **PASS**. The smoke sampled a short arithmetic
rollout, assigned a scalar reward/advantage, ran one `importance_sampling`
forward/backward step, and applied one optimizer step.

```json
{"example":"math_rl","status":"PASS","loss_sum":-590.7610627205562,"loss_values":60,"duration_seconds":12.805}
```
