# Tinker cookbook compatibility: `rl_loop`

Status: **supported for the default no-KL GRPO-style loop**

Source recipe: `tinker_cookbook.recipes.rl_loop`

## What the cookbook example does

`rl_loop` samples groups of answers for GSM8K prompts, grades each answer with a
client-side reward function, centers rewards within the group, and trains the
policy using Tinker's `importance_sampling` loss.

## Tinker API surface

- `TrainingClient.save_weights_and_get_sampling_client()`
- `SamplingClient.sample(...)` with generated-token logprobs
- `TrainingClient.forward_backward(datums, loss_fn="importance_sampling")`
- `TrainingClient.optim_step(...)`
- final state and sampler checkpoints

## SkyRL-TX support

The model-side requirements are supported:

- Generated-token sampling and generated-token logprobs are returned by the
  SkyRL-TX sampler.
- `importance_sampling` is one of the documented Tinker losses wired through the
  SkyRL data path.
- Ephemeral sampler sync via `save_weights_and_get_sampling_client()` matches the
  pattern used by cookbook RL loops.
- Checkpointing at the end of the run is supported.

## Required adjustments

- Run against the loaded SkyRL-TX model with `base_url=http://localhost:8000`.
- Keep `model_name` equal to the model used to launch the server, or at least
  use the matching tokenizer/renderer. The Modal example loads `Qwen/Qwen3-8B`.
- Scale down `batch_size`, `group_size`, and `max_tokens` for smoke validation;
  cookbook defaults are benchmark-oriented and will take much longer than this
  repo's tiny arithmetic RL client.

## Unsupported or risky pieces

The default `rl_loop` does not use a KL penalty, prompt logprobs, tools, or
images, so there is no known SkyRL-TX feature blocker. Large benchmark runs
still need normal training validation for throughput, memory, and reward quality.

## Executed SkyRL-TX smoke

Smoke code: `skyrl-tx/cookbook_smoke_client.py::CookbookSmokeRunner.rl_loop`

Validated with:

```bash
modal run skyrl-tx/modal_train.py::run_cookbook --lora-rank 4
```

Recorded result on 2 x `H100:8`: **PASS**. The smoke synced sampler weights,
sampled a rollout from the SkyRL-TX server, reused generated-token logprobs, ran
one `importance_sampling` forward/backward step, and applied one optimizer step.

```json
{"example":"rl_loop","status":"PASS","loss_sum":-265.6109986276941,"loss_values":34,"duration_seconds":49.045}
```
