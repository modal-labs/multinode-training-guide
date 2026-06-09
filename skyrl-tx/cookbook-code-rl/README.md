# Tinker cookbook compatibility: `code_rl`

Status: **partial: model training path supported, sandbox infrastructure external**

Source recipe: `tinker_cookbook.recipes.code_rl.train`

## What the cookbook example does

`code_rl` trains on programming tasks. The policy generates code, a sandbox runs
tests, rewards come from test pass/fail and formatting signals, and the shared RL
trainer updates the model from the resulting trajectories.

## Tinker API surface

- `save_weights_and_get_sampling_client()` for rollout sampling
- `SamplingClient.sample(...)`
- `forward_backward(..., loss_fn="importance_sampling")` by default
- `optim_step(...)`
- periodic/final checkpoints
- optional async/off-policy training mode

## SkyRL-TX support

The Tinker backend requirements are mostly supported:

- Text-only code prompts are ordinary `EncodedTextChunk` inputs.
- Default `importance_sampling` training is supported.
- Generated-token logprobs, optimizer steps, and checkpoints are supported.
- Async/off-policy scheduling is client-side; it still submits standard sample
  and forward/backward requests to the API.

## Required adjustments

- Pass `base_url=http://localhost:8000`, `model_name` matching the loaded server,
  and a LoRA rank allowed by the server.
- Provide a working sandbox backend. The cookbook supports SandboxFusion/Docker
  or Modal sandboxing; SkyRL-TX does not provide that execution environment.
- For Modal runs, do not rely on Modal NFS for checkpoint or sandbox state. Use
  normal local files plus Modal Volumes, as the SkyRL-TX example does.
- Keep `kl_penalty_coef=0.0` and `compute_post_kl=False` for the supported path.

## Unsupported or risky pieces

The code-execution sandbox is the main non-SkyRL dependency. Full-scale code RL
also has high variance and long rollout latency, so it needs a dedicated Modal
validation run before calling it production-ready. KL/post-KL settings remain
unvalidated for this SkyRL-TX example.

## Executed SkyRL-TX smoke

Smoke code: `skyrl-tx/cookbook_smoke_client.py::CookbookSmokeRunner.code_rl`

Validated with:

```bash
modal run skyrl-tx/modal_train.py::run_cookbook --lora-rank 4
```

Recorded result on 2 x `H100:8`: **PASS** for the model-side path. The smoke
sampled a code-shaped prompt, supplied a client-side reward, ran one
`importance_sampling` forward/backward step, and applied one optimizer step. It
did **not** run SandboxFusion/Docker/Modal sandbox tests.

```json
{"example":"code_rl","status":"PASS","loss_sum":-497.1251588719726,"loss_values":38,"duration_seconds":11.802,"external_dependency":"sandbox not exercised"}
```
