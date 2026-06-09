# Tinker cookbook compatibility: `preference/rlhf`

Status: **partial: reward-policy smoke validated; full pipeline needs adaptation**

Source recipe: `tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline`

## What the cookbook example does

The RLHF pipeline chains three stages:

1. policy SFT on NoRobots-style chat data
2. preference/reward model SFT on comparison prompts
3. policy RL where a sampler-backed preference model scores generated responses

## Tinker API surface

- SFT stages use `cross_entropy`, `optim_step`, and checkpoint saves.
- The RL stage loads the SFT checkpoint as the initial policy.
- The reward/preference model is used through `create_sampling_client(model_path=...)`
  and `sample_async(...)`.
- The policy RL stage uses the shared RL trainer, defaulting to
  `importance_sampling`.

## SkyRL-TX support

The underlying operations are mostly available:

- Text SFT for both the policy and preference model is supported.
- Loading a saved state for the policy RL stage is supported.
- Sampling from saved sampler weights is supported.
- Default no-KL `importance_sampling` policy RL is supported.

## Required adjustments

- The pipeline CLI does not expose `base_url` directly, so run it only after
  adding a base-url parameter or using an SDK-supported environment variable for
  the local SkyRL-TX API.
- Use a single loaded base model for all stages. The SkyRL-TX server does not
  load different base models for policy and reward model inside one run.
- Lower `lora_rank` or launch the server with a matching `max_lora_rank`; the
  cookbook default rank 64 exceeds this repo's smoke default rank 8.
- Keep KL settings disabled in the RL stage unless separately validated.

## Unsupported or risky pieces

The full three-stage pipeline has not been validated in this Modal example.
Checkpoint handoff between stages must use paths visible to the same SkyRL-TX
server/volume. Multiple large stages may also exceed the intended scope of this
small example without additional orchestration.

## Executed SkyRL-TX smoke

Smoke code: `skyrl-tx/cookbook_smoke_client.py::CookbookSmokeRunner.preference_rlhf`

Validated with:

```bash
modal run skyrl-tx/modal_train.py::run_cookbook --lora-rank 4
```

Recorded result on 2 x `H100:8`: **PASS**. The smoke trained a tiny
preference/reward LoRA with `cross_entropy`, saved sampler weights for that
reward model, sampled from the saved reward sampler, then ran one policy
`importance_sampling` update.

```json
{"example":"preference_rlhf","status":"PASS","loss_sum":-50.875,"loss_values":30,"policy_loss_sum":-114.83354728658126,"policy_loss_values":22,"duration_seconds":61.387}
```
