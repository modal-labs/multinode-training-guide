# Tinker cookbook compatibility: `search_tool`

Status: **partial: policy training path supported, retrieval stack external**

Source recipe: `tinker_cookbook.recipes.search_tool.train`

## What the cookbook example does

`search_tool` replicates a Search-R1-style tool-use setup. The model interacts
with a Wikipedia search tool backed by Chroma embeddings, receives answer/format
rewards, and trains with the shared RL trainer.

## Tinker API surface

- rollout sampling with `SamplingClient.sample(...)`
- default `forward_backward(..., loss_fn="importance_sampling")`
- `optim_step(...)`
- optional streaming minibatches, which still submit standard
  forward/backward/optimizer requests
- checkpoints

## SkyRL-TX support

The model-training path is compatible:

- Tool calls are rendered into text transcripts before they reach SkyRL-TX.
- Text prompts, sampling, generated-token logprobs, `importance_sampling`, and
  checkpoints are supported.
- Streaming minibatches are a client-side scheduling strategy over normal API
  calls.

## Required adjustments

- The recipe CLI does not currently expose `base_url`; add one or use an
  SDK-supported environment variable to point at `http://localhost:8000`.
- Run the SkyRL-TX server with the same model/tokenizer used by the recipe.
- Provide the Chroma service, collection, and embedding setup separately. The
  SkyRL-TX example does not start a retrieval database.
- Scale down `batch_size`, `group_size`, and `max_trajectory_tokens` for smoke
  validation.

## Unsupported or risky pieces

The retrieval/indexing stack is outside SkyRL-TX. The public cookbook's intended
Wikipedia-scale index can require substantial RAM and separate service
orchestration. KL/post-KL paths should remain disabled unless prompt-logprob
support is separately validated.

## Executed SkyRL-TX smoke

Smoke code: `skyrl-tx/cookbook_smoke_client.py::CookbookSmokeRunner.search_tool`

Validated with:

```bash
modal run skyrl-tx/modal_train.py::run_cookbook --lora-rank 4 --example search_tool
```

Recorded result on 2 x `H100:8`: **PASS** for the model-side transcript path.
The smoke rendered a toy search observation into text, sampled a response, ran
one `importance_sampling` forward/backward step, and applied one optimizer step.
It did **not** start Chroma or build a Wikipedia retrieval index.

```json
{"example":"search_tool","status":"PASS","loss_sum":-795.1834402664549,"loss_values":78,"duration_seconds":22.584,"external_dependency":"Chroma not exercised"}
```
