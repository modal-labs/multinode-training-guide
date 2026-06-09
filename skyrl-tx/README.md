# SkyRL-TX Qwen SFT and RL on Modal

This example runs [SkyRL-TX](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-tx)
as a Tinker-compatible training server on a Modal multi-node GPU cluster.
It uses `Qwen/Qwen3-8B` by default and exercises both:

- supervised fine-tuning with Tinker's `cross_entropy` loss
- policy-gradient RL with Tinker's `ppo` loss over sampled arithmetic rollouts

The default topology is 2 nodes × 8 H100 GPUs. Within each node SkyRL-TX uses
tensor parallelism; across nodes it uses JAX FSDP.

## Tinker cookbook compatibility reports

These notes assess 10 public
[Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
recipes against this example's pinned SkyRL-TX JAX backend and Modal launch
shape:

| Cookbook recipe | Status | Report |
| --- | --- | --- |
| `sl_loop` | Supported | [cookbook-sl-loop](cookbook-sl-loop/) |
| `rl_loop` | Supported for default no-KL loop | [cookbook-rl-loop](cookbook-rl-loop/) |
| `chat_sl` | Supported for text-only chat SFT | [cookbook-chat-sl](cookbook-chat-sl/) |
| `math_rl` | Supported for default no-KL RL; partial for KL diagnostics | [cookbook-math-rl](cookbook-math-rl/) |
| `code_rl` | Partial: sandbox infrastructure is external | [cookbook-code-rl](cookbook-code-rl/) |
| `preference/dpo` | Partial: custom DPO smoke passed; full recipe still needs dataset adaptation | [cookbook-preference-dpo](cookbook-preference-dpo/) |
| `preference/rlhf` | Partial: reward-policy smoke passed; full pipeline needs adaptation | [cookbook-preference-rlhf](cookbook-preference-rlhf/) |
| `distillation/on_policy_distillation` | Partial: same-server teacher smoke passed; separate teacher models remain unvalidated | [cookbook-distillation-on-policy](cookbook-distillation-on-policy/) |
| `search_tool` | Partial: retrieval stack is external | [cookbook-search-tool](cookbook-search-tool/) |
| `vlm_classifier` | Partial API smoke only; real VLM support is not provided by this text-only example | [cookbook-vlm-classifier](cookbook-vlm-classifier/) |

## Prerequisites

- Modal CLI installed and authenticated
- Modal environment selected: `export MODAL_ENVIRONMENT=<your-env>`
- Modal secret `huggingface-secret` with `HF_TOKEN`
- Access to multi-node GPU clusters

Run all commands from the repo root.

## Quickstart

Download the Qwen checkpoint into the persistent Hugging Face cache volume:

```bash
modal run skyrl-tx/modal_train.py::download_model
```

Run the supervised fine-tuning smoke:

```bash
modal run --detach skyrl-tx/modal_train.py::run_sft
```

Run the RL smoke:

```bash
modal run --detach skyrl-tx/modal_train.py::run_rl
```

Run the cookbook compatibility smoke suite:

```bash
modal run --detach skyrl-tx/modal_train.py::run_cookbook --lora-rank 4
```

Use detached mode for the training jobs; the image build, model load, JAX
initialization, and first compile can take several minutes.

## Cluster sizing

The launcher reads this sizing environment variable at import time:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SKYRL_TX_N_NODES` | `2` | Number of Modal containers in the JAX cluster |

Each container always requests a full `H100:8` node. Partial H100 allocations do
not work for multi-node SkyRL-TX runs.

For the default `Qwen/Qwen3-8B` run:

```text
total GPUs = SKYRL_TX_N_NODES × 8 = 16 by default
tensor_parallel_size = 8
fully_sharded_data_parallel_size = SKYRL_TX_N_NODES = 2
```

## How it works

`run_sft` and `run_rl` create a per-run ephemeral Modal Dict for coordination,
then launch clustered `run_sft_cluster` or `run_rl_cluster` functions. Rank 0
starts the SkyRL-TX Tinker API server:

```bash
uv run --extra gpu --extra tinker --extra jax -m skyrl.tinker.api \
  --base-model Qwen/Qwen3-8B \
  --backend jax \
  --backend-config '{"tensor_parallel_size": 8, "fully_sharded_data_parallel_size": 2, ...}'
```

Ranks 1..N start SkyRL-TX JAX workers:

```bash
uv run --extra gpu --extra tinker --extra jax -m skyrl.backends.jax \
  --coordinator-address <rank-0-ip>:7777 \
  --num-processes <N> \
  --process-id <rank>
```

After the API server reports healthy, rank 0 runs `sft_client.py`,
`rl_client.py`, or `cookbook_smoke_client.py` against `http://localhost:8000`.

## Volumes

| Volume | Mount path | Purpose |
| --- | --- | --- |
| `skyrl-tx-hf-cache` | `/root/.cache/huggingface` | Qwen model cache |
| `skyrl-tx-checkpoints` | `/checkpoints` | SkyRL-TX LoRA checkpoints |

Each training job saves both a Tinker training-state checkpoint and a sampler
checkpoint. The clients run a small evaluation pass after training, then load the
saved sampler checkpoint for a sample/reward evaluation. Rank 0 lists the
checkpoint files it found and commits the Modal volume. Successful runs print
lines like:

```text
sft_state_checkpoint=file://...
sft_eval_loss=...
sft_sampler_checkpoint=file://...
sft_sampler_eval_sample=...
sft_checkpoint_file=... bytes=...
sft_checkpoint_volume_committed=...
rl_state_checkpoint=file://...
rl_eval_loss_outputs=...
rl_sampler_checkpoint=file://...
rl_eval mean_reward=... trajectories=...
rl_checkpoint_file=... bytes=...
rl_checkpoint_volume_committed=...
cookbook_result={"example": "...", "status": "PASS", ...}
cookbook_summary={"passed": 10, "unexpected_failures": 0}
cookbook_cookbook_results=.../cookbook_results.jsonl bytes=...
```

## Adjusting the smoke

Both entrypoints expose small training-loop knobs:

```bash
modal run --detach skyrl-tx/modal_train.py::run_sft --steps 16 --lora-rank 8 --learning-rate 1e-6
modal run --detach skyrl-tx/modal_train.py::run_rl --steps 8 --samples-per-prompt 4 --learning-rate 1e-6
```

The clients intentionally use tiny arithmetic datasets so the example validates
the end-to-end SkyRL-TX path without requiring a full benchmark-scale run. The
default `1e-6` learning rate keeps short LoRA smoke runs stable on the tiny
batches used here.
