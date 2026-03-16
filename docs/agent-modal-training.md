# Agent Guide: Running And Debugging Training Jobs On Modal

This document captures repo-specific workflow for agents launching and debugging distributed training jobs on Modal.

## Scope

- Keep secrets out of this file.
- Record durable operational behavior, not one-off credentials.
- Update this file as new Modal runtime patterns are learned.

## Shell Setup

- Do not assume repo-specific shell helper commands exist.
- If `modal` is not available in a non-interactive shell, activate the appropriate local environment first.
- Set `MODAL_ENVIRONMENT` explicitly, or pass `--env`, when the target environment matters.
- If the workspace has multiple Modal environments, activating the CLI alone may still launch into the wrong default environment.
- Prefer interactive shells when environment bootstrapping only exists in shell init.

## Launching Jobs

- Prefer detached launches for real training:

```bash
modal run --detach path/to/file.py::function ...
```

- Use the smallest dataset subset that still exercises the full stack when proving a new path.
- Once one configuration works, reuse the exact topology for follow-on runs before changing tuning parameters.

## Scheduling Expectations

- Large GPU jobs, especially clustered B200 jobs, may wait 5-10 minutes for worker assignment.
- While waiting, the launcher may only show spinner output.
- A message like `Function '...' is waiting to be scheduled...` is normal during capacity waits.
- Empty logs do not mean the app is broken if workers have not started yet.

## Detached App Workflow

- After the function invocation exists, `Ctrl-C` is safe for detached runs; Modal keeps the app running and prints the app ID.
- In an attached `modal run` without `--detach`, `Ctrl-C` still terminates the app even if workers are already active.
- `--detach` does not protect the run during image build or object creation before the function has actually been invoked.
- Interrupting the local client during that earlier phase is still fatal.
- For long image-build or object-creation phases, `--detach` is safer even for non-training helper jobs.
- For long helper jobs that continue well after user code starts, such as very large model downloads, detached mode is still preferable so a local session interruption does not kill the producer mid-transfer.
- Track jobs by app ID.
- Useful commands:

```bash
modal app list --json
modal app logs <app-id>
modal app stop <app-id>
```

## Reading App State

- `modal app list --json` is the fastest way to see whether an app exists and whether workers are active.
- Its JSON output uses display-style keys such as `App ID`, `State`, `Tasks`, and `Stopped at`; do not assume snake_case field names when scripting against it.
- If the tabular `modal app list` output looks stale during rapid state changes, prefer `modal app list --json`; in this repo it reflected active task counts sooner.
- `Tasks: 0` on a detached app usually means capacity has not landed yet.
- A detached clustered app can briefly show `Tasks > 0` and then drop back to `Tasks: 0` before logs start; in this repo that meant Modal recycled the placement and retried later, not that user code had completed.
- `Tasks: N` indicates workers are active; at that point logs should begin to matter.
- `State: stopped` with a recent `Stopped at` timestamp is the quickest confirmation that a detached run has exited.

## Reading Logs

- For multi-node runs, expect one startup section per node.
- Common useful milestones:
  - `Running megatron command: ...`
  - dataset tokenization/stat summaries
  - model load progress like `Loading: ...`
  - `The training of Epoch 0 starts...`
  - checkpoint save progress like `Saving: ...`
  - `completed successfully with return code 0`

- When debugging, search logs for:
  - `Traceback`
  - `RuntimeError`
  - `ValueError`
  - `OOM`
  - `out of memory`
  - `completed successfully`
- `modal app logs` only shows function/container logs after user code starts. It does not expose the earlier image-build phase.
- If app logs are empty but the app is still active, check `modal app list --json` and any expected output volume instead of assuming failure.
- Hugging Face `snapshot_download` progress bars can stall or report very little even while large shards are actively downloading.
- Modal client heartbeat warnings during long-running downloads were noisy but not fatal in this repo when the fetched-file count kept increasing.
- If logs lag behind what the GPUs are doing, `modal container exec <container-id> -- nvidia-smi` is the quickest way to distinguish a real hang from a slow first step or delayed log flush.
- During long GLM-5 runs, checkpoint files appeared inside the live container before the success lines showed up in `modal app logs`; if a run looks stalled, inspect the checkpoint directory in-container before declaring it wedged.

## Post-Train Eval Workflow

- For the ms-swift GLM DPO examples here, dataset prep now writes both `train.jsonl` and a deterministic held-out `eval.jsonl`.
- The evaluation chain is intentionally split into four stages so drift can be localized:
  - multinode Megatron-native scoring against the saved checkpoint
  - multinode Megatron-to-HF export
  - single-node HF/ms-swift-native scoring and greedy generation
  - single-node SGLang scoring and greedy generation
- Use the `evaluate_all` entrypoint in the relevant launcher when you want the full chain plus a parity report in one command.
- If only one stage is failing, rerun that specific stage rather than retraining.
- In the current GLM-4.7 DPO runs here, `checkpoint-25` and `checkpoint-30` contained a large `adapter_model.safetensors` plus `iter_00000xx/{.metadata,common.pt,metadata.json}`, but no full Megatron tensor-shard files. If Megatron-native reload fails, inspect the checkpoint tree first before assuming the sharded checkpoint exists.

## SGLang Logprob Checks

- On this repo's GLM-4.7 merged-export path, `--disable-cuda-graph` was not enough to keep SGLang startup short. It still spent minutes in piecewise graph capture until `--disable-piecewise-cuda-graph` was added explicitly.
- The original `/generate` + `logprob_start_len` prompt-logprob path was not trustworthy for GLM-4.7 here and should not be treated as a validated parity check.
- The teacher-forced `/generate` + `token_ids_logprob` path is also currently failing parity on the merged GLM-4.7 export: Megatron and HF stay near `-1.1` mean token logprob on the held-out example, while SGLang returns about `-11.93` for every response token and greedily emits token id `0` (`"!"`) as the next token. Treat that as a real framework/export bug until proven otherwise.
- This kind of scoring check is still more useful than only comparing generated text, because it catches conversion or serving drift even when a model can still start and answer requests.

## Volume Checks

- `modal volume ls <volume-name> / --env <env>` is useful when an app is still silent and you need to know whether helper jobs have started writing outputs.
- `modal volume ls` takes the path inside the volume itself, not the in-container mount path. For example, query `/distilabel-math-dpo-16`, not `/data/distilabel-math-dpo-16`.
- An empty volume with a still-running app usually means the run has not reached user code yet.
- For large model downloads, `modal container list --env <env>` plus `modal container exec <container-id> -- ...` is often the best way to inspect live cache growth inside the running container.
- Remember that live files inside a running container are not the same as committed volume state; other apps should wait for the producing app to finish and commit.
- During active Hugging Face downloads, `modal volume ls` can lag far behind the real in-container progress. Prefer app logs or container inspection for live status, and use the volume view mainly to confirm committed state.

## Image Build Behavior

- In this repo's Modal app layout, invoking a lightweight helper function can still trigger image creation for other functions defined in the same app.
- That matters for big training images: a helper like dataset prep or model download may still spend time building the training image before any user code runs.
- When validating very recent upstream support, pin the exact git commit in the Modal image definition instead of relying on a floating branch name.

## Large Hugging Face Downloads

- GLM-5 uses many multi-GB shards; expect long download phases even before training starts.
- Hugging Face Hub downloads are now Xet-backed by default; `HF_HUB_ENABLE_HF_TRANSFER` is deprecated and should not be the first knob to reach for.
- `HF_XET_HIGH_PERFORMANCE=1` materially improved `zai-org/GLM-5` download throughput in this repo.
- If a replacement app will reuse the same shared cache volume, stop the previous app first and confirm `State: stopped` before launching the new one, otherwise cache locks can serialize or stall progress.

## GLM-5 Topology Notes

- `zai-org/GLM-5` has `num_hidden_layers=78` in `config.json`.
- A plain `PP=4` split fails because 78 decoder layers do not divide evenly by 4.
- ms-swift supports uneven decoder pipeline splits via `--decoder_first_pipeline_num_layers` and `--decoder_last_pipeline_num_layers`.
- In this repo, a balanced 4-stage split of `20 / 19 / 19 / 20` is produced by setting `decoder_first_pipeline_num_layers=20` and `decoder_last_pipeline_num_layers=20`.
- A 2-node `TP=2, EP=4, PP=2` GLM-5 DPO probe got past argument parsing and model construction checks, but then OOMed during model initialization on a B200 at roughly 176 GiB allocated.
- In this runtime, PyTorch warns that `PYTORCH_CUDA_ALLOC_CONF` is deprecated and `PYTORCH_ALLOC_CONF` is the preferred env var.
- A 4-node `TP=2, EP=4, PP=4` probe with the uneven `20 / 19 / 19 / 20` split reached W&B init and `The training of Epoch 0 starts...`, which means the cluster topology, model load, and trainer startup path are valid.
- That same probe then failed on the first DPO reference-model forward in `megatron.core.transformer.experimental_attention_variant.dsa.DSAttention.forward` with `ValueError: not enough values to unpack (expected 4, got 3)`.
- The failing run used Megatron-SWIFT's default `padding_free=True`; the next GLM-5 workaround to test is forcing `--padding_free false`.
- With `padding_free=false`, the next GLM-5 probe got through model load and into `Epoch 0`, but then crashed in `compute_dsa_indexer_loss` because `dsa_indexer_loss_coeff` was still `None`.
- For the current GLM-5 Modal launcher in this repo, pass `--dsa_indexer_loss_coeff 0.0` to disable the broken DSA indexer-loss path and avoid the `Tensor * NoneType` failure.
- After fixing that, the first train step completed, but checkpoint export crashed in `swift.megatron.model.gpt_bridge.GPTBridge._set_mla_attn_state` because `mg_attn.core_attention` was `None` during HF/safetensors weight export.
- In the current GLM-5 launcher here, pass `--save_safetensors false` so the run keeps Megatron-Core checkpoints and skips the broken HF export path.
- With `padding_free=false`, `dsa_indexer_loss_coeff=0.0`, and `save_safetensors=false`, the 4-node GLM-5 DPO probe completed `1/1` training step, saved `checkpoint-1/iter_0000001`, and exited with return code `0`.

## Dataset Preprocessing Behavior

- ms-swift may warn that some dataset rows are invalid and filter them before training.
- In the GLM-4.7 DPO runs here, the 256-sample preference subset was filtered from 256 rows to 253 rows at runtime.
- Warnings of the form `There are errors in the dataset, the data will be deleted` were noisy but not fatal as long as a non-empty dataset remained.
- Distinguish between:
  - filtering warnings that still leave a train/val dataset
  - hard failures that stop the app or leave an empty dataset

## W&B Notes

- W&B may fall back to `/tmp` for local metadata if the intended path is not writable.
- That warning did not block successful training in this repo.
- Project and run URLs in logs are useful proof that the trainer reached runtime initialization.

## Known Working GLM-4.7 DPO Path

- File: `ms-swift/modal_train_dpo.py`
- Validated topology:
  - `N_NODES=2`
  - `TP=2`
  - `EP=4`
  - `PP=2`
  - `CP=1`

- Validated proof run:
  - dataset folder: `distilabel-math-dpo-16`
  - run id: `glm47-dpo-mn16-probe`
  - `--max-epochs 1`
  - `--beta 0.05`
  - `--lr 5e-6`
  - `--eval-iters 0`

- Result:
  - reached training loop
  - saved checkpoint
  - exited with return code `0`

## Known Working GLM-5 DPO Path

- File: `ms-swift/modal_train_dpo_glm5.py`
- Validated topology:
  - `N_NODES=4`
  - `TP=2`
  - `EP=4`
  - `PP=4`
  - `CP=1`
  - uneven pipeline split auto-selected as `20 / 19 / 19 / 20`

- Validated proof run:
  - dataset folder: `distilabel-math-dpo-16`
  - run id: `glm5-dpo-mn16-probe-hadamard8`
  - `--max-epochs 1`
  - `--beta 0.1`
  - `--lr 5e-6`
  - `--eval-iters 0`
  - `--no-padding-free`
  - `--dsa-indexer-loss-coeff 0.0`
  - `--no-save-safetensors`

- Result:
  - reached training loop
  - completed `1/1` train step in about `3m48s`
  - saved Megatron checkpoint `checkpoint-1/iter_0000001`
  - exited with return code `0`

## Debugging Strategy

1. Prove the distributed launch with a tiny dataset subset.
2. Wait through scheduling before assuming failure.
3. Confirm app creation and worker assignment with `modal app list --json`.
4. Inspect logs only after workers appear.
5. Do not change topology and hyperparameters simultaneously after a successful proof run.
6. Once one run is known-good, launch the next sweep runs from the same topology.
