# Agent Guide: Running Training Jobs On Modal

This document captures durable repo-specific workflow for agents launching and debugging training jobs on Modal in this repository.

## Scope

- Keep secrets out of this file.
- Record reusable workflow and runtime behavior, not one-off experiment notes.
- Prefer repo-root commands so the runbook works from a clean checkout.

## Environment Setup

- Make sure the `modal` CLI is available and authenticated in the shell you are using.
- Set `MODAL_ENVIRONMENT` explicitly, or pass `--env <env>`, when the target environment matters.
- If your local setup relies on shell init for auth or helper tooling, ensure that setup is loaded before launching jobs.
- Store credentials in local shell config or Modal secrets, not in tracked files.

## Common Entrypoints

These are the main training entrypoints from the repo root:

| Example | Entrypoint |
|---------|------------|
| `benchmark/` | `modal run benchmark/modal_train.py` |
| `lightning/` | `modal run lightning/modal_train.py::train_multi_node` |
| `megatron/` | `modal run --detach megatron/modal_train.py::train_lora` |
| `miles/` | `modal run miles/modal_train.py --recipe <recipe>` |
| `ms-swift/` | `modal run --detach ms-swift/modal_train.py::train_model` |
| `nanoGPT/` | `modal run nanoGPT/modal_train.py::speedrun_multi_node` |
| `resnet50/` | `modal run --detach resnet50/modal_train.py` |
| `slime/` | `modal run -d slime/modal_train.py::train` |
| `starcoder/` | `modal run starcoder/modal_train.py::train_multi_node --launch-type torchrun` |
| `verl/` | `modal run verl/modal_train.py::train_multi_node` |

Check the example README before launching if that directory has extra setup steps such as model download, dataset prep, or required secrets.

## Launching Jobs

- Prefer detached launches for real training and any helper job that may run for a long time:

```bash
modal run --detach path/to/modal_train.py::function
```

- Use the smallest dataset subset that still exercises the full stack when proving a new path.
- Once one topology works, keep the topology fixed and change one tuning parameter at a time.
- Treat model downloads, dataset prep, conversion, and training as separate checkpoints when the example supports that split.

## Scheduling Expectations

- Large GPU jobs may wait several minutes before worker assignment.
- During capacity waits, the client may only show spinner output or a scheduling message.
- Empty logs do not mean failure if workers have not started yet.
- A detached app may briefly gain tasks and then drop back to zero while Modal retries placement.

## Detached App Workflow

- After the function invocation exists, `Ctrl-C` is safe for detached runs and the app keeps running on Modal.
- In attached mode, `Ctrl-C` stops the app even if containers are already active.
- `--detach` does not protect the run before the function invocation exists. Interrupting during image build or object creation can still kill the launch.
- Track long-running jobs by app ID.

Useful commands:

```bash
modal app list --json
modal app logs <app-id>
modal app stop <app-id>
```

## Reading App State

- `modal app list --json` is the fastest way to confirm that an app exists and whether workers are active.
- The JSON output uses display-style keys such as `App ID`, `State`, `Tasks`, and `Stopped at`.
- `Tasks: 0` usually means workers have not landed yet or the placement is being retried.
- `Tasks: N` means containers are active and logs should start to matter.
- `State: stopped` with a recent stop time is the quickest confirmation that a detached run has exited.

## Logs, Containers, And Volumes

- `modal app logs <app-id>` only shows function and container logs after user code starts. It does not show the earlier image-build phase.
- For multi-node jobs, expect repeated startup blocks, one per node.
- When logs lag behind actual progress, inspect live containers directly:

```bash
modal container list --env <env>
modal container exec <container-id> -- nvidia-smi
```

- Use volume inspection when you need to confirm whether a job has started writing committed outputs:

```bash
modal volume ls <volume-name> / --env <env>
```

- `modal volume ls` takes the path inside the volume, not the in-container mount path.
- Live files inside a running container are not the same as committed volume state. If a producer app is still running, container inspection is often more accurate than volume listings.

## Image Build And Startup Behavior

- A lightweight helper function can still trigger image creation for heavier functions defined in the same app.
- When helper jobs feel stuck before user code runs, check whether the delay is image build time rather than training logic.
- For very recent upstream dependencies, pin exact commits or versions in the Modal image definition instead of relying on floating branches.

## Large Downloads And Shared Caches

- Model downloads can take much longer than the first visible log lines suggest.
- Progress bars may stall or update slowly even while shards are actively downloading.
- If a replacement app will reuse the same shared cache volume, stop the previous app first and confirm it has fully stopped before launching the next one.

## Debugging Strategy

1. Prove the launch path with the smallest useful dataset or shortest run.
2. Wait through scheduling before assuming failure.
3. Confirm app creation and active task count with `modal app list --json`.
4. Inspect logs only after workers appear.
5. If logs are ambiguous, check live containers and committed volumes separately.
6. After a known-good run, change one variable at a time.

## Updating This Runbook

- Add new notes when they describe durable Modal behavior or a stable repo workflow.
- Do not preserve temporary experiment history here once it has served its purpose.
- If a lesson only applies to one example, prefer that example's README over this shared runbook.
