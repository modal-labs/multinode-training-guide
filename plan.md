# Miles v2 Modal Launcher

## Goal Description

Build a Modal-based launcher for the Miles training framework (`miles/`) as a sibling directory in the multinode-training-guide repository. The launcher must follow the exact same contract and code style as the existing `slime/` launcher. Miles is a fork of Slime with additional capabilities (LoRA, FP8/INT4, FSDP, MilesRouter). The initial scope covers the Megatron backend with LoRA support, verified by running a Qwen3-4B LoRA smoke test on Modal that completes at least 2 training steps.

All code must be freshly written following slime patterns. The previous `miles/` implementation in this repo has been deleted and must not be referenced.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: The `miles/` directory follows the same launcher contract as `slime/`, providing five Modal functions: `list_configs`, `download_model`, `prepare_dataset`, `convert_checkpoint`, and `train`
  - Positive Tests (expected to PASS):
    - `miles/modal_train.py` defines an `app` with all five functions decorated appropriately
    - `EXPERIMENT_CONFIG=qwen3_4b_lora_smoke modal run miles/modal_train.py::list_configs` prints available experiments with node/GPU topology
    - Each function accepts an `experiment` parameter loaded from `EXPERIMENT_CONFIG`
  - Negative Tests (expected to FAIL):
    - Running `list_configs` with a nonexistent experiment config raises `ValueError` with available config names
    - Running `train` without setting `EXPERIMENT_CONFIG` results in a graceful no-op or error (no crash)

- AC-2: `MilesConfig` produces correct CLI args via attribute reflection, and `prepare_miles_config()` resolves HuggingFace repo-id paths and materializes inline YAML configs to temp files
  - Positive Tests (expected to PASS):
    - A `MilesConfig` subclass with `lora_rank = 64` produces `--lora-rank 64` in `cli_args()` output
    - Boolean `True` attributes produce flags without values (e.g., `colocate = True` → `--colocate`)
    - Boolean `False` or `None` attributes are omitted from CLI args
    - List attributes expand correctly (e.g., `eval_prompt_data = ["aime24", "/data/aime.jsonl"]` → `--eval-prompt-data aime24 /data/aime.jsonl`)
    - Fields in `_MILES_SKIP` (`environment`, `async_mode`, `miles_model_script`) are excluded from CLI args
    - `prepare_miles_config()` replaces HF repo-id strings (e.g., `"Qwen/Qwen3-4B"`) with local snapshot paths
    - `prepare_miles_config()` converts inline dict values for YAML config fields to temp YAML file paths
  - Negative Tests (expected to FAIL):
    - Private attributes (starting with `_`) do not appear in CLI args
    - Callable attributes (methods) do not appear in CLI args
    - The `environment` dict is never serialized as a CLI flag

- AC-3: `ModalConfig` supports `local_miles` overlay (mounted at `/root/miles`), `patch_files` injection, and `image_run_commands` for image customization
  - Positive Tests (expected to PASS):
    - Setting `local_miles="/path/to/miles"` on `ModalConfig` causes the image to mount that directory at `/root/miles`
    - Setting `patch_files=["patches/fix.patch"]` copies the patch to `/tmp/fix.patch` in the image
    - Setting `image_run_commands=["cd /root && git apply /tmp/fix.patch"]` runs those commands during image build
    - All three features can be combined in a single config
  - Negative Tests (expected to FAIL):
    - Omitting `local_miles` (or setting to `None`) does not add any local directory overlay
    - An empty `patch_files` list does not inject any files

- AC-4: A smoke-test config `qwen3_4b_lora_smoke.py` exists that faithfully corresponds to the reference script `run-qwen3-4b-megatron-lora-result.sh` with reduced iteration count
  - AC-4.1: The config includes correct Qwen3-4B model architecture sourced via `miles_model_script = "scripts/models/qwen3-4B.sh"`
    - Positive: CLI args include `--swiglu`, `--num-layers 36`, `--hidden-size 2560`, `--ffn-hidden-size 9728`, `--num-attention-heads 32`, `--group-query-attention`, `--num-query-groups 8`
    - Negative: Model architecture args are not hardcoded as class attributes (they come from the shell script)
  - AC-4.2: LoRA configuration matches the reference: `lora_rank=64`, `lora_alpha=32`, `lora_dropout=0.0`, `target_modules="all-linear"`, `megatron_to_hf_mode="bridge"`
    - Positive: All LoRA flags appear in generated CLI args
    - Negative: Missing `target_modules` causes Miles validation failure
  - AC-4.3: Topology: 1 node, 4x H200, colocated, with `use_miles_router=True` and `calculate_per_token_loss=True`
    - Positive: `total_nodes()` returns 1; GPU string resolves to `"H200:4"`
    - Negative: Setting `colocate=False` without `rollout_num_gpus` raises an error or changes topology
  - AC-4.4: `prepare_data()` downloads the `dapo-math-17k` dataset to the data volume
    - Positive: After `prepare_dataset` runs, `/data/dapo-math-17k/dapo-math-17k.jsonl` exists
    - Negative: Running `train` without prior `prepare_dataset` fails due to missing data file
  - AC-4.5: Reduced for smoke testing: `num_rollout=2`, WandB disabled, evaluation disabled
    - Positive: Config does not include `use_wandb` or eval-related flags
    - Negative: Setting `num_rollout=2` with standard batch sizing produces at least 2 training steps
  - AC-4.6: Per-config environment includes `NCCL_ALGO=Ring` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
    - Positive: These env vars appear in the Ray job runtime environment
    - Negative: `NCCL_ALGO=Ring` is NOT in the base `MilesConfig.environment` defaults (it is per-config)

- AC-5: Running the smoke-test config on Modal produces at least 2 training steps and exits cleanly
  - Positive Tests (expected to PASS):
    - `EXPERIMENT_CONFIG=qwen3_4b_lora_smoke modal run miles/modal_train.py::download_model` completes without error
    - `EXPERIMENT_CONFIG=qwen3_4b_lora_smoke modal run miles/modal_train.py::prepare_dataset` populates the data volume
    - `EXPERIMENT_CONFIG=qwen3_4b_lora_smoke modal run -d miles/modal_train.py::train` logs at least 2 training step completions and exits with code 0
    - Training output shows rollout generation and loss computation for each step
  - Negative Tests (expected to FAIL):
    - Running with an incorrect Docker image tag fails at container startup
    - Running without `download_model` first fails due to missing model checkpoint

- AC-6: All launcher code is freshly written following slime patterns; no code from the deleted `miles/` directory is referenced or copied
  - Positive Tests (expected to PASS):
    - `git log` shows new files created on the implementation branch
    - Code style matches `slime/` conventions (same naming, same patterns, same structure)
  - Negative Tests (expected to FAIL):
    - `git diff` against any deleted miles/ commit shows no copied content
    - No imports or references to any old miles/ module paths

- AC-7: Raw checkpoint conversion is structurally present but explicitly unverified in v1; only bridge mode is tested
  - Positive Tests (expected to PASS):
    - `convert_checkpoint` function exists in `modal_train.py`
    - `miles/modal_helpers/convert_hf_to_torch_dist.py` wrapper exists with `SKIP_RELEASE_RENAME` support
    - Running `convert_checkpoint` with bridge mode config prints "bridge mode — no conversion needed" and returns
  - Negative Tests (expected to FAIL):
    - Raw conversion is not part of the smoke-test verification workflow

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes a complete `miles/` launcher directory with all five Modal functions fully implemented, a comprehensive smoke-test config for Qwen3-4B LoRA, a README documenting usage, and structural support for raw conversion and async training (code present, not verified). The launcher handles LoRA configs, model script sourcing, dev overlays, patch injection, and multi-node clustering.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes `miles/` with `modal_train.py`, `configs/base.py`, `configs/__init__.py`, `modal_helpers/utils.py`, and the smoke-test config. The `train` function works for the Qwen3-4B LoRA bridge-mode colocated case. `convert_checkpoint` is present but may only handle bridge-mode bypass. README is minimal.

### Allowed Choices

- Can use: The Docker image `radixark/miles:dev-202604101227` (pinned tag)
- Can use: Same Modal volume names as slime for HF cache (`huggingface-cache`); new `miles-data` and `miles-checkpoints` for framework-specific data
- Can use: Same reflection pattern as `SlimeConfig.cli_args()` for `MilesConfig`
- Can use: Same Ray orchestration and `modal.experimental.clustered` pattern as slime
- Cannot use: Any code, patterns, or references from the deleted `miles/` directory in this repo
- Cannot use: FSDP backend for v1 verification (Megatron only)
- Cannot use: Shared/extracted launcher code between slime/ and miles/ (keep independent)

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The implementation follows a "near-copy then adapt" strategy:

1. **Base classes** (`configs/base.py`): Create `ModalConfig` (identical to slime's) and `MilesConfig` (analogous to `SlimeConfig`). Key differences:
   - `_MILES_SKIP = {"environment", "async_mode", "miles_model_script"}`
   - `YAML_CONFIG_FIELDS` remains the same (Miles uses the same YAML config pattern)
   - `miles_model_script` points to `/root/miles/scripts/models/*.sh` instead of `/root/slime/...`
   - Base `environment` includes `PYTHONPATH=/root/Megatron-LM`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and `NCCL_NVLS_ENABLE=1`

2. **Modal app** (`modal_train.py`): Mirror slime's structure with these adaptations:
   - `MILES_ROOT = "/root/miles"` instead of `SLIME_ROOT = "/root/slime"`
   - Docker image: `radixark/miles:dev-202604101227`
   - Volume names: `miles-data`, `miles-checkpoints` (reuse `huggingface-cache`)
   - Config objects loaded as `exp_mod.modal` and `exp_mod.miles`
   - `local_miles` field instead of `local_slime`

3. **Helpers** (`modal_helpers/utils.py`): Same functions adapted for Miles:
   - `prepare_miles_config()` — same HF resolution and YAML materialization
   - `build_train_cmd()` — sources `miles_model_script` from `/root/miles/`
   - `get_modal_cluster_context()` and `start_ray_head()` — identical to slime

4. **Smoke-test config** (`configs/qwen3_4b_lora_smoke.py`):
   - Sources model architecture via `miles_model_script = "scripts/models/qwen3-4B.sh"`
   - LoRA: rank 64, alpha 32, target_modules all-linear, bridge mode
   - Topology: 1 node, 4x H200, colocated
   - Reduced: `num_rollout=2`, no WandB, no eval
   - Per-config env: `NCCL_ALGO=Ring`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`

5. **Verification workflow**: `download_model` → `prepare_dataset` → `train`

### Relevant References

- `slime/configs/base.py` — Base classes to mirror (ModalConfig, SlimeConfig, volume paths, _SLIME_SKIP, YAML_CONFIG_FIELDS)
- `slime/modal_train.py` — Modal app structure to follow (image setup, volumes, five functions)
- `slime/modal_helpers/utils.py` — Helper functions to adapt (cluster context, command building, config prep)
- `slime/modal_helpers/convert_hf_to_torch_dist.py` — Conversion wrapper with SKIP_RELEASE_RENAME
- `slime/configs/__init__.py` — Dynamic config loader to copy
- `slime/configs/qwen_4b_gsm8k.py` — Example of a complete self-contained config
- `/home/ec2-user/nan_wonderland/miles/examples/lora/run-qwen3-4b-megatron-lora-result.sh` — Reference script for smoke-test config (LoRA args, topology, env vars)
- `/home/ec2-user/nan_wonderland/miles/scripts/models/qwen3-4B.sh` — Model architecture args
- `/home/ec2-user/nan_wonderland/miles/miles/utils/arguments.py` — Miles CLI argument definitions (2151 lines)
- `/home/ec2-user/nan_wonderland/miles/train.py` — Miles training entry point

## Dependencies and Sequence

### Milestones

1. **Launcher Scaffold**: Create the `miles/` directory structure with base classes and Modal app skeleton
   - Create `miles/configs/base.py` with `ModalConfig` and `MilesConfig`
   - Create `miles/configs/__init__.py` with dynamic loader
   - Create `miles/modal_helpers/__init__.py` and `miles/modal_helpers/utils.py`
   - Create `miles/modal_helpers/convert_hf_to_torch_dist.py`
   - Create `miles/modal_train.py` with all five functions

2. **Smoke-Test Config**: Build the Qwen3-4B LoRA verification config
   - Create `miles/configs/qwen3_4b_lora_smoke.py` with full config matching reference script
   - Implement `prepare_data()` for dapo-math-17k dataset download
   - Verify `list_configs` prints correct topology

3. **Modal Verification**: Run the full smoke-test workflow on Modal
   - Run `download_model` to populate HF cache volume
   - Run `prepare_dataset` to populate data volume
   - Run `train` and verify at least 2 training steps complete

4. **Documentation**: Write README.md for the miles launcher
   - Document usage, available commands, config authoring guide

Milestone 1 is a prerequisite for Milestone 2. Milestone 2 is a prerequisite for Milestone 3. Milestone 4 can proceed in parallel with Milestone 3.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|--------------------------|------------|
| task1 | Analyze Miles CLI args and identify all differences from Slime that affect the base config class design | AC-2 | analyze | - |
| task2 | Create `miles/configs/base.py` with `ModalConfig` and `MilesConfig` base classes, volume paths, and skip fields | AC-1, AC-2, AC-3 | coding | task1 |
| task3 | Create `miles/configs/__init__.py` with dynamic config loader | AC-1 | coding | - |
| task4 | Create `miles/modal_helpers/__init__.py` and `miles/modal_helpers/utils.py` with cluster coordination, command building, and config preparation | AC-1, AC-2 | coding | task2 |
| task5 | Create `miles/modal_helpers/convert_hf_to_torch_dist.py` wrapper with SKIP_RELEASE_RENAME support | AC-7 | coding | - |
| task6 | Create `miles/modal_train.py` with all five Modal functions (list_configs, download_model, prepare_dataset, convert_checkpoint, train) | AC-1, AC-3 | coding | task2, task3, task4, task5 |
| task7 | Analyze the reference script `run-qwen3-4b-megatron-lora-result.sh` and map all args to MilesConfig attributes for the smoke-test config | AC-4 | analyze | task2 |
| task8 | Create `miles/configs/qwen3_4b_lora_smoke.py` with complete Qwen3-4B LoRA smoke-test config including prepare_data() | AC-4 | coding | task6, task7 |
| task9 | Run `list_configs` to verify the config loads correctly and prints expected topology | AC-1, AC-4.3 | coding | task8 |
| task10 | Run `download_model` on Modal to populate HF cache with Qwen3-4B | AC-5 | coding | task9 |
| task11 | Run `prepare_dataset` on Modal to populate data volume with dapo-math-17k | AC-5, AC-4.4 | coding | task10 |
| task12 | Run `train` on Modal and verify at least 2 training steps complete | AC-5 | coding | task11 |
| task13 | Debug and fix any issues discovered during Modal verification | AC-5 | coding | task12 |
| task14 | Write `miles/README.md` with usage documentation | AC-1 | coding | task8 |
| task15 | Final review: verify no deleted miles/ code is referenced and code style matches slime | AC-6 | analyze | task13 |

## Claude-Codex Deliberation

### Agreements

- Building `miles/` as a sibling launcher mirroring `slime/` contract is the correct approach for v1
- Separate `modal` and `miles` config objects per experiment module, following the slime pattern exactly
- Megatron-only scope for v1 is reasonable; FSDP is deferred
- Docker image should be pinned to `radixark/miles:dev-202604101227`, not `:latest`
- `NCCL_ALGO` and `CUBLAS_WORKSPACE_CONFIG` are per-config environment variables, not base defaults
- Raw checkpoint conversion code should be present but deferred from v1 verification
- Smoke-test config should faithfully correspond to the reference script with reduced iteration count
- The verification workflow is `download_model` → `prepare_dataset` → `train`
- `MilesRouter` and `calculate_per_token_loss` should be included in the smoke-test config as they are in the reference script
- No shared code extraction between slime/ and miles/ for now

### Resolved Disagreements

- **NCCL_NVLS_ENABLE in base defaults**: Codex initially argued Miles detects NVLink at runtime so it should not be a base default. User confirmed NVLink is always present (`HAS_NVLINK=1`). Resolution: include `NCCL_NVLS_ENABLE=1` in base `MilesConfig.environment` defaults. Rationale: universal in this deployment environment.
- **"Mirror slime exactly" vs "same launcher contract"**: Claude initially used "mirror exactly". Codex argued this was too loose/strict simultaneously. Resolution: reworded to "same launcher contract" — same five functions, same config pattern, small justified divergences allowed. Rationale: the goal is functional equivalence, not character-for-character copy.
- **AC-5 verification method**: Claude proposed inferring step count from config. Codex required observed verification from runtime output. Resolution: AC-5 now requires actual Modal run log showing at least 2 training steps completing. Rationale: config-level inference is not proof of behavior.
- **AC-4 completeness**: Codex required explicit pinning of all non-default flags from the reference script. Resolution: AC-4 now includes `use_miles_router`, `calculate_per_token_loss`, `target_modules`, `lora_alpha`, and exact topology. Rationale: loose AC allows configs that won't actually run.
- **Data plan**: Codex flagged missing data preparation strategy. Resolution: added explicit `prepare_data()` requirement and `prepare_dataset` step in verification workflow. Rationale: Miles LoRA example depends on dataset paths.

### Convergence Status

- Final Status: `converged`
- Rounds: 3
- All REQUIRED_CHANGES addressed; no high-impact DISAGREE remains

## Pending User Decisions

- DEC-1: GPU type for Modal smoke test
  - Claude Position: H200 (most commonly used in slime configs)
  - Codex Position: H200 plausible, needs explicit confirmation
  - Tradeoff Summary: H200 provides ample memory for Qwen3-4B LoRA; H100 would also work but H200 is the repo standard
  - Decision Status: `H200 (4x H200 colocated)` — User confirmed

- DEC-2: Shared launcher code extraction after v1
  - Claude Position: Not now, keep independent
  - Codex Position: Either approach works; shared code only if a third launcher is added
  - Tradeoff Summary: Extracting now adds complexity without clear benefit; deferring keeps v1 simple
  - Decision Status: `Not now — keep miles/ and slime/ independent` — User confirmed

- DEC-3: "Two training steps" metric interpretation
  - Claude Position: Directional minimum bar
  - Codex Position: N/A — open question
  - Tradeoff Summary: Hard requirement would need exact step counting; directional allows any successful training as proof
  - Decision Status: `Directional minimum bar — any successful training steps prove the launcher works` — User confirmed

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Docker Image
- Use `radixark/miles:dev-202604101227` as the base Docker image (pinned tag, not `:latest`)

### Branch
- Create implementation branch based on `nan/slime-refactor`

### Key Differences from Slime Launcher
- Framework root: `/root/miles` (not `/root/slime`)
- Config object name: `miles` (not `slime`) per experiment module
- Docker image: `radixark/miles:dev-202604101227` (not `slimerl/slime:nightly-dev-20260329a`)
- Volume names: `miles-data`, `miles-checkpoints` (reuse `huggingface-cache`)
- Dev overlay field: `local_miles` (not `local_slime`)
- Model script field: `miles_model_script` (not `slime_model_script`)
- Base environment: includes `NCCL_NVLS_ENABLE=1` (NVLink always present)
- `_MILES_SKIP` includes `miles_model_script` instead of `slime_model_script`
- `build_train_cmd` references `/root/miles/train.py` or `/root/miles/train_async.py`

### External Dependencies
- Miles source reference: `/home/ec2-user/nan_wonderland/miles`
- Slime pattern reference: `/home/ec2-user/multinode-training-guide/slime`
- Modal documentation: https://modal.com/llms.txt

--- Original Design Draft Start ---


right now i want to to write miles_v2. you need to completely ignore the previous implementation of miles in this repo(i deleted it, you can see in git, do not read anything about it). You need to check the styles in slime instead, as that is the specific style I want. 

For reference, you should go to the source /home/ec2-user/nan_wonderland/miles for more information about miles and for context, miles is a fork of slime, so the styles are very similar.

You should definitely check the entire /home/ec2-user/multinode-training-guide/slime and /home/ec2-user/nan_wonderland/slime repo—specifically to see the pattern of what we inject and how to include miles.

the docker we will use for miles is `radixark/miles:dev-202604101227` and it is same as `radixark/miles:latest` if you run `docker images`.

Specifically, you need to check:
1. What our abstraction is for the separate miles, especially when there are local miles where our abstraction is going to be used.
2. What object you need to add to the support miles, particularly where local miles require that object.

since we are designing for miles launcher for modal, you should get yourself familiar with modal, https://modal.com/llms.txt is a good starting point.

To verify if you finish it or not, you should create a corresponding config of `/home/ec2-user/nan_wonderland/miles/examples/lora/run-qwen3-4b-megatron-lora-result.sh` and run it with modal and check if it can finish two training steps. YOU DO NOT NEED ANYTHING FROM `/home/ec2-user/multinode-training-guide/miles`, miles docker should have everything you need to run lora stuff for qwen3-4b. so you should completely ditch and ignore everything in `/home/ec2-user/multinode-training-guide/miles`

you should create branch based on `nan/slime-refactor` here right now in multinode-training-guide
--- Original Design Draft End ---
