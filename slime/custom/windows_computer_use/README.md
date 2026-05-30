# Windows Computer Use — VLM RL Training

Train a vision-language model (Qwen3-VL-2B) to control a Windows desktop via
RL (GRPO). The model sees screenshots and emits keyboard actions to accomplish
tasks ranging from simple Notepad saves to PowerShell commands and multi-step
file operations.

## Architecture

```
Slime (GRPO on Modal H200s)
  │
  │  VLM multi-turn rollout
  │  ↕ each turn:
  │     1. Screenshot the Windows VM
  │     2. Feed screenshot + history to Qwen3-VL via SGLang
  │     3. Model emits <action>sendkey ...</action> or <action>type ...</action>
  │     4. Execute action on VM via RPC
  │     5. Repeat until <done/> or max_turns
  │
  │  Reward: check output file with task-specific checker
  ▼
Windows VM (Modal Sandbox with QEMU/KVM)
  └── Windows Server 2022 + RPC server
```

## Prerequisites

1. **Windows disk image** on the `windows-qemu-disk` Modal Volume.
   If you don't have one, install it first using
   [windows-sandboxes](https://github.com/modal-projects/windows-sandboxes):
   ```bash
   cd /path/to/windows-sandboxes
   python main.py install
   ```

2. **Modal secrets**: `huggingface-secret` (HF_TOKEN) and `wandb-secret` (WANDB_API_KEY).

## Quick Start

```bash
# 1. Download the model
EXPERIMENT_CONFIG=qwen3vl_windows_computer_use modal run slime/modal_train.py::download_model

# 2. Prepare the dataset
EXPERIMENT_CONFIG=qwen3vl_windows_computer_use modal run slime/modal_train.py::prepare_dataset

# 3. Train (detached)
EXPERIMENT_CONFIG=qwen3vl_windows_computer_use modal run -d slime/modal_train.py::train

# Smoke test (boots VM, runs tasks, checks varying rewards)
modal run slime/test_windows_env.py
```

## Task Levels

The dataset contains 47 tasks across 4 difficulty levels:

| Level | Type | Count | Description |
|-------|------|-------|-------------|
| 1 | `notepad_simple` | 30 | Type text, save as `C:\output.txt` |
| 2 | `notepad_filename` | 8 | Type text, save to a specified filename |
| 3 | `powershell` | 6 | Run PowerShell commands to create files |
| 4 | `multistep` | 3 | Create directories + files in sequence |

### Reward Checkers

Each task specifies a checker function for computing reward:

- **`exact_match`** — 1.0 for exact match, 0.5 for case-insensitive substring, 0.2 for any content, 0.0 for empty
- **`date_format`** — 1.0 for YYYY-MM-DD on its own, 0.5 if present in longer text
- **`has_windows_dirs`** — checks for Windows/Users/Program Files in directory listing
- **`non_empty`** — 1.0 for any non-empty content
- **`has_step1_step2`** — checks for multi-step file append results

## Action Space

```xml
<action>sendkey KEY</action>     <!-- e.g., meta_l-r, ctrl-s, ret, tab -->
<action>type TEXT</action>        <!-- type text into active window -->
<action>typeline TEXT</action>    <!-- type text + Enter -->
<action>wait SECONDS</action>     <!-- wait up to 10 seconds -->
<done/>                           <!-- signal task completion -->
```

## Files

| File | Purpose |
|------|---------|
| `configs/qwen3vl_windows_computer_use.py` | SlimeConfig — model, infra, hyperparameters |
| `custom/windows_computer_use/env_windows.py` | RL environment wrapping the Windows VM |
| `custom/windows_computer_use/rollout.py` | VLM multi-turn rollout (generate function) |
| `custom/windows_computer_use/reward.py` | Custom reward model |
| `custom/windows_computer_use/dataset.py` | Multi-task dataset generation |
| `custom/windows_computer_use/sandbox_manager.py` | VM lifecycle (boot, COW disk, login) |
| `custom/windows_computer_use/vm_client.py` | HTTP client for the in-VM RPC server |

## How it works

### COW disk overlays

Each rollout boots a fresh Windows VM using a QEMU copy-on-write overlay:
```bash
qemu-img create -f qcow2 -b /vol/windows-disk.qcow2 -F qcow2 /tmp/rollout_disk.qcow2
```
This is instant and never modifies the base image.

### VLM multi-turn pattern

The rollout follows Slime's VLM multi-turn pattern (same as `geo3k_vlm_multi_turn`):
- Each turn adds a screenshot as a new image in the conversation
- SGLang handles multi-image VLM inference natively
- Only model-generated tokens are trained (`loss_mask=0` for observations)
- The environment computes reward at episode end

### Task metadata encoding

Task metadata (output path, checker type, difficulty) is JSON-encoded in the
`target` column of the parquet dataset. The `build_env` function parses this
to configure each rollout with the correct output path and reward checker.

### Scaling up

To try harder tasks or larger models:
- Change `hf_checkpoint` and `slime_model_script` in the config
- Add new task types in `dataset.py` and corresponding checkers in `env_windows.py`
- Increase `actor_num_gpus_per_node` / `tensor_model_parallel_size` for larger models
