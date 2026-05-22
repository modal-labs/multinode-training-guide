# Windows Computer Use — VLM RL Training

Train a vision-language model (Qwen3-VL-2B) to control a Windows desktop via
RL (GRPO). The model sees screenshots and emits keyboard actions to accomplish
tasks like opening Notepad, typing text, and saving files.

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
  │  Reward: check if C:\output.txt matches the target text
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
```

## Task: Notepad File Saving

The initial task is deliberately simple — open Notepad, type a specific sentence,
and save it as `C:\output.txt`:

1. Model receives: a screenshot + instruction ("type 'Hello World' and save as C:\output.txt")
2. Expected actions: `sendkey meta_l-r` → `type notepad` → `sendkey ret` → wait →
   `type Hello World` → `sendkey ctrl-s` → navigate save dialog → `<done/>`
3. Reward: binary check — does `C:\output.txt` contain the exact target text?

The dataset contains 50 different sentences at varying lengths.

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
| `custom/windows_computer_use/dataset.py` | Dataset generation |
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

### Scaling up

To try harder tasks or larger models:
- Change `hf_checkpoint` and `slime_model_script` in the config
- Add new environment classes for different Windows tasks
- Increase `actor_num_gpus_per_node` / `tensor_model_parallel_size` for larger models
