# miles example

## Example of async RL with Miles + Harbor on Modal

This example mirrors the `slime/` launcher shape, but uses upstream Miles for async training and Harbor-backed tasks for rollout verification.

Recommended bootstrap order:

```bash
export MODAL_ENVIRONMENT=peyton-agents

modal run miles/modal_train.py::prepare_dataset --config hello-qwen-0-6b
modal run miles/modal_train.py::download_model --config hello-qwen-0-6b
modal run miles/modal_train.py::train_single_node --config hello-qwen-0-6b
```

After the Harbor proof loop works, move to the USACO config:

```bash
modal run miles/modal_train.py::prepare_dataset --config usaco-qwen-0-6b
modal run miles/modal_train.py::download_model --config usaco-qwen-0-6b
modal run miles/modal_train.py::train_multi_node --config usaco-qwen-0-6b
```

Scale up with:

```bash
modal run miles/modal_train.py::prepare_dataset --config usaco-qwen-1-7b
modal run miles/modal_train.py::download_model --config usaco-qwen-1-7b
modal run miles/modal_train.py::train_multi_node --config usaco-qwen-1-7b
```

Notes:

- `train_single_node` is the fastest way to validate the Harbor integration.
- `train_multi_node` is fixed to a 2-node clustered run with `rdma=True`.
- The multi-node configs are intentionally non-colocated and sync weights every rollout step.

## Recipe-Based Miles Jobs

For non-Harbor Miles bringup, use the recipe-driven launcher modeled on the PR #65 job-submission pattern:

```bash
export MODAL_ENVIRONMENT=peyton-agents

modal run miles/modal_recipe_train.py::prepare_dataset
modal run miles/modal_recipe_train.py::download_model --recipe glm4-7-flash-lora
MILES_N_NODES=4 modal run miles/modal_recipe_train.py --recipe glm4-7-flash-lora --dry-run
```

That path keeps the GLM-4.7-Flash model and training settings in [`miles/recipes/glm4-7-flash-lora.args`](/home/ec2-user/projects/multinode-training-guide/miles/recipes/glm4-7-flash-lora.args) and uses Ray job submission for faster iteration.
