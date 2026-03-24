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
