# Multi-node Lightning

Minimal PyTorch Lightning multi-node demo, based on https://lightning.ai/docs/overview/train-models/multi-node-training#lightning-fabric.

This demo uses Lightning's Fabric launcher to configure and launch a multi-node training job.

## Usage

**8 x A100, single node:**

```
MODAL_PROFILE=modal_labs modal run modal_train.py::train_single_node
```

**2 x 8 x H100, multi-node:**

```
MODAL_PROFILE=modal_labs modal run modal_train.py::train_multi_node
