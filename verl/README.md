# VERL Multi-node Training Guide

Example of full-finetuning of Qwen3-32B via RLVR on GSM.

## Quickstart

```
modal run modal_train.py::prep_dataset
modal run modal_train.py::download_model
modal run modal_train.py::convert_hf_to_mcore
modal run modal_train.py::train_multi_node
modal run modal_train.py::train_multi_node -- trainer.total_epochs=20
```
