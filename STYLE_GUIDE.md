# Style Guide

Follow these standards so that each example can be run consistently.

## Secrets

Secrets should be consistently named and used.

- `huggingface-secret` should be used for the huggingface token. This secret should always have the `HF_TOKEN` environment variable set.
- `wandb-secret` should be used for Weights & Biases. This secret should always be optional i.e. the script should work without it.

## File Structure

Most of the examples should be structured as follows:

```
multinode-training-guide/
    foo/
        modal_train.py
        train.py
        README.md
        ...
```

`modal_train.py` should be the entrypoint for training. In most cases, this will use `torchrun` to launch copies of `train.py`.

## Dependencies

Dependencies in images should always be pinned, and `python_version` should be set.
