# slime example

## Example of Async RL using slime on Modal

In separate processes, run:
```
modal run slime/modal_train.py::prepare_dataset  # once
modal run slime/modal_train.py::download_model --config qwen-8b-multi  # once
```

Then you can run:
```
modal run slime/modal_train.py::train_multi_node --config qwen-8b-multi
```

## Code-golf RL example (SLIME + Harbor + Modal sandboxes)

A dedicated example lives in:

`slime/code_golf_harbor_modal`

It adds:

- MBPP -> Harbor task conversion
- custom `--custom-rm-path` reward that runs Harbor-style verification in Modal sandboxes
- size-aware code-golf reward shaping
- checkpoint serving + Harbor eval utilities
