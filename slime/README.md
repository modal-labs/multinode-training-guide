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
