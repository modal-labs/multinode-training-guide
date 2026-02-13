# slime example

## Example of Async RL using slime on Modal

```
modal profile activate modal-labs
modal deploy slime/tests/modal_train.py # once
modal run slime/tests/modal_train.py::prepare # once
modal run slime/tests/modal_train.py::execute
```
<!-- prepare_dataset
download_model
train -->
