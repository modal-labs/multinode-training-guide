# ms-swift with Modal

This demo trains GLM4.7 using ms-swift: https://swift.readthedocs.io/en/latest/.


# Usage 

```
modal run modal_train.py::download_model
```

```
modal run modal_train.py::prepare_dataset --hf-dataset openai/gsm8k --data-folder gsm8k --split train
```

```
modal run modal_train.py::train_model
```