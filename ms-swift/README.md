# ms-swift with Modal

This demo trains GLM4.7 using ms-swift: https://swift.readthedocs.io/en/latest/.


# Usage 

```
modal run train_glm_4_7_lora.py::download_model
```

```
modal run train_glm_4_7_lora.py::prepare_dataset --hf-dataset openai/gsm8k --data-folder gsm8k --split train
```

```
modal run train_glm_4_7_lora.py::train_model
```