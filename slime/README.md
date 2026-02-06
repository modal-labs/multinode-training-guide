# AsyncRL using slime on Modal



```
modal profile activate modal-labs
modal config set-environment [Your Enviroment Name]
modal deploy modal_train.py # once
modal run modal_train.py::prepare_dataset # once
```

Then, you can run
```
modal run modal_train.py::train_single_node 
```
OR
```
modal run modal_train.py::train_multi_node 
```

The following secrets need to be set: `wandb-secret`.