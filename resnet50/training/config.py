"""
Hyperparameter configuration and training cluster configuration.
"""
nodes = 4  # Number of containers in the cluster.
gpus_per_node = 8  # Number of GPUs per container.
batch_size = 512 * gpus_per_node * nodes

epsilon = 1e-5
if batch_size == 512:
    learning_rate = float(2**2)
    warmup_epochs = 10 / 2**6
elif batch_size == 4096:
    learning_rate = float(2**3.5)
    warmup_epochs = 10 / 2**3
elif batch_size == 8192:
    learning_rate = float(2**4)
    warmup_epochs = 10 / 2**2
elif batch_size == 16384:
    learning_rate = float(2**4.5)
    warmup_epochs = 10 / 2**1
elif batch_size == 32768:
    learning_rate = float(2**5)
    warmup_epochs = 14
    epsilon = 0
else:
    raise ValueError(f"Unsupported batch size: {batch_size}")

epochs = 90
momentum = 0.9
weight_decay = 0.0001
debug = False
benchmark = False
runtime = "gvisor"
assert runtime in ["gvisor", "runc"]
run_name = f"lars-{batch_size}-{nodes}x{gpus_per_node}"
if benchmark:
    run_name += "-benchmark"
if runtime == "runc":
    run_name += "-runc"
if debug:
    run_name += "-debug"
