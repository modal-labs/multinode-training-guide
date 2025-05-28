# Utils

This directory contains utility scripts.

- `mlx_monitor.py`: A script to monitor the network and GPU usage of a machine.

## mlx_monitor.py

Add the monitor script to your image:

```python
image = base_image.add_local_file("../utils/mlx_monitor.py", remote_path="/root/mlx_monitor.py")
```

Then shell into one of your containers and run the script. Container IDs can be found in the Containers tab of your Function's dashboard.

```bash
$ modal shell ta-01JWC5ZQZSAGXK2YMC8VSZGCG0
$> uv run mlx_monitor.py
```

This should give you a table of the network and GPU usage of the machine:

![MLX Monitor screenshot](../assets/mlx_monitor.png)
