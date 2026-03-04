import glob
import os


# Reads RDMA port counters from sysfs for all devices.
def read_port_counters(
    path_pattern: str, counters_dict: dict[str, float], multiplier: float = 1
) -> dict[str, float]:
    for path in glob.glob(path_pattern):
        metric = os.path.basename(path)
        if metric in counters_dict:
            with open(path, "r") as f:
                counters_dict[metric] += float(f.read().strip()) * multiplier
    return counters_dict
