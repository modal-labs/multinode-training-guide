import glob
import os


# Reads IB port counters from sysfs for all devices.
def read_port_counters(
    counters_dict: dict[str, float], multiplier: float = 1
) -> dict[str, float]:
    # Initialize a dict to store the counters
    # Loop through all ports and counters on every device
    for path in glob.glob("/sys/class/infiniband/*/ports/*/counters/*"):
        # Get the metric name from the path
        metric = os.path.basename(path)
        # If the metric is in the counters dict, read the value and add it to the dict
        if metric in counters_dict:
            with open(path, "r") as f:
                # Optionally multiplies the resulting value by 4 to convert from words to bytes
                counters_dict[metric] += float(f.read()) * multiplier
    return counters_dict
