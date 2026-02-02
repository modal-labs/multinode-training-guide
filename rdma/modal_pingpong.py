import os
import subprocess

import modal
import modal.experimental

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
)
app = modal.App("rdma-pingpong", image=image)

# The number of containers (i.e. nodes) in the cluster. This can be between 1 and 8.
n_nodes = 2
# Typically this matches the number of GPUs per container.
n_proc_per_node = 8

@app.function(
    gpu=f"H100:{n_proc_per_node}",
    timeout=60 * 60, # 1 hour
)
@modal.experimental.clustered(n_nodes, rdma=True)
def fi_pingpong():
    cluster_info = modal.experimental.get_cluster_info()
    container_rank: int = cluster_info.rank
    local_ip: str = cluster_info.container_ips[container_rank]

    # Create ephemeral dicts and queues
    with modal.Dict.ephemeral() as d, modal.Queue.ephemeral() as q:
        next_element = q.get()
        if next_element is None:
            d["rank"] = container_rank
            d["task_type"] = "receive_task"
            d["local_ip"] = local_ip
            # launch fi_pingpong in server mode
            d["args"] = ["-p", "efa"]
            q.put(d)
        else:
            # receive task already exists
            d["rank"] = container_rank
            d["task_type"] = "send_task"
            d["local_ip"] = local_ip
            # launch fi_pingpong in client mode
            d["args"] = ["-p", "efa", next_element["local_ip"]]
            q.put(d)

        print(f"Running fi_pingpong's {d['task_type']} with args: {' '.join(d['args'])}")
        subprocess.run(["fi_pingpong", *d["args"]], check=True)