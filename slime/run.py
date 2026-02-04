#!/usr/bin/env python3
"""
Run SLIME GRPO training on Modal.

Usage:
    modal run run.py --config glm-4-7
    modal run run.py --config glm-4-7 --gpu-name H100 --gpu-count 8 --nodes 4
"""

import os
from typing import Optional

from configs import get_config

def _setup_env_from_args():
    import sys

    config = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config = args[i + 1]
            break
        elif arg.startswith("--config="):
            config = arg.split("=", 1)[1]
            break

    if not config:
        return

    cfg = get_config(config)

    gpu_name = None
    gpu_count = None
    nodes = None

    for i, arg in enumerate(args):
        if arg == "--gpu-name" and i + 1 < len(args):
            gpu_name = args[i + 1]
        elif arg.startswith("--gpu-name="):
            gpu_name = arg.split("=", 1)[1]
        elif arg == "--gpu-count" and i + 1 < len(args):
            gpu_count = args[i + 1]
        elif arg.startswith("--gpu-count="):
            gpu_count = arg.split("=", 1)[1]
        elif arg == "--nodes" and i + 1 < len(args):
            nodes = args[i + 1]
        elif arg.startswith("--nodes="):
            nodes = arg.split("=", 1)[1]

    os.environ["APP_NAME"] = cfg.app_name
    os.environ["GPU_NAME"] = gpu_name or cfg.gpu.split(":")[0]
    os.environ["GPU_COUNT"] = str(gpu_count or int(cfg.gpu.split(":")[1]))
    os.environ["NUM_NODES"] = str(nodes or cfg.n_nodes)


_setup_env_from_args()

from modal_train import app, train_multi_node


@app.local_entrypoint()
def main(
    config: str,
    gpu_name: Optional[str] = None,
    gpu_count: Optional[int] = None,
    nodes: Optional[int] = None,
):
    print(f"Config:  {config}")
    print(f"App:     {os.environ['APP_NAME']}")
    print(f"GPU:     {os.environ['GPU_NAME']}:{os.environ['GPU_COUNT']}")
    print(f"Nodes:   {os.environ['NUM_NODES']}")

    train_multi_node.remote(config)
