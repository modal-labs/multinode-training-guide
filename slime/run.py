#!/usr/bin/env python3
"""
Run SLIME GRPO training on Modal.

Usage:
    python run.py <config-name> [options]
    python run.py glm-4-7
    python run.py glm-4-7 --gpu-name H100 --gpu-count 8 --nodes 4
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Run SLIME GRPO training on Modal")


@app.command()
def train(
    config: str = typer.Argument(..., help="Config name (e.g., glm-4-7, qwen-4b)"),
    gpu_name: Optional[str] = typer.Option(None, "--gpu-name", "-g", help="GPU name (e.g., H100, H200)"),
    gpu_count: Optional[int] = typer.Option(None, "--gpu-count", "-c", help="GPU count per node"),
    nodes: Optional[int] = typer.Option(None, "--nodes", "-n", help="Number of nodes"),
):
    """Run multi-node GRPO training with the specified config."""
    script_dir = Path(__file__).parent

    # Import config
    sys.path.insert(0, str(script_dir))
    from configs import get_config

    cfg = get_config(config)

    # Build environment variables with overrides
    env = {
        **os.environ,
        "APP_NAME": cfg.app_name,
        "GPU_NAME": gpu_name or cfg.gpu.split(":")[0],
        "GPU_COUNT": str(gpu_count or cfg.gpu.split(":")[1]),
        "NUM_NODES": str(nodes or cfg.n_nodes),
    }

    # Print config info
    typer.echo(f"Config:  {config}")
    typer.echo(f"App:     {cfg.app_name}")
    typer.echo(f"GPU:     {env['GPU_NAME']}:{env['GPU_COUNT']}")
    typer.echo(f"Nodes:   {env['NUM_NODES']}")

    # Run modal
    cmd = [
        "modal",
        "run",
        "-d",
        "modal_train.py::train_multi_node",
        "--config",
        config,
    ]

    subprocess.run(cmd, cwd=script_dir, env=env, check=True)


if __name__ == "__main__":
    app()
