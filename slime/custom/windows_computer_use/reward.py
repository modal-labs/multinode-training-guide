"""Custom reward model for Windows computer use.

Reads the reward computed by the environment during the rollout
(stored in sample.metadata["env_reward"]) and returns it.
"""

from __future__ import annotations

from typing import Any

from slime.utils.types import Sample


async def compute_reward(args: Any, sample: Sample) -> float:
    """Return the reward stored by the environment.

    The WindowsComputerUseEnv computes reward at the end of each episode
    by checking whether C:\\output.txt contains the target text.
    The generate function stores it in sample.metadata["env_reward"].
    """
    metadata = sample.metadata or {}
    return float(metadata.get("env_reward", 0.0))
