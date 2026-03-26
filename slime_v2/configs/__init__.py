"""Experiment config registry.

To add a new experiment:
  1. Create configs/<your_experiment>.py with `modal` and `slime` instances.
  2. Import it here and add an entry to CONFIGS.
"""

from . import (
    qwen_4b_gsm8k,
    qwen_8b_gsm8k,
    glm47_flash_dapo,
    glm47_flash_dapo_multinode,
    qwen3vl_geo3k_vlm,
)

CONFIGS = {
    "qwen-4b-gsm8k": qwen_4b_gsm8k,
    "qwen-8b-gsm8k": qwen_8b_gsm8k,
    "glm4.7-flash-dapo": glm47_flash_dapo,
    "glm4.7-flash-dapo-2n": glm47_flash_dapo_multinode,
    "qwen3vl-geo3k-vlm": qwen3vl_geo3k_vlm,
}


def get_module(name: str):
    """Return the config module for the given config name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config {name!r}. Available: {sorted(CONFIGS)}")
    return CONFIGS[name]
