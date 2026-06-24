"""Qwen2.5-1.5B GRPO on OpenMathInstruct-2 — single node, 8 GPUs.

Mirrors NeMo-RL's examples/configs/grpo_math_1B.yaml, bumped to a full 8-GPU
node. The dataset is downloaded automatically by NeMo-RL at runtime into the
mounted HF cache, so download_data is only used to warm that cache.
"""

from configs.base import ModalConfig, NemoRLConfig

modal = ModalConfig(gpu="H100")


class _Recipe(NemoRLConfig):
    entrypoint = "examples/run_grpo.py"
    base_config = "examples/configs/grpo_math_1B.yaml"

    num_nodes = 1
    gpus_per_node = 8

    hf_model = "Qwen/Qwen2.5-1.5B"
    hf_datasets = ["nvidia/OpenMathInstruct-2"]

    overrides = {
        "policy.model_name": "Qwen/Qwen2.5-1.5B",
        # Spread generation + training across all 8 GPUs.
        "policy.generation.vllm_cfg.tensor_parallel_size": 1,
        "logger.wandb_enabled": True,
        "logger.wandb.project": "nemo-rl-grpo",
        "logger.wandb.name": "qwen2.5-1.5b-math",
    }


nemo_rl = _Recipe()
