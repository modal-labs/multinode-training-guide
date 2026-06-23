"""Llama-3.1-8B-Instruct GRPO on OpenMathInstruct-2 — 2 nodes x 8 GPUs.

Mirrors NeMo-RL's examples/configs/grpo_math_8B.yaml scaled to two nodes. This
is the multi-node example: Modal provisions a 2-node RDMA cluster, the launcher
brings up Ray across both nodes, and the driver runs on the head node with
cluster.num_nodes=2.
"""

from configs.base import ModalConfig, NemoRLConfig

modal = ModalConfig(gpu="H100")


class _Recipe(NemoRLConfig):
    entrypoint = "examples/run_grpo.py"
    base_config = "examples/configs/grpo_math_8B.yaml"

    num_nodes = 2
    gpus_per_node = 8

    hf_model = "meta-llama/Llama-3.1-8B-Instruct"
    hf_datasets = ["nvidia/OpenMathInstruct-2"]

    overrides = {
        "policy.model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "policy.dtensor_cfg.enabled": True,
        "policy.dtensor_cfg.tensor_parallel_size": 8,
        "policy.dtensor_cfg.sequence_parallel": True,
        "policy.dtensor_cfg.activation_checkpointing": True,
        "logger.wandb_enabled": True,
        "logger.wandb.project": "nemo-rl-grpo",
        "logger.wandb.name": "llama3.1-8b-math-2node",
    }


nemo_rl = _Recipe()
