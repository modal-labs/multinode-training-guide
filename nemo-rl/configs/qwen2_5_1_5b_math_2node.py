"""Qwen2.5-1.5B GRPO on OpenMathInstruct-2 — 2 nodes x 8 GPUs.

Same recipe as qwen2_5_1_5b_math (grpo_math_1B.yaml), scaled to two nodes.
Modal provisions a 2-node RDMA cluster and NeMo-RL schedules across all 16 GPUs.
"""

from configs.base import ModalConfig, NemoRLConfig

modal = ModalConfig(gpu="H100")


class _Recipe(NemoRLConfig):
    entrypoint = "examples/run_grpo.py"
    base_config = "examples/configs/grpo_math_1B.yaml"

    num_nodes = 2
    gpus_per_node = 8

    hf_model = "Qwen/Qwen2.5-1.5B"
    hf_datasets = ["nvidia/OpenMathInstruct-2"]

    overrides = {
        "policy.model_name": "Qwen/Qwen2.5-1.5B",
        # Base config ties max_input_seq_length, max_model_len, and max_new_tokens
        # all to max_total_sequence_length (512). A 512-token prompt then leaves no
        # room for generation and vLLM raises. Bump it so prompt + output fit.
        "policy.max_total_sequence_length": 1024,
        "policy.generation.vllm_cfg.tensor_parallel_size": 1,
        "logger.wandb_enabled": True,
        "logger.wandb.project": "nemo-rl-grpo",
        "logger.wandb.name": "qwen2.5-1.5b-math-2node",
    }


nemo_rl = _Recipe()
