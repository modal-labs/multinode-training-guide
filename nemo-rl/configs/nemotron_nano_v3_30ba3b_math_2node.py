"""Nemotron-3-Nano-30B-A3B (MoE) GRPO on OpenMathInstruct-2 — 2 nodes x 8 GPUs.

Uses NeMo-RL's convergence-tested 2-node recipe
``examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2.yaml`` verbatim as
the base config: FSDP2 with expert_parallel_size=8 for the 30B/3B-active MoE,
nemo-automodel TransformerEngine backend + DeepEP, and vLLM tensor_parallel_size=4
for generation. Modal provisions a 2-node RDMA cluster (16xH100) and the driver
runs on the head node.

This recipe is only shipped in NeMo-RL >= v0.6.0, so this config pins the
matching v0.6.0 image rather than the repo default (v0.5.0).
"""

from configs.base import ModalConfig, NemoRLConfig

modal = ModalConfig(gpu="H100", docker_image="nvcr.io/nvidia/nemo-rl:v0.6.0")


class _Recipe(NemoRLConfig):
    entrypoint = "examples/run_grpo.py"
    base_config = "examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2.yaml"

    num_nodes = 2
    gpus_per_node = 8

    # Model and tokenizer live in separate HF repos; only the model is prefetched
    # here, the tokenizer is fetched on first use.
    hf_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    hf_datasets = ["nvidia/OpenMathInstruct-2"]

    # The recipe already sets every MoE/parallelism knob; only override logging.
    overrides = {
        "logger.wandb_enabled": True,
        "logger.wandb.project": "nemo-rl-grpo",
        "logger.wandb.name": "nemotron-nano-v3-30ba3b-math-2node",
    }


nemo_rl = _Recipe()
