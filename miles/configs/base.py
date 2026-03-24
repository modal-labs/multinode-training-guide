"""Base configuration for Miles + Harbor training on Modal."""

from dataclasses import dataclass, field
from pathlib import Path


USACO_GIT_URL = "https://github.com/laude-institute/harbor-datasets.git"
USACO_GIT_COMMIT = "56fac05ddbf784bfaa640f4919bceef50433fbed"


@dataclass
class RLConfig:
    model_name: str
    model_id: str
    miles_model_name: str
    model_args: str

    app_name: str = "miles-harbor"
    n_nodes: int = 1
    gpu: str = "H100:8"
    sync: bool = False

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 1
    rollout_num_gpus: int = 1
    rollout_num_gpus_per_engine: int = 1

    wandb_project: str = "miles-harbor"
    wandb_run_name_prefix: str = ""

    harbor_task_mode: str = "hello"
    harbor_task_limit: int = 1
    harbor_task_ids: list[str] = field(default_factory=list)
    harbor_agent_import_path: str = "harbor_agent.SimpleHarborAgent"
    dataset_relpath: str = ""

    miles_args: str = ""
    extra_args: list[str] = field(default_factory=list)

    @property
    def train_script(self) -> str:
        return "train.py" if self.sync else "train_async.py"

    def _clean_args(self, args: str) -> str:
        lines = []
        for line in args.split("\n"):
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()
            if line:
                lines.append(line)
        return " ".join(lines)

    def dataset_path(self, data_path: Path) -> Path:
        return data_path / self.dataset_relpath

    def bootstrap_checkpoint_path(self, checkpoints_path: Path) -> Path:
        return checkpoints_path / "bootstrap" / self.miles_model_name

    def generate_train_args(self, hf_model_path: str, data_path: Path, checkpoints_path: Path) -> str:
        cleaned_model_args = self._clean_args(self.model_args)
        ref_load_path = self.bootstrap_checkpoint_path(checkpoints_path)
        base_args = (
            f"--hf-checkpoint {hf_model_path} "
            f"--ref-load {ref_load_path} "
            f"--model-name {self.miles_model_name}"
        )

        cleaned = self._clean_args(self.miles_args)
        cleaned = cleaned.replace("{data_path}", str(data_path))
        cleaned = cleaned.replace("{checkpoints_path}", str(checkpoints_path))
        cleaned = cleaned.replace("{dataset_path}", str(self.dataset_path(data_path)))
        extra = " ".join(self.extra_args) if self.extra_args else ""
        return f"{base_args} {cleaned_model_args} {cleaned} {extra}".strip()


QWEN3_0_6B_MODEL_ARGS = """
    --swiglu
    --num-layers 28
    --hidden-size 1024
    --ffn-hidden-size 3072
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 8
    --use-rotary-position-embeddings
    --disable-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --vocab-size 151936
    --kv-channels 128
    --qk-layernorm
"""

QWEN3_1_7B_MODEL_ARGS = """
    --swiglu
    --num-layers 28
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 8
    --use-rotary-position-embeddings
    --disable-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --vocab-size 151936
    --kv-channels 128
    --qk-layernorm
"""

DEFAULT_PERF_ARGS = """
    --tensor-model-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096
"""

DEFAULT_OPTIMIZER_ARGS = """
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
"""

DEFAULT_GRPO_ARGS = """
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
"""

DEFAULT_MILES_ROUTER_ARGS = """
    --use-miles-router
    --sglang-router-port 30000
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.6
"""

DEFAULT_HARBOR_ARGS = """
    --input-key prompt
    --rollout-shuffle
    --rollout-max-response-len 1024
    --rollout-temperature 0.8
    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path harbor_agent_function.run
    --custom-rm-path generate.reward_func
    --rollout-function-path generate.RolloutFn
    --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
    --generate-multi-samples
    --update-weights-interval 1
"""
