"""Base configuration with defaults for SLIME GRPO training on Modal."""

from dataclasses import dataclass, field, fields
from typing import Optional


def _to_cli_args(obj, exclude: list[str] | None = None) -> list[str]:
    """Convert dataclass fields to CLI args. Underscores become hyphens. Bools become flags."""
    args = []
    exclude = exclude or []
    for f in fields(obj):
        if f.name in exclude:
            continue
        val = getattr(obj, f.name)
        flag = f"--{f.name.replace('_', '-')}"
        if val is None:
            continue
        elif isinstance(val, bool):
            if val:
                args.append(flag)
        else:
            args.append(f"{flag} {val}")
    return args


@dataclass
class ModelArchitectureConfig:
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    vocab_size: int = 151936
    rotary_base: int = 1000000
    norm_epsilon: float = 1e-6
    kv_channels: Optional[int] = None
    normalization: str = "RMSNorm"
    swiglu: bool = True
    group_query_attention: bool = True
    use_rotary_position_embeddings: bool = True
    disable_bias_linear: bool = True
    add_qkv_bias: bool = False
    qk_layernorm: bool = False

    def to_args(self) -> str:
        return " ".join(_to_cli_args(self))


@dataclass
class PerformanceConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    max_tokens_per_gpu: int = 9216
    sequence_parallel: bool = True
    use_dynamic_batch_size: bool = True

    def to_args(self) -> str:
        return " ".join(_to_cli_args(self))


@dataclass
class RolloutConfig:
    num_rollout_infinite: int = 3000
    num_rollout_default: int = 250
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 1024
    rollout_temperature: float = 1.0
    over_sampling_batch_size: int = 64
    global_batch_size: int = 256

    def to_args(self, data_path, is_infinite_run: bool) -> str:
        num_rollout = self.num_rollout_infinite if is_infinite_run else self.num_rollout_default
        return " ".join([
            f"--prompt-data {data_path}/gsm8k/train.parquet",
            "--input-key messages --label-key label --apply-chat-template --rollout-shuffle --rm-type math",
            f"--num-rollout {num_rollout}",
            "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
            *_to_cli_args(self, exclude=["num_rollout_infinite", "num_rollout_default"]),
        ])


@dataclass
class ClusterConfig:
    """Cluster and GPU allocation for training/rollout."""
    # Actor (training)
    actor_num_nodes: int = 2
    actor_num_gpus_per_node: int = 8
    # Critic (for PPO, usually None for GRPO)
    critic_num_nodes: Optional[int] = None
    critic_num_gpus_per_node: Optional[int] = None
    # Rollout (inference)
    rollout_num_gpus: Optional[int] = None # defaults to actor_num_gpus_per_node * actor_num_nodes
    num_gpus_per_node: int = 8
    # Colocation
    colocate: bool = False
    offload: bool = False
    offload_train: Optional[bool] = None
    offload_rollout: Optional[bool] = None

    def to_args(self) -> str:
        return " ".join(_to_cli_args(self))


@dataclass
class SGLangConfig:
    """SGLang inference engine configuration."""
    sglang_mem_fraction_static: float = 0.7

    def to_args(self) -> str:
        return " ".join([*_to_cli_args(self), "--sglang-enable-metrics"])


@dataclass
class GRPOConfig:
    advantage_estimator: str = "grpo"
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.00
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.00
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28

    def to_args(self) -> str:
        return " ".join(_to_cli_args(self))


@dataclass
class OptimizerConfig:
    optimizer: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98

    def to_args(self) -> str:
        return " ".join(_to_cli_args(self))


@dataclass
class EvalConfig:
    eval_interval: int = 20
    n_samples_per_eval_prompt: int = 1
    eval_max_response_len: int = 1024
    eval_top_k: int = 1

    def to_args(self, data_path) -> str:
        return " ".join([f"--eval-prompt-data gsm8k {data_path}/gsm8k/test.parquet", *_to_cli_args(self)])


@dataclass
class CIConfig:
    ci_test: bool = True
    ci_disable_kl_checker: bool = True
    ci_metric_checker_key: str = "eval/gsm8k"
    ci_metric_checker_threshold: float = 0.55

    def to_args(self) -> str:
        return " ".join(_to_cli_args(self))


@dataclass
class MiscConfig:
    """Miscellaneous training settings."""
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_backend: str = "flash"
    megatron_to_hf_mode: str = "bridge"

    def to_args(self) -> str:
        return " ".join([*_to_cli_args(self), "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32"])


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Model
    model_name: str
    model_org: str = "Qwen"
    
    # Nested configs
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ci: CIConfig = field(default_factory=CIConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)
    
    # Modal
    app_name: str = "slime-grpo"
    n_nodes: int = 4
    gpu_type: str = "H100:2"
    
    # Wandb
    wandb_project: str = "slime-grpo"
    wandb_run_name_prefix: str = ""
    
    # Training mode
    use_async: bool = False
    
    # Extra args passthrough (for any slime arg not in configs)
    extra_args: list[str] = field(default_factory=list)
    
    @property
    def model_id(self) -> str:
        return f"{self.model_org}/{self.model_name}"
    
    @property
    def train_script(self) -> str:
        return "slime/train_async.py" if self.use_async else "slime/train.py"
    
    def generate_train_args(self, models_path, data_path, is_infinite_run: bool) -> str:
        args = [
            f"--hf-checkpoint {models_path}/{self.model_name}/ --ref-load {models_path}/{self.model_name}/",
            self.architecture.to_args(),
            self.cluster.to_args(),
            self.rollout.to_args(data_path, is_infinite_run),
            self.optimizer.to_args(),
            self.grpo.to_args(),
            self.performance.to_args(),
            self.eval.to_args(data_path),
            self.sglang.to_args(),
            self.ci.to_args(),
            self.misc.to_args(),
        ]
        if self.extra_args:
            args.extend(self.extra_args)
        return " ".join(args)
