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
    """Model architecture parameters."""
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
class MegatronConfig:
    """Megatron training settings - parallelism, batching, optimizer."""
    # Parallelism
    tensor_model_parallel_size: int = 1  # Split model horizontally across GPUs (TP)
    pipeline_model_parallel_size: int = 1  # Split model layers across GPUs (PP)
    context_parallel_size: int = 1  # Parallelism for long sequences
    expert_model_parallel_size: int = 1  # MoE: split experts across GPUs
    expert_tensor_parallel_size: int = 1  # MoE: TP within each expert
    sequence_parallel: bool = True  # Reduce activation memory
    # Batching
    max_tokens_per_gpu: int = 9216  # Micro-batch size in tokens
    use_dynamic_batch_size: bool = True  # Variable batch sizes based on sequence lengths
    global_batch_size: int = 256  # Total batch size across all GPUs
    # Optimizer
    optimizer: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    # Misc
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_backend: str = "flash"
    megatron_to_hf_mode: str = "bridge"

    def to_args(self) -> str:
        return " ".join([
            *_to_cli_args(self),
            "--accumulate-allreduce-grads-in-fp32",
            "--attention-softmax-in-fp32",
        ])


@dataclass
class SGLangConfig:
    """SGLang inference engine settings - parallelism, memory, generation."""
    # Parallelism
    rollout_num_gpus_per_engine: Optional[int] = None  # TP size - GPUs per engine
    sglang_data_parallel_size: Optional[int] = None  # DP size - data parallel within engine
    sglang_pipeline_parallel_size: Optional[int] = None  # PP size - pipeline parallel
    sglang_expert_parallel_size: Optional[int] = None  # EP size - expert parallel (MoE)
    # Memory
    sglang_mem_fraction_static: float = 0.7  # Fraction of GPU memory for KV cache
    # Generation
    rollout_batch_size: int = 32  # Prompts per inference batch
    n_samples_per_prompt: int = 8  # Responses to generate per prompt
    rollout_max_response_len: int = 1024  # Max tokens to generate
    rollout_temperature: float = 1.0  # Sampling temperature
    # Debug/testing
    sglang_disable_cuda_graph: bool = False  # Disable CUDA graph capture (faster startup)

    def to_args(self) -> str:
        return " ".join([*_to_cli_args(self), "--sglang-enable-metrics"])


@dataclass
class OrchestrationConfig:
    """Cluster allocation, colocation, and run settings."""
    # Actor (training) GPU allocation
    actor_num_nodes: int = 2
    actor_num_gpus_per_node: int = 8 # overrides num_gpus_per_node
    # Critic (for PPO, None for GRPO)
    critic_num_nodes: Optional[int] = None
    critic_num_gpus_per_node: Optional[int] = None
    # Rollout GPU allocation
    rollout_num_gpus: Optional[int] = None  # Total GPUs for rollout (defaults to actor GPUs)
    num_gpus_per_node: int = 8
    # Colocation (training and rollout share GPUs)
    colocate: bool = False
    offload: bool = False
    offload_train: Optional[bool] = None
    offload_rollout: Optional[bool] = None
    # Run settings
    num_rollout_infinite: int = 3000  # Rollout batches for infinite runs
    num_rollout_default: int = 250  # Rollout batches for default runs
    over_sampling_batch_size: int = None  # Oversampling buffer size

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
class GRPOConfig:
    """GRPO algorithm settings."""
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
class EvalConfig:
    """Evaluation settings."""
    eval_interval: int = 20
    n_samples_per_eval_prompt: int = 1
    eval_max_response_len: int = 1024
    eval_top_k: int = 1

    def to_args(self, data_path) -> str:
        return " ".join([f"--eval-prompt-data gsm8k {data_path}/gsm8k/test.parquet", *_to_cli_args(self)])


@dataclass
class CIConfig:
    """CI and testing settings."""
    ci_test: bool = True
    ci_disable_kl_checker: bool = True
    ci_metric_checker_key: str = "eval/gsm8k"
    ci_metric_checker_threshold: float = 0.55

    def to_args(self) -> str:
        return " ".join(_to_cli_args(self))


@dataclass
class RLConfig:
    """Main RL training configuration."""
    
    # Model
    model_name: str
    model_org: str = "Qwen"
    
    # Nested configs
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: MegatronConfig = field(default_factory=MegatronConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ci: CIConfig = field(default_factory=CIConfig)
    
    # Modal
    app_name: str = "slime-grpo"
    n_nodes: int = 4
    gpu: str = "H100:8"  # Modal GPU spec (e.g., "H100:8", "A100:4")
    
    # Wandb
    wandb_project: str = "slime-grpo"
    wandb_run_name_prefix: str = ""
    
    # Training mode
    sync: bool = True
    
    # Extra args passthrough (for any slime arg not in configs)
    extra_args: list[str] = field(default_factory=list)
    
    @property
    def model_id(self) -> str:
        return f"{self.model_org}/{self.model_name}"
    
    @property
    def train_script(self) -> str:
        return "slime/train.py" if self.sync else "slime/train_async.py"
    
    def generate_train_args(self, models_path, data_path, is_infinite_run: bool) -> str:
        args = [
            f"--hf-checkpoint {models_path}/{self.model_name}/ --ref-load {models_path}/{self.model_name}/",
            self.architecture.to_args(),
            self.training.to_args(),
            self.sglang.to_args(),
            self.orchestration.to_args(data_path, is_infinite_run),
            self.grpo.to_args(),
            self.eval.to_args(data_path),
            self.ci.to_args(),
        ]
        if self.extra_args:
            args.extend(self.extra_args)
        return " ".join(args)
