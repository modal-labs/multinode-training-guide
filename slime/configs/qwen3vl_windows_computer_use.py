"""Qwen3-VL-2B-Instruct GRPO for Windows computer use.

Trains a VLM to control a Windows VM by looking at screenshots and
emitting keyboard/mouse actions. Tasks range from simple Notepad save
to PowerShell commands and multi-step file operations.

Uses Slime's VLM multi-turn rollout with a custom environment that
wraps Windows sandboxes (QEMU VMs on Modal).
"""

from configs.base import ModalConfig, SlimeConfig, DATA_PATH

modal = ModalConfig(
    gpu="H200",
    image_run_commands=[
        # Install modal SDK + PIL inside the Slime container so rollout
        # workers can create Windows sandboxes and process screenshots.
        "pip install modal Pillow",
    ],
)


class _Slime(SlimeConfig):
    # VL models need the model script for architecture args + rotary base
    slime_model_script = "scripts/models/qwen3-1.7B.sh"
    environment = {
        **SlimeConfig.environment,
        "MODEL_ARGS_ROTARY_BASE": "5000000",
        # Ensure custom modules are importable by Ray/Slime workers
        "PYTHONPATH": "/root/Megatron-LM/:/root",
    }

    # ── Model ─────────────────────────────────────────────────────────────────
    hf_checkpoint = "Qwen/Qwen3-VL-2B-Instruct"
    load = "Qwen/Qwen3-VL-2B-Instruct"
    megatron_to_hf_mode = "bridge"

    # ── Infrastructure ────────────────────────────────────────────────────────
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = True

    # ── Data ──────────────────────────────────────────────────────────────────
    prompt_data = f"{DATA_PATH}/windows_computer_use/train.parquet"
    eval_prompt_data = [
        "windows_computer_use",
        f"{DATA_PATH}/windows_computer_use/test.parquet",
    ]
    input_key = "messages"
    label_key = "target"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "custom"

    # ── Rollout — VLM multi-turn with Windows env ─────────────────────────────
    num_rollout = 4
    rollout_batch_size = 8
    n_samples_per_prompt = 8
    rollout_max_response_len = 4096
    rollout_temperature = 1.0
    rollout_num_gpus_per_engine = 1
    sglang_mem_fraction_static = 0.7
    sglang_cuda_graph_bs = [1, 2, 4, 8, 16, 24, 32]
    global_batch_size = 32

    # Custom multi-turn VLM rollout + environment + reward
    custom_generate_function_path = "custom.windows_computer_use.rollout.generate"
    custom_rm_path = "custom.windows_computer_use.reward.compute_reward"
    custom_config_path = {
        "max_turns": 10,
        "rollout_interaction_env_path": ("custom.windows_computer_use.env_windows"),
    }

    # ── Eval ──────────────────────────────────────────────────────────────────
    eval_interval = 20
    n_samples_per_eval_prompt = 1
    eval_max_response_len = 4096

    # ── Training ──────────────────────────────────────────────────────────────
    tensor_model_parallel_size = 1
    sequence_parallel = False
    qkv_format = "bshd"
    micro_batch_size = 1
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"

    # ── Algorithm (GRPO) ──────────────────────────────────────────────────────
    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    kl_coef = 0.0
    entropy_coef = 0.0

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = "adam"
    lr = 1e-5
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = True
    wandb_project = "slime-windows-computer-use"
    wandb_group = "qwen3-vl-2b-windows-multitask"
    disable_wandb_random_suffix = True

    def prepare_data(self) -> None:
        """Generate prompt/target parquet for the Notepad task."""
        from custom.windows_computer_use.dataset import generate_dataset

        generate_dataset(f"{DATA_PATH}/windows_computer_use")


slime = _Slime()
