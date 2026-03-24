"""Fast Harbor proof config for Qwen3-0.6B."""

from .base import (
    DEFAULT_GRPO_ARGS,
    DEFAULT_HARBOR_ARGS,
    DEFAULT_MILES_ROUTER_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_PERF_ARGS,
    QWEN3_0_6B_MODEL_ARGS,
    RLConfig,
)


def get_config(sync: bool = False) -> RLConfig:
    return RLConfig(
        model_name="Qwen3-0.6B",
        model_id="Qwen/Qwen3-0.6B",
        miles_model_name="qwen3-0.6B",
        model_args=QWEN3_0_6B_MODEL_ARGS,
        app_name="miles-harbor-hello-qwen3-0p6b",
        n_nodes=1,
        gpu="H100:8",
        sync=sync,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        wandb_project="miles-harbor",
        wandb_run_name_prefix="hello-qwen3-0p6b",
        harbor_task_mode="hello",
        harbor_task_limit=8,
        dataset_relpath="harbor/hello-world/train.jsonl",
        miles_args=f"""
            --prompt-data {{dataset_path}}
            --num-rollout 2
            --rollout-batch-size 2
            --n-samples-per-prompt 2
            --global-batch-size 4
            --attention-dropout 0.0
            --hidden-dropout 0.0
            --accumulate-allreduce-grads-in-fp32
            --attention-softmax-in-fp32
            --attention-backend flash
            --actor-num-nodes 1
            --actor-num-gpus-per-node 1
            --rollout-num-gpus 1
            --ci-test
            --use-fault-tolerance
            --rollout-health-check-interval 5
            --rollout-health-check-timeout 10
            --rollout-health-check-first-wait 0
            {DEFAULT_PERF_ARGS}
            {DEFAULT_OPTIMIZER_ARGS}
            {DEFAULT_GRPO_ARGS}
            {DEFAULT_MILES_ROUTER_ARGS}
            {DEFAULT_HARBOR_ARGS}
        """,
    )
