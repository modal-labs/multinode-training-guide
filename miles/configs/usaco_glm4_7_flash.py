"""USACO Harbor config for GLM-4.7-Flash full-weight training."""

from .base import (
    DEFAULT_HARBOR_ARGS,
    GLM4_7_FLASH_MODEL_ARGS,
    RLConfig,
)


def get_config(sync: bool = False) -> RLConfig:
    return RLConfig(
        model_name="GLM-4.7-Flash",
        model_id="zai-org/GLM-4.7-Flash",
        miles_model_name="glm4-7-flash",
        model_args=GLM4_7_FLASH_MODEL_ARGS,
        use_ref_load=False,
        app_name="miles-harbor-usaco-glm4-7-flash",
        n_nodes=5,
        gpu="H100:8",
        sync=True,
        actor_num_nodes=4,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=4,
        wandb_project="miles-harbor",
        wandb_run_name_prefix="usaco-glm4-7-flash",
        harbor_task_mode="usaco",
        harbor_task_limit=4,
        harbor_task_ids=["84", "86", "84", "86"],
        dataset_relpath="harbor/usaco/train-limit-4.jsonl",
        miles_args=f"""
            --prompt-data {{dataset_path}}
            --num-rollout 8
            --rollout-batch-size 2
            --n-samples-per-prompt 2
            --global-batch-size 8
            --save {{checkpoints_path}}/runs/usaco-glm4-7-flash
            --save-interval 1
            --no-save-optim
            --attention-dropout 0.0
            --hidden-dropout 0.0
            --accumulate-allreduce-grads-in-fp32
            --attention-softmax-in-fp32
            --attention-backend flash
            --actor-num-nodes 4
            --actor-num-gpus-per-node 8
            --rollout-num-gpus 8
            --rollout-num-gpus-per-engine 4
            --tensor-model-parallel-size 4
            --sequence-parallel
            --pipeline-model-parallel-size 1
            --context-parallel-size 1
            --expert-model-parallel-size 8
            --expert-tensor-parallel-size 1
            --recompute-granularity full
            --recompute-method uniform
            --recompute-num-layers 1
            --use-dynamic-batch-size
            --max-tokens-per-gpu 8192
            --advantage-estimator grpo
            --kl-loss-coef 0.0
            --kl-loss-type low_var_kl
            --kl-coef 0.0
            --entropy-coef 0.0
            --eps-clip 0.2
            --eps-clip-high 0.28
            --optimizer adam
            --lr 1e-5
            --lr-decay-style constant
            --weight-decay 0.1
            --adam-beta1 0.9
            --adam-beta2 0.98
            --use-miles-router
            --sglang-router-port 30000
            --sglang-mem-fraction-static 0.8
            --use-fault-tolerance
            --rollout-health-check-interval 5
            --rollout-health-check-timeout 10
            --rollout-health-check-first-wait 0
            {DEFAULT_HARBOR_ARGS}
        """,
    )
