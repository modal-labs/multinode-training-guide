import os
import modal
import modal.experimental

app = modal.App("example-grpo-slime-1-12")


# copied from verl modal_train
from pathlib import Path

CONTAINER_HOME: Path = Path("/home/ec2-user")
SLIME_REPO_PATH: Path = CONTAINER_HOME / "slime"

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260106a")
    .run_commands(
        # Update slime from GitHub to get the latest version with onload_weights
        "cd /root/slime && git pull origin main && pip install -e ."
    )
    .entrypoint([])
)

DATA_PATH: Path = Path("/data")
data_volume: modal.Volume = modal.Volume.from_name(
    "grpo-slime-example-data", create_if_missing=True
)
MODELS_PATH: Path = Path("/models")
checkpoints_volume: modal.Volume = modal.Volume.from_name(
    "grpo-slime-example-checkpoints", create_if_missing=True
)


@app.function(
    image=image,
    volumes={
        DATA_PATH.as_posix(): data_volume,
        MODELS_PATH.as_posix(): checkpoints_volume,
    },
)
def prepare():
    checkpoints_volume.reload()
    import slime.utils.external_utils.command_utils as U

    MODEL_NAME = "Qwen2.5-0.5B-Instruct"

    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir {MODELS_PATH}/{MODEL_NAME}")
    # U.hf_download_dataset("zhuzilin/gsm8k")
    U.exec_command(f"hf download --repo-type dataset zhuzilin/gsm8k --local-dir {DATA_PATH}/gsm8k")
    data_volume.commit()
    checkpoints_volume.commit()

@app.function(
    image=image,
    gpu="H100:2",
    volumes={
        MODELS_PATH.as_posix(): checkpoints_volume,
        DATA_PATH.as_posix(): data_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret-clairez"),
    ],
    timeout=24 * 60 * 60,
    experimental_options={
        "efa_enabled": True,
    },
)
def execute():
    import slime.utils.external_utils.command_utils as U

    FEW_GPU = U.get_bool_env_var("SLIME_TEST_FEW_GPU", "1")
    TIGHT_DEVICE_MEMORY = U.get_bool_env_var("SLIME_TEST_TIGHT_DEVICE_MEMORY", "1")

    MODEL_NAME = "Qwen2.5-0.5B-Instruct"
    MODEL_TYPE = "qwen2.5-0.5B"
    NUM_GPUS = 2 if FEW_GPU else 4


    ckpt_args = f"--hf-checkpoint {MODELS_PATH}/{MODEL_NAME}/ " f"--ref-load {MODELS_PATH}/{MODEL_NAME}/ "

    rollout_args = (
        f"--prompt-data {DATA_PATH}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else 250} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 256 "
    )

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data gsm8k {DATA_PATH}/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        # "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {0.6 if TIGHT_DEVICE_MEMORY else 0.7} "
        "--sglang-enable-metrics "
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.55 "  # loose threshold at 250 step
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {2 if FEW_GPU else 4} "
        "--colocate "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="slime/train.py",
    )


@app.local_entrypoint()
def main():
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare.remote()
    execute.remote()