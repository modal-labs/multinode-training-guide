import os
import subprocess
from pathlib import Path
from typing import Literal, Optional
import time
import modal
import modal.experimental

app = modal.App("example-grpo-slime")


# copied from verl modal_train
from pathlib import Path

CONTAINER_HOME: Path = Path("/home/ec2-user")
SLIME_REPO_PATH: Path = CONTAINER_HOME / "slime"

image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260106a")
    .run_commands(
        # Update slime from GitHub to get the latest version with onload_weights
        "cd /root/slime && git remote set-url origin https://github.com/czhang-modal/slime.git && git fetch origin && git checkout claire/slime && pip install -e ."
    )
    .entrypoint([])
)

with image.imports():
    import ray
    from ray.job_submission import JobSubmissionClient

DATA_PATH: Path = Path("/data")
data_volume: modal.Volume = modal.Volume.from_name(
    "grpo-slime-example-data", create_if_missing=True
)
MODELS_PATH: Path = Path("/models")
checkpoints_volume: modal.Volume = modal.Volume.from_name(
    "grpo-slime-example-checkpoints", create_if_missing=True
)

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_ID: str = f"Qwen/{MODEL_NAME}"
TRAINING_CHECKPOINT_DIR: Path = MODELS_PATH / "training_checkpoints" / MODEL_ID


N_NODES = 4

# Wandb configuration
WANDB_PROJECT = "slime-grpo"
WANDB_RUN_NAME_PREFIX = "sync-qwen-0.5b-gsm8k"

# Ray configuration
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265

SINGLE_NODE_MASTER_ADDR = "127.0.0.1"



def _init_ray(rank: int, main_node_addr: str, node_ip_addr: str, n_nodes: int):
    """Initialize Ray cluster across Modal containers.

    Rank 0 starts the head node, opens a tunnel to the Ray dashboard, and waits
    for all worker nodes to connect. Other ranks start as workers and connect
    to the head node address.
    """
    os.environ["SLIME_HOST_IP"] = node_ip_addr 

    if rank == 0:
        subprocess.Popen(
            [
                "ray",
                "start",
                "--head",
                f"--node-ip-address={node_ip_addr}",
                "--dashboard-host=0.0.0.0",
            ]
        )

        for _ in range(10):
            try:
                ray.init(address="auto")
            except ConnectionError:
                time.sleep(1)
                continue
            break
        else:
            raise Exception("Failed to connect to Ray")

        for _ in range(60):
            print("Waiting for worker nodes to connect...")
            alive_nodes = [n for n in ray.nodes() if n["Alive"]]
            print(f"Alive nodes: {alive_nodes}")

            if len(alive_nodes) == n_nodes:
                print("All worker nodes connected")
                break
            time.sleep(1)
        else:
            raise Exception("Failed to connect to all worker nodes")
    else:
        subprocess.Popen(
            [
                "ray",
                "start",
                f"--node-ip-address={node_ip_addr}",
                "--address",
                f"{main_node_addr}:{RAY_PORT}",
            ]
        )


@app.function(
    image=image,
    volumes={MODELS_PATH.as_posix(): checkpoints_volume},
    timeout=24 * 60 * 60,
)
def download_model(
    repo_id: str = MODEL_ID,
    revision: Optional[str] = None,  # include a revision to prevent surprises!
):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=MODELS_PATH / MODEL_NAME, revision=revision)
    print(f"Model downloaded to {MODELS_PATH / MODEL_NAME}")

    checkpoints_volume.commit()

@app.function(
    image=image,
    volumes={DATA_PATH.as_posix(): data_volume},
    timeout=24 * 60 * 60,
)
def prepare_dataset():
    from datasets import load_dataset

    data_volume.reload()
    dataset = load_dataset("zhuzilin/gsm8k")
    dataset["train"].to_parquet(f"{DATA_PATH}/gsm8k/train.parquet")
    dataset["test"].to_parquet(f"{DATA_PATH}/gsm8k/test.parquet")
    data_volume.commit()


def generate_slime_cmd(n_nodes: int, arglist: list[str], master_addr: str):
    import slime.utils.external_utils.command_utils as U


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

    # Model architecture args for Qwen2.5-0.5B-Instruct (from HF config)
    model_args = (
        "--swiglu "
        "--num-layers 24 "
        "--hidden-size 896 "
        "--ffn-hidden-size 4864 "
        "--num-attention-heads 14 "
        "--use-rotary-position-embeddings "
        "--disable-bias-linear "
        "--add-qkv-bias "
        "--normalization RMSNorm "
        "--norm-epsilon 1e-6 "
        "--rotary-base 1000000 "
        "--group-query-attention "
        "--num-query-groups 2 "
        "--vocab-size 151936 "
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
        "--sglang-mem-fraction-static .7 " # or .6, or .5 as the verl default
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
        "--actor-num-gpus-per-node 2 " # number of gpus per node
        "--colocate "
        "--megatron-to-hf-mode bridge "
    )

    wandb_args = (
        f"{U.get_default_wandb_args(__file__, WANDB_RUN_NAME_PREFIX)} "
        f"--wandb-project {WANDB_PROJECT} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{model_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    if arglist:
        train_args.extend(arglist)

    # must have nvlink, eg. pass check_has_nvlink() in slime/utils/external_utils/command_utils.py

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            # If setting this in FSDP, the computation communication overlapping may have issues
            **({"CUDA_DEVICE_MAX_CONNECTIONS": "1"}),
            "NCCL_NVLS_ENABLE": "1",
            "no_proxy": master_addr,
            # This is needed by megatron / torch distributed in multi-node setup
            "MASTER_ADDR": master_addr,
        }
    }

    return f"python3 slime/train.py {train_args}", runtime_env



async def run_training(n_nodes: int, arglist: list[str], master_addr: str):
    """Submit verl training job to Ray cluster and stream logs.

    Uses Ray's JobSubmissionClient to submit the training command and
    asynchronously tails logs until the job completes.
    """
    client = JobSubmissionClient("http://127.0.0.1:8265")

    slime_cmd, runtime_env = generate_slime_cmd(n_nodes, arglist, master_addr)
    job_id = client.submit_job(
        entrypoint=slime_cmd,
        runtime_env=runtime_env
    )
    print(f"Job submitted with ID: {job_id}")

    async for line in client.tail_job_logs(job_id):
        print(line, end="", flush=True)


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
@modal.experimental.clustered(N_NODES, rdma=True)
async def train_multi_node(*arglist):
    """Main entry point for multi-node GRPO training on Modal.

    Spins up N_NODES containers with 8xH100 GPUs each, connected via RDMA.
    Rank 0 initializes Ray and submits the training job; other ranks join
    as workers and block until training completes.
    """
    checkpoints_volume.reload()
    data_volume.reload()

    cluster_info = modal.experimental.get_cluster_info()
    print(f"Rank: {cluster_info.rank}, task id: {os.environ['MODAL_TASK_ID']}")
    print(f"Container IPs: {cluster_info.container_ips}")
    print(f"Container IPv4 IPs: {cluster_info.container_ipv4_ips}")

    ray_main_node_addr = cluster_info.container_ipv4_ips[0]
    my_ip_addr = cluster_info.container_ipv4_ips[cluster_info.rank]

    _init_ray(cluster_info.rank, ray_main_node_addr, my_ip_addr, N_NODES)

    if cluster_info.rank == 0:
        with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
            dashboard_url = tunnel.url
            print(f"Dashboard URL: {dashboard_url}")

            await run_training(N_NODES, list(arglist), ray_main_node_addr)
    else:
        # We have to keep the worker node alive until the training is complete. Once rank 0
        # finishes, all workers will be terminated.
        while True:
            time.sleep(10)



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
async def train_single_node(*arglist):
    checkpoints_volume.reload()
    data_volume.reload()

    _init_ray(0, SINGLE_NODE_MASTER_ADDR, SINGLE_NODE_MASTER_ADDR, 1)

    with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Dashboard URL: {tunnel.url}")
        await run_training(1, list(arglist), SINGLE_NODE_MASTER_ADDR)