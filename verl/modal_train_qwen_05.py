import os
import re
import subprocess
from pathlib import Path
from typing import Literal, Optional
import time
import modal
import modal.experimental

app = modal.App("example-grpo-verl-datasets-fix")


VERL_REPO_PATH: Path = Path("/root/verl")
image = (
    modal.Image.from_registry("verlai/verl:vllm011.latest")
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc15",
        "libnl-route-3-200",
        "git",
    )
    .run_commands(
        f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}",
    )
    .uv_pip_install(f"{VERL_REPO_PATH}")
    .entrypoint([])
)

DATA_PATH: Path = Path("/data")
data_volume: modal.Volume = modal.Volume.from_name(
    "grpo-verl-example-data", create_if_missing=True
)
MODELS_PATH: Path = Path("/models")
checkpoints_volume: modal.Volume = modal.Volume.from_name(
    "grpo-verl-example-checkpoints", create_if_missing=True
)


# MODEL_ID: str = "Qwen/Qwen3-32B"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HF_MODEL_DIR: Path = MODELS_PATH / "hf_models" / MODEL_ID
MCORE_MODEL_DIR: Path = MODELS_PATH / "mcore_models" / MODEL_ID
TRAINING_CHECKPOINT_DIR: Path = MODELS_PATH / "training_checkpoints" / MODEL_ID


download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub==1.2.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    volumes={MODELS_PATH.as_posix(): checkpoints_volume},
    image=download_image,
    timeout=60 * 60 * 24,
)
def download_model(
    repo_id: str = MODEL_ID,
    revision: Optional[str] = None,  # include a revision to prevent surprises!
):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=HF_MODEL_DIR, revision=revision)
    print(f"Model downloaded to {HF_MODEL_DIR}")

    checkpoints_volume.commit()


@app.function(
    image=image,
    gpu="H100:2",
    timeout=60 * 60 * 24,
    volumes={MODELS_PATH.as_posix(): checkpoints_volume},
)
def convert_hf_to_mcore() -> None:
    checkpoints_volume.reload()
    subprocess.run(
        [
            "python",
            VERL_REPO_PATH / "scripts" / "converter_hf_to_mcore.py",
            "--hf_model_path",
            str(HF_MODEL_DIR),
            "--output_path",
            str(MCORE_MODEL_DIR),
        ],
        check=True,
    )
    checkpoints_volume.commit()


@app.function(image=image, volumes={DATA_PATH.as_posix(): data_volume})
def prep_dataset() -> None:
    """Download and preprocess GSM8K math dataset into train/test parquet files.

    Runs verl's gsm8k.py preprocessing script and saves output to the data volume.
    Must be run before training.
    """
    subprocess.run(
        [
            "python",
            VERL_REPO_PATH / "examples" / "data_preprocess" / "gsm8k.py",
            "--local_dir",
            DATA_PATH,
        ],
        check=True,
    )


def extract_solution(
    solution_str: str, method: Literal["strict", "flexible"] = "strict"
) -> Optional[str]:
    """Extract numerical answer from model output.

    'strict' mode requires the GSM8K '#### N' format and validates model formatting.
    'flexible' mode finds the last valid number in the string as a fallback.
    Returns None if no answer is found.
    """
    assert method in ["strict", "flexible"]

    if method == "strict":
        # This also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer: Optional[str] = None
        else:
            final_answer = solution.group(0)
            final_answer = (
                final_answer.split("#### ")[1].replace(",", "").replace("$", "")
            )
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer: Optional[str] = None
        if len(answer) == 0:
            # No reward if there is no answer.
            pass
        else:
            invalid_str: list[str] = ["", "."]
            # Find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_reward(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs
) -> float:
    """Compute binary reward for GSM8K math problems.

    Returns 1.0 if extracted answer matches ground truth exactly, else 0.0.
    Signature matches verl's custom reward function interface.
    """
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is None:
        return 0.0
    else:
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0


PATH_TO_REWARD_FUNCTION: Path = Path("/root/modal_train_qwen_05.py")
REWARD_FUNCTION_NAME: str = "compute_reward"


def generate_verl_cmd(n_nodes: int, arglist: list[str]) -> list[str]:
    """Build the verl trainer command for GRPO training on GSM8K with Qwen3-32B + Megatron."""

    cmd: list[str] = [
        "python",
        "-u",
        "-m",
        "verl.trainer.main_ppo",
        "--config-path=config",
        "--config-name=ppo_megatron_trainer.yaml",
        # ----- Data / algorithm -----
        "algorithm.adv_estimator=grpo",
        f"data.train_files={DATA_PATH / 'train.parquet'}",
        f"data.val_files={DATA_PATH / 'test.parquet'}",
        "data.prompt_key=prompt",
        "data.return_raw_chat=True",
        "data.max_prompt_length=768",
        "data.max_response_length=2048",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        # ----- Base model (HF view) -----
        f"actor_rollout_ref.model.path={HF_MODEL_DIR}",
        "actor_rollout_ref.model.use_fused_kernels=True",
        # ===== Resource Layout =====
        f"trainer.nnodes={n_nodes}",
        "trainer.n_gpus_per_node=8",
        # ===== Batching =====
        # EITHER actor_rollout_ref.rollout.n should be an integer divisor of: trainer.n_gpus_per_node * trainer.nnodes
        # OR actor_rollout_ref.rollout.n * data.train_batch_size should be evenly divisible by: trainer.n_gpus_per_node * trainer.nnodes
        "actor_rollout_ref.rollout.n=4",
        "data.train_batch_size=16",
        # ===== ACTOR (Megatron) =====
        # TODO vllm placement
        "actor_rollout_ref.actor.strategy=megatron",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        # "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16",
        # Dynamic batch size to balance MFU vs memory
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={2816 * 4}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        # Parallelism: 16 GPUs total → DP=2, TP=4, PP=2, CP=1, EP=1
        "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1",
        "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2",
        "actor_rollout_ref.actor.megatron.context_parallel_size=1",
        "actor_rollout_ref.actor.megatron.expert_model_parallel_size=1",
        "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1",
        # NOTE TO PEYTON - THIS IS USING BF16
        # enable this to use HF hceckpoints natively
        "actor_rollout_ref.actor.megatron.dtype=bfloat16",
        # "actor_rollout_ref.actor.megatron.use_mbridge=True",
        # Offload to keep 32B comfortable on 16×H100
        # removing these for now to see if it works without them
        "actor_rollout_ref.actor.megatron.param_offload=False",
        # i oomed so now i'm tryin ggrad offload
        "actor_rollout_ref.actor.megatron.grad_offload=False",
        # keep optimizer offload for now to see what memory footprint is like
        "actor_rollout_ref.actor.megatron.optimizer_offload=False",
        # Load from dist (mcore) checkpoint we converted earlier
        "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True",
        f"actor_rollout_ref.actor.megatron.dist_checkpointing_path={MCORE_MODEL_DIR}",
        # ----- Actor checkpoint: save *only* model, no optimizer -----
        'actor_rollout_ref.actor.checkpoint.save_contents=["model"]',
        # ===== ROLLOUT (vLLM) =====
        # TODO
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.data_parallel_size=1",
        "actor_rollout_ref.rollout.expert_parallel_size=1",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        # Make vLLM lighter so Megatron can breathe
        # TODO vllm placement
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5", # claire: edited .25 -> .5 to fix OOM
        # "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={2048 * 18}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        # Sampling params
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.7",
        "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        # ===== REF MODEL (Megatron) =====
        "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1",
        "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2",
        "actor_rollout_ref.ref.megatron.context_parallel_size=1",
        "actor_rollout_ref.ref.megatron.expert_model_parallel_size=1",
        "actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1",
        # removing these for now to see if it works without them
        # had to turn it back on to get it to work
        "actor_rollout_ref.ref.megatron.param_offload=True",
        "actor_rollout_ref.ref.megatron.use_dist_checkpointing=True",
        f"actor_rollout_ref.ref.megatron.dist_checkpointing_path={MCORE_MODEL_DIR}",
        "actor_rollout_ref.ref.megatron.dtype=bfloat16",
        # "actor_rollout_ref.ref.megatron.use_mbridge=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={2048 * 18}",  # 8192 + 768
        # ===== Algo / trainer / reward =====
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=verl_grpo_qwen3_32b",
        "trainer.experiment_name=qwen3_32b_megatron_gsm8k",
        "trainer.test_freq=5",
        f"trainer.default_local_dir={TRAINING_CHECKPOINT_DIR}",
        # disable periodic checkpoint saves for now (you can bump >0 later)
        "trainer.save_freq=-1",
        "trainer.resume_mode=auto",
        f"custom_reward_function.path={str(PATH_TO_REWARD_FUNCTION)}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
    ]

    if arglist:
        cmd.extend(arglist)

    return cmd


with image.imports():
    import ray
    from ray.job_submission import JobSubmissionClient


# Ray configuration
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def _init_ray(rank: int, main_node_addr: str, node_ip_addr: str, n_nodes: int):
    """Initialize Ray cluster across Modal containers.

    Rank 0 starts the head node, opens a tunnel to the Ray dashboard, and waits
    for all worker nodes to connect. Other ranks start as workers and connect
    to the head node address.
    """
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


async def run_training(n_nodes: int, arglist: list[str]):
    """Submit verl training job to Ray cluster and stream logs.

    Uses Ray's JobSubmissionClient to submit the training command and
    asynchronously tails logs until the job completes.
    """
    client = JobSubmissionClient()

    verl_cmd = generate_verl_cmd(n_nodes, arglist)
    job_id = client.submit_job(entrypoint=" ".join(verl_cmd))
    print(f"Job submitted with ID: {job_id}")

    async for line in client.tail_job_logs(job_id):
        print(line, end="", flush=True)


N_NODES = 2


@app.function(
    image=image,
    gpu="H100:8",
    volumes={MODELS_PATH: checkpoints_volume, DATA_PATH: data_volume},
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
    assert N_NODES > 1
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

            await run_training(N_NODES, list(arglist))
    else:
        # We have to keep the worker node alive until the training is complete. Once rank 0
        # finishes, all workers will be terminated.
        while True:
            time.sleep(10)


@app.function(
    image=image,
    gpu="H100:8",
    volumes={MODELS_PATH: checkpoints_volume, DATA_PATH: data_volume},
    secrets=[
        modal.Secret.from_name("wandb-secret-clairez"),
    ],
    timeout=24 * 60 * 60,
)
async def train_single_node(*arglist):
    assert N_NODES == 1
    checkpoints_volume.reload()
    data_volume.reload()

    _init_ray(0, "127.0.0.1", "127.0.0.1", 1)

    with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Dashboard URL: {tunnel.url}")
        await run_training(1, list(arglist))