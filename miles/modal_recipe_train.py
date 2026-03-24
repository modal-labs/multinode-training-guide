"""
Thin Modal launcher for recipe-driven multi-node Miles training.

Design:
- Bootstrap the Ray cluster once inside a clustered modal.Cls.
- Submit the actual Miles job from a rank-0 modal.method().
- Keep model/training settings in native Miles args files under miles/recipes/.
- Own only infrastructure-critical flags in Python:
  cluster size, GPUs per node, model path resolution, checkpoint path, and
  optional custom config transport.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import modal
import modal.experimental


here = pathlib.Path(__file__).parent.resolve()

APP_NAME = os.environ.get("MILES_RECIPE_APP_NAME", "miles-recipes")
MILES_IMAGE = os.environ.get("MILES_IMAGE", "radixark/miles:dev-202603231227")
CLUSTER_NODES = int(os.environ.get("MILES_N_NODES", "1"))
DEFAULT_GPU = os.environ.get("MILES_GPU", "H100:8")
LOCAL_MILES_PATH = os.environ.get("USE_LOCAL_MILES", "")

HF_CACHE_PATH = pathlib.Path("/root/.cache/huggingface")
DATA_PATH = pathlib.Path("/data")
CHECKPOINTS_PATH = pathlib.Path("/checkpoints")
REMOTE_RECIPES_DIR = pathlib.Path("/root/miles-recipes")
REMOTE_RUNTIME_DIR = pathlib.Path("/root/miles-modal-runtime")
REMOTE_MILES_DIR = pathlib.Path("/root/miles")
REMOTE_TRAIN_SCRIPT = REMOTE_MILES_DIR / "train.py"
REMOTE_BOOTSTRAP_SCRIPT = REMOTE_RUNTIME_DIR / "train_bootstrap.py"

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("miles-recipe-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(
    "miles-recipe-checkpoints", create_if_missing=True
)


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    model_id: str
    args_file: str
    recommended_nodes: int
    gpu: str


RECIPES = {
    "glm4-7-flash-lora": Recipe(
        name="glm4-7-flash-lora",
        description="GLM-4.7-Flash Miles LoRA validation recipe on multiple nodes.",
        model_id="zai-org/GLM-4.7-Flash",
        args_file="glm4-7-flash-lora.args",
        recommended_nodes=4,
        gpu="H100:8",
    ),
}


def _get_recipe(name: str) -> Recipe:
    if name not in RECIPES:
        available = ", ".join(sorted(RECIPES))
        raise ValueError(f"Unknown recipe: {name}. Available recipes: {available}")
    return RECIPES[name]


def _parse_gpus_per_node(gpu: str) -> int:
    try:
        return int(gpu.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"GPU spec must include a per-node count like 'H100:8'; got {gpu!r}"
        ) from exc


def _clean_arg_text(arg_text: str) -> str:
    lines: list[str] = []
    for raw_line in arg_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _parse_arg_text(arg_text: str) -> list[str]:
    cleaned = _clean_arg_text(arg_text)
    return shlex.split(cleaned) if cleaned else []


def _load_recipe_text(recipe: Recipe, remote: bool = False) -> str:
    base_dir = REMOTE_RECIPES_DIR if remote else here / "recipes"
    return (base_dir / recipe.args_file).read_text()


def _build_enforced_args(
    *,
    model_path: str,
    cluster_nodes: int,
    gpus_per_node: int,
    checkpoint_dir: pathlib.Path,
    custom_config_path: Optional[str],
    wandb_key: Optional[str],
) -> list[str]:
    args = [
        "--train-backend",
        "megatron",
        "--hf-checkpoint",
        model_path,
        "--save",
        checkpoint_dir.as_posix(),
        "--actor-num-nodes",
        str(cluster_nodes),
        "--actor-num-gpus-per-node",
        str(gpus_per_node),
        "--num-gpus-per-node",
        str(gpus_per_node),
        "--colocate",
    ]
    if custom_config_path:
        args.extend(["--custom-config-path", custom_config_path])
    if wandb_key:
        args.extend(["--use-wandb", "--wandb-key", wandb_key])
    return args


def _build_miles_argv(
    recipe: Recipe,
    *,
    model_path: str,
    cluster_nodes: int,
    gpus_per_node: int,
    checkpoint_dir: pathlib.Path,
    extra_args_text: str,
    custom_config_path: Optional[str],
    wandb_key: Optional[str],
    remote_recipe: bool,
) -> list[str]:
    recipe_args = _parse_arg_text(_load_recipe_text(recipe, remote=remote_recipe))
    extra_args = _parse_arg_text(extra_args_text)
    enforced_args = _build_enforced_args(
        model_path=model_path,
        cluster_nodes=cluster_nodes,
        gpus_per_node=gpus_per_node,
        checkpoint_dir=checkpoint_dir,
        custom_config_path=custom_config_path,
        wandb_key=wandb_key,
    )
    return [
        "python3",
        REMOTE_BOOTSTRAP_SCRIPT.as_posix(),
        REMOTE_TRAIN_SCRIPT.as_posix(),
        *recipe_args,
        *extra_args,
        *enforced_args,
    ]


def _read_optional_file(path_str: str) -> str:
    if not path_str:
        return ""
    return pathlib.Path(path_str).read_text()


def _get_last_flag_value(args: list[str], flag: str) -> Optional[str]:
    value = None
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            value = args[idx + 1]
    return value


def _get_int_flag(
    args: list[str], flag: str, default: Optional[int] = None
) -> Optional[int]:
    value = _get_last_flag_value(args, flag)
    if value is None:
        return default
    return int(value)


def _validate_batch_shape(
    *,
    recipe: Recipe,
    recipe_and_extra_args: list[str],
    cluster_nodes: int,
    gpus_per_node: int,
) -> None:
    global_batch_size = _get_int_flag(recipe_and_extra_args, "--global-batch-size")
    if global_batch_size is None:
        return

    micro_batch_size = _get_int_flag(
        recipe_and_extra_args, "--micro-batch-size", default=1
    )
    tensor_parallel_size = _get_int_flag(
        recipe_and_extra_args, "--tensor-model-parallel-size", default=1
    )
    pipeline_parallel_size = _get_int_flag(
        recipe_and_extra_args, "--pipeline-model-parallel-size", default=1
    )
    context_parallel_size = _get_int_flag(
        recipe_and_extra_args, "--context-parallel-size", default=1
    )

    total_gpus = cluster_nodes * gpus_per_node
    model_parallel_size = (
        tensor_parallel_size * pipeline_parallel_size * context_parallel_size
    )
    if total_gpus % model_parallel_size != 0:
        raise ValueError(
            f"Recipe {recipe.name} uses total_gpus={total_gpus}, but "
            f"tensor/pipeline/context parallel sizes multiply to "
            f"{model_parallel_size}, which does not divide total_gpus."
        )

    data_parallel_size = total_gpus // model_parallel_size
    required_multiple = micro_batch_size * data_parallel_size
    if global_batch_size % required_multiple != 0:
        raise ValueError(
            f"Recipe {recipe.name} resolves to global_batch_size={global_batch_size}, "
            f"micro_batch_size={micro_batch_size}, and data_parallel_size={data_parallel_size}. "
            f"Megatron requires global_batch_size to be divisible by "
            f"micro_batch_size * data_parallel_size = {required_multiple}. "
            "Adjust --global-batch-size or --micro-batch-size in the recipe/extra args."
        )


def _build_runtime_env(master_addr: str, wandb_key: Optional[str]) -> dict:
    env_vars = {
        "MASTER_ADDR": master_addr,
        "MILES_HOST_IP": master_addr,
        "no_proxy": master_addr,
        "PYTHONPATH": f"{REMOTE_RUNTIME_DIR.as_posix()}:/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    if wandb_key:
        env_vars["WANDB_API_KEY"] = wandb_key
    return {"env_vars": env_vars}


def _print_recipe_table():
    print("Available recipes:")
    for recipe in sorted(RECIPES.values(), key=lambda item: item.name):
        print(
            f"  - {recipe.name}: {recipe.description} "
            f"(model={recipe.model_id}, nodes={recipe.recommended_nodes}, gpu={recipe.gpu})"
        )


image = (
    modal.Image.from_registry(MILES_IMAGE)
    .entrypoint([])
    .add_local_dir(here / "runtime", remote_path=REMOTE_RUNTIME_DIR.as_posix(), copy=True)
    .add_local_dir(here / "recipes", remote_path=REMOTE_RECIPES_DIR.as_posix(), copy=True)
)

if LOCAL_MILES_PATH:
    image = image.add_local_dir(
        LOCAL_MILES_PATH,
        remote_path=REMOTE_MILES_DIR.as_posix(),
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
    ).run_commands(f"uv pip install --system -e {REMOTE_MILES_DIR}")


with image.imports():
    import ray
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from ray.job_submission import JobSubmissionClient


app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={
        HF_CACHE_PATH.as_posix(): hf_cache_volume,
        DATA_PATH.as_posix(): data_volume,
        CHECKPOINTS_PATH.as_posix(): checkpoints_volume,
    },
    timeout=24 * 60 * 60,
    scaledown_window=60 * 60,
    retries=2,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=CLUSTER_NODES, rdma=True)
class MilesCluster:
    @modal.enter()
    def bootstrap_ray(self):
        hf_cache_volume.reload()
        data_volume.reload()
        checkpoints_volume.reload()
        self.rank = None
        self.node_ips: list[str] = []
        self.main_addr = None
        self.node_addr = None
        self.client = None
        self._ray_ready = False

    def _ensure_ray_started(self):
        if self._ray_ready:
            return

        cluster_info = modal.experimental.get_cluster_info()
        self.rank = cluster_info.rank
        if cluster_info.container_ipv4_ips:
            self.node_ips = cluster_info.container_ipv4_ips
        elif CLUSTER_NODES == 1:
            self.node_ips = ["127.0.0.1"]
        else:
            raise RuntimeError(
                "Modal did not provide container IPv4s for a multi-node cluster."
            )

        self.main_addr = self.node_ips[0]
        self.node_addr = self.node_ips[min(self.rank, len(self.node_ips) - 1)]

        if self.rank == 0:
            print(f"Starting Ray head at {self.node_addr}")
            subprocess.Popen(
                [
                    "ray",
                    "start",
                    "--head",
                    f"--node-ip-address={self.node_addr}",
                    "--dashboard-host=0.0.0.0",
                    "--disable-usage-stats",
                ]
            )

            for _ in range(30):
                try:
                    ray.init(address="auto")
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("Failed to connect to the Ray head node")

            for _ in range(60):
                alive_nodes = [node for node in ray.nodes() if node["Alive"]]
                print(f"Alive nodes: {len(alive_nodes)}/{len(self.node_ips)}")
                if len(alive_nodes) == len(self.node_ips):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Not all Ray worker nodes connected")

            self.client = JobSubmissionClient(f"http://127.0.0.1:{RAY_DASHBOARD_PORT}")
            print("Ray cluster is ready.")
        else:
            print(f"Starting Ray worker at {self.node_addr}, head={self.main_addr}")
            subprocess.Popen(
                [
                    "ray",
                    "start",
                    f"--node-ip-address={self.node_addr}",
                    "--address",
                    f"{self.main_addr}:{RAY_PORT}",
                    "--disable-usage-stats",
                ]
            )
        self._ray_ready = True

    @modal.method()
    async def submit_training(
        self,
        recipe_name: str,
        *,
        gpus_per_node: int,
        extra_args_text: str = "",
        custom_config_yaml: str = "",
        wandb_key: str = "",
    ) -> dict:
        self._ensure_ray_started()

        if self.rank != 0:
            while True:
                time.sleep(10)

        recipe = _get_recipe(recipe_name)
        cluster_nodes = len(self.node_ips)

        try:
            model_path = snapshot_download(repo_id=recipe.model_id, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Model {recipe.model_id} is not present in the shared HF cache. "
                f"Run `modal run miles/modal_recipe_train.py::download_model --recipe {recipe.name}` first."
            ) from exc

        run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = CHECKPOINTS_PATH / recipe.name / run_id
        custom_config_path = None
        if custom_config_yaml:
            custom_config_path = f"/tmp/{recipe.name}-{run_id}-overrides.yaml"
            pathlib.Path(custom_config_path).write_text(custom_config_yaml)

        argv = _build_miles_argv(
            recipe,
            model_path=model_path,
            cluster_nodes=cluster_nodes,
            gpus_per_node=gpus_per_node,
            checkpoint_dir=checkpoint_dir,
            extra_args_text=extra_args_text,
            custom_config_path=custom_config_path,
            wandb_key=wandb_key or None,
            remote_recipe=True,
        )
        entrypoint = shlex.join(argv)
        runtime_env = _build_runtime_env(self.main_addr, wandb_key or None)

        print(f"Recipe: {recipe.name}")
        print(f"Model: {recipe.model_id}")
        print(f"Nodes: {cluster_nodes}")
        print(f"GPUs per node: {gpus_per_node}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Entrypoint: {entrypoint}")

        async with modal.forward.aio(RAY_DASHBOARD_PORT) as tunnel:
            print(f"Dashboard URL: {tunnel.url}")
            job_id = self.client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
            print(f"Submitted Ray job: {job_id}")

            async for line in self.client.tail_job_logs(job_id):
                print(line, end="", flush=True)

        status = self.client.get_job_status(job_id).value
        await checkpoints_volume.commit.aio()
        print(f"\nFinal status: {status}")
        return {
            "job_id": job_id,
            "status": status,
            "recipe": recipe.name,
            "checkpoint_dir": checkpoint_dir.as_posix(),
        }


@app.function(
    image=image,
    volumes={HF_CACHE_PATH.as_posix(): hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=24 * 60 * 60,
)
def download_model(
    recipe: str = "glm4-7-flash-lora",
    revision: Optional[str] = None,
    model_id: Optional[str] = None,
):
    resolved_model_id = model_id or _get_recipe(recipe).model_id
    hf_cache_volume.reload()
    path = snapshot_download(
        repo_id=resolved_model_id,
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded {resolved_model_id} to {path}")
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={DATA_PATH.as_posix(): data_volume},
    timeout=24 * 60 * 60,
)
def prepare_dataset(
    hf_dataset: str = "zhuzilin/gsm8k",
    data_folder: str = "gsm8k",
):
    data_volume.reload()
    dataset = load_dataset(hf_dataset)
    output_dir = DATA_PATH / data_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset["train"].to_parquet((output_dir / "train.parquet").as_posix())
    dataset["test"].to_parquet((output_dir / "test.parquet").as_posix())
    data_volume.commit()
    print(f"Prepared dataset {hf_dataset} under {output_dir}")


@app.local_entrypoint()
def main(
    recipe: str = "glm4-7-flash-lora",
    gpu: str = "",
    extra_args: str = "",
    extra_args_file: str = "",
    custom_config: str = "",
    list_recipes: bool = False,
    dry_run: bool = False,
    allow_cluster_mismatch: bool = False,
):
    if list_recipes:
        _print_recipe_table()
        return

    selected_recipe = _get_recipe(recipe)
    selected_gpu = gpu or selected_recipe.gpu
    gpus_per_node = _parse_gpus_per_node(selected_gpu)

    if (
        not allow_cluster_mismatch
        and CLUSTER_NODES != selected_recipe.recommended_nodes
    ):
        raise ValueError(
            f"Recipe {selected_recipe.name} expects MILES_N_NODES={selected_recipe.recommended_nodes}, "
            f"but this process was started with MILES_N_NODES={CLUSTER_NODES}. "
            f"Rerun with the recommended value or pass --allow-cluster-mismatch."
        )

    merged_extra_args = "\n".join(
        part for part in [extra_args, _read_optional_file(extra_args_file)] if part
    )
    custom_config_yaml = _read_optional_file(custom_config)
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    recipe_and_extra_args = _parse_arg_text(_load_recipe_text(selected_recipe))
    recipe_and_extra_args.extend(_parse_arg_text(merged_extra_args))
    _validate_batch_shape(
        recipe=selected_recipe,
        recipe_and_extra_args=recipe_and_extra_args,
        cluster_nodes=CLUSTER_NODES,
        gpus_per_node=gpus_per_node,
    )
    checkpoint_dir = CHECKPOINTS_PATH / selected_recipe.name / "DRY_RUN"

    if dry_run:
        argv = _build_miles_argv(
            selected_recipe,
            model_path="$MODEL_PATH",
            cluster_nodes=CLUSTER_NODES,
            gpus_per_node=gpus_per_node,
            checkpoint_dir=checkpoint_dir,
            extra_args_text=merged_extra_args,
            custom_config_path="/tmp/custom-config.yaml" if custom_config_yaml else None,
            wandb_key="$WANDB_API_KEY" if wandb_key else None,
            remote_recipe=False,
        )
        print(f"Recipe: {selected_recipe.name}")
        print(f"Model: {selected_recipe.model_id}")
        print(f"Cluster nodes: {CLUSTER_NODES}")
        print(f"GPU: {selected_gpu}")
        print(shlex.join(argv))
        return

    print(f"Recipe: {selected_recipe.name}")
    print(f"Model: {selected_recipe.model_id}")
    print(f"Cluster nodes: {CLUSTER_NODES}")
    print(f"GPU: {selected_gpu}")

    cluster = MilesCluster.with_options(gpu=selected_gpu)()
    result = cluster.submit_training.remote(
        recipe_name=selected_recipe.name,
        gpus_per_node=gpus_per_node,
        extra_args_text=merged_extra_args,
        custom_config_yaml=custom_config_yaml,
        wandb_key=wandb_key,
    )
    print(result)
