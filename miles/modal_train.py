"""Launch Miles training jobs on Modal.

This module defines the clustered `MilesCluster` launcher, helper functions for
model download and dataset preparation, and a local CLI entrypoint for
submitting runs.

It supports two execution modes:
  - deployed: submit runs to a deployed `MilesCluster` with a fixed compute shape
  - ephemeral: launch a one-off app directly from the local entrypoint

The Python wrapper owns only Modal and Ray orchestration plus a small set of
infrastructure-critical flags. Model and training arguments live in `recipes/`.
"""

import datetime as dt
import os
import pathlib
import re
import shlex
import subprocess
import time
from typing import Optional

import modal
import modal.experimental

from recipes import (
    Recipe,
    format_recipe_table,
    get_optional_recipe,
    load_recipe_text,
    merge_arg_texts,
    parse_arg_text,
    read_arg_file,
)

here = pathlib.Path(__file__).parent.resolve()

APP_NAME = os.environ.get("MILES_APP_NAME", "miles-modal")
MILES_IMAGE = os.environ.get("MILES_IMAGE", "radixark/miles:dev-202603231227")
CLUSTER_NODES = int(os.environ.get("MILES_N_NODES", "1"))
DEFAULT_GPU = os.environ.get("MILES_GPU", "H100:8")
LOCAL_MILES_PATH = os.environ.get("USE_LOCAL_MILES", "")

HF_CACHE_PATH = pathlib.Path("/root/.cache/huggingface")
DATA_PATH = pathlib.Path("/data")
CHECKPOINTS_PATH = pathlib.Path("/checkpoints")
REMOTE_RECIPES_DIR = pathlib.Path("/root/recipes")
REMOTE_PATCH_DIR = pathlib.Path("/root/miles-modal-patches")
REMOTE_MILES_DIR = pathlib.Path("/root/miles")
REMOTE_TRAIN_SCRIPT = REMOTE_MILES_DIR / "train.py"

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("miles-example-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(
    "miles-example-checkpoints", create_if_missing=True
)


def _parse_gpus_per_node(gpu: str) -> int:
    try:
        return int(gpu.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"GPU spec must include a per-node count like 'H100:8'; got {gpu!r}"
        ) from exc


def _resolve_model_id(recipe: Recipe | None, model_id: str) -> str:
    if model_id:
        return model_id
    if recipe:
        return recipe.model_id
    raise ValueError("Pass --recipe or --model-id.")


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return sanitized or "run"


def _resolve_run_label(recipe: Recipe | None, *, model_id: str, run_name: str) -> str:
    if run_name:
        return _sanitize_path_component(run_name)
    if recipe:
        return recipe.name
    return _sanitize_path_component(model_id)


def _resolve_base_args_text(
    recipe: Recipe | None,
    *,
    args_text: str,
    recipes_dir: pathlib.Path,
) -> str:
    parts: list[str] = []
    if recipe:
        parts.append(load_recipe_text(recipe, base_dir=recipes_dir))
    if args_text:
        parts.append(args_text)
    return merge_arg_texts(*parts)


def _build_enforced_args(
    *,
    model_path: str,
    actor_nodes: int,
    gpus_per_node: int,
    checkpoint_dir: pathlib.Path,
    custom_config_path: Optional[str],
    wandb_key: Optional[str],
    colocate: bool,
    rollout_num_gpus: Optional[int],
) -> list[str]:
    if actor_nodes < 1:
        raise ValueError(f"actor_nodes must be >= 1, got {actor_nodes}")

    args = [
        "--train-backend",
        "megatron",
        "--hf-checkpoint",
        model_path,
        "--ref-load",
        model_path,
        "--save",
        checkpoint_dir.as_posix(),
        "--actor-num-nodes",
        str(actor_nodes),
        "--actor-num-gpus-per-node",
        str(gpus_per_node),
        "--num-gpus-per-node",
        str(gpus_per_node),
    ]
    if colocate:
        args.append("--colocate")
    else:
        if rollout_num_gpus is None or rollout_num_gpus < 1:
            raise ValueError(
                "rollout_num_gpus must be >= 1 when launching non-colocated rollout."
            )
        args.extend(["--rollout-num-gpus", str(rollout_num_gpus)])
    if custom_config_path:
        args.extend(["--custom-config-path", custom_config_path])
    if wandb_key:
        args.extend(["--use-wandb", "--wandb-key", wandb_key])
    return args


def _build_miles_argv(
    *,
    base_args_text: str,
    model_path: str,
    actor_nodes: int,
    gpus_per_node: int,
    checkpoint_dir: pathlib.Path,
    extra_args_text: str,
    custom_config_path: Optional[str],
    wandb_key: Optional[str],
    colocate: bool,
    rollout_num_gpus: Optional[int],
) -> list[str]:
    base_args = parse_arg_text(base_args_text)
    extra_args = parse_arg_text(extra_args_text)
    enforced_args = _build_enforced_args(
        model_path=model_path,
        actor_nodes=actor_nodes,
        gpus_per_node=gpus_per_node,
        checkpoint_dir=checkpoint_dir,
        custom_config_path=custom_config_path,
        wandb_key=wandb_key,
        colocate=colocate,
        rollout_num_gpus=rollout_num_gpus,
    )
    return [
        "python3",
        REMOTE_TRAIN_SCRIPT.as_posix(),
        *base_args,
        *extra_args,
        *enforced_args,
    ]


def _resolve_actor_nodes(
    cluster_nodes: int, *, colocate: bool, actor_nodes: int
) -> int:
    if colocate:
        return cluster_nodes
    if actor_nodes > 0:
        return actor_nodes
    if cluster_nodes < 2:
        raise ValueError(
            "Non-colocated rollout needs spare cluster capacity. "
            "Set MILES_N_NODES>=2 or pass --colocate."
        )
    return cluster_nodes - 1


def _resolve_rollout_num_gpus(
    cluster_nodes: int,
    *,
    actor_nodes: int,
    gpus_per_node: int,
    colocate: bool,
    rollout_num_gpus: int,
) -> Optional[int]:
    if colocate:
        return None
    if rollout_num_gpus > 0:
        return rollout_num_gpus
    spare_gpus = (cluster_nodes - actor_nodes) * gpus_per_node
    if spare_gpus < 1:
        raise ValueError(
            "Non-colocated rollout needs spare GPUs after reserving actor nodes. "
            f"cluster_nodes={cluster_nodes}, actor_nodes={actor_nodes}, "
            f"gpus_per_node={gpus_per_node}"
        )
    return spare_gpus


def _build_runtime_env(master_addr: str, wandb_key: Optional[str]) -> dict:
    env_vars = {
        "MASTER_ADDR": master_addr,
        "MILES_HOST_IP": master_addr,
        "no_proxy": master_addr,
        "PYTHONPATH": f"{REMOTE_PATCH_DIR.as_posix()}:/root/Megatron-LM",
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
    for line in format_recipe_table():
        print(line)


image = (
    modal.Image.from_registry(MILES_IMAGE)
    .entrypoint([])
    .add_local_dir(here / "recipes", remote_path=REMOTE_RECIPES_DIR.as_posix())
    .add_local_dir(
        here / "modal_patches",
        remote_path=REMOTE_PATCH_DIR.as_posix(),
    )
)

if LOCAL_MILES_PATH:
    image = image.add_local_dir(
        LOCAL_MILES_PATH,
        remote_path=REMOTE_MILES_DIR.as_posix(),
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
    ).run_commands(f"pip install -e {REMOTE_MILES_DIR} --no-deps")


with image.imports():
    import ray
    from huggingface_hub import snapshot_download
    from ray.job_submission import JobSubmissionClient


app = modal.App(APP_NAME)


# ---- Training Cluster Cls ---- #


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
        self.node_ips = []
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
            # Modal may omit container IPv4s for size-1 clustered functions.
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
        recipe_name: str = "",
        *,
        model_id: str = "",
        base_args_text: str = "",
        gpus_per_node: int,
        extra_args_text: str = "",
        custom_config_yaml: str = "",
        wandb_key: str = "",
        run_name: str = "",
        actor_nodes: int | None = None,
        colocate: bool = True,
        rollout_num_gpus: int | None = None,
    ) -> dict:
        self._ensure_ray_started()

        if self.rank != 0:
            while True:
                time.sleep(10)

        recipe = get_optional_recipe(recipe_name)
        resolved_model_id = _resolve_model_id(recipe, model_id)
        resolved_base_args_text = _resolve_base_args_text(
            recipe,
            args_text=base_args_text,
            recipes_dir=REMOTE_RECIPES_DIR,
        )
        if not resolved_base_args_text:
            raise ValueError(
                "No training args were provided. Choose a recipe or pass --args/--args-file."
            )
        run_label = _resolve_run_label(
            recipe,
            model_id=resolved_model_id,
            run_name=run_name,
        )

        try:
            model_path = snapshot_download(
                repo_id=resolved_model_id, local_files_only=True
            )
        except Exception as exc:
            recipe_hint = (
                f"Run `modal run miles/modal_train.py::download_model --recipe {recipe.name}` first."
                if recipe
                else f"Run `modal run miles/modal_train.py::download_model --model-id {resolved_model_id}` first."
            )
            raise RuntimeError(
                f"Model {resolved_model_id} is not present in the shared HF cache. "
                f"{recipe_hint}"
            ) from exc

        run_id = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = CHECKPOINTS_PATH / run_label / run_id
        custom_config_path = None
        if custom_config_yaml:
            custom_config_path = f"/tmp/{run_label}-{run_id}-overrides.yaml"
            pathlib.Path(custom_config_path).write_text(custom_config_yaml)

        resolved_actor_nodes = actor_nodes if actor_nodes is not None else CLUSTER_NODES
        resolved_rollout_num_gpus = rollout_num_gpus
        argv = _build_miles_argv(
            base_args_text=resolved_base_args_text,
            model_path=model_path,
            actor_nodes=resolved_actor_nodes,
            gpus_per_node=gpus_per_node,
            checkpoint_dir=checkpoint_dir,
            extra_args_text=extra_args_text,
            custom_config_path=custom_config_path,
            wandb_key=wandb_key or None,
            colocate=colocate,
            rollout_num_gpus=resolved_rollout_num_gpus,
        )
        entrypoint = shlex.join(argv)
        runtime_env = _build_runtime_env(self.main_addr, wandb_key or None)

        print(f"Recipe: {recipe.name if recipe else '<none>'}")
        print(f"Model: {resolved_model_id}")
        print(f"Run label: {run_label}")
        print(f"Cluster nodes: {CLUSTER_NODES}")
        print(f"Actor nodes: {resolved_actor_nodes}")
        print(f"Colocate: {colocate}")
        if not colocate:
            print(f"Rollout GPUs: {resolved_rollout_num_gpus}")
        print(f"GPUs per node: {gpus_per_node}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Entrypoint: {entrypoint}")

        with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
            print(f"Dashboard URL: {tunnel.url}")
            job_id = self.client.submit_job(
                entrypoint=entrypoint, runtime_env=runtime_env
            )
            print(f"Submitted Ray job: {job_id}")

            async for line in self.client.tail_job_logs(job_id):
                print(line, end="", flush=True)

        status = self.client.get_job_status(job_id).value
        checkpoints_volume.commit()
        print(f"\nFinal status: {status}")
        return {
            "job_id": job_id,
            "status": status,
            "recipe": recipe.name if recipe else None,
            "model_id": resolved_model_id,
            "run_name": run_label,
            "checkpoint_dir": checkpoint_dir.as_posix(),
        }


# ---- Model Download Utility ---- #


@app.function(
    image=image,
    volumes={HF_CACHE_PATH.as_posix(): hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=24 * 60 * 60,
)
def download_model(
    recipe: str = "qwen3-30b-a3b-lora",
    revision: Optional[str] = None,
    model_id: Optional[str] = None,
):
    from huggingface_hub import snapshot_download

    selected_recipe = get_optional_recipe(recipe)
    resolved_model_id = model_id or (
        selected_recipe.model_id if selected_recipe else ""
    )
    if not resolved_model_id:
        raise ValueError("Pass --recipe or --model-id.")
    hf_cache_volume.reload()
    path = snapshot_download(
        repo_id=resolved_model_id,
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded {resolved_model_id} to {path}")
    hf_cache_volume.commit()


# ---- Dataset Processing Utility ---- #


@app.function(
    image=image,
    volumes={DATA_PATH.as_posix(): data_volume},
    timeout=24 * 60 * 60,
)
def prepare_dataset(
    hf_dataset: str = "zhuzilin/gsm8k",
    data_folder: str = "gsm8k",
    train_limit: int = 0,
    test_limit: int = 0,
):
    from datasets import load_dataset

    data_volume.reload()
    dataset = load_dataset(hf_dataset)
    train_split = dataset["train"]
    test_split = dataset["test"]
    if train_limit > 0:
        train_split = train_split.select(range(min(train_limit, len(train_split))))
    if test_limit > 0:
        test_split = test_split.select(range(min(test_limit, len(test_split))))
    output_dir = DATA_PATH / data_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    train_split.to_parquet((output_dir / "train.parquet").as_posix())
    test_split.to_parquet((output_dir / "test.parquet").as_posix())
    data_volume.commit()
    print(
        f"Prepared dataset {hf_dataset} under {output_dir} "
        f"(train_rows={len(train_split)}, test_rows={len(test_split)})"
    )


# ---- Local Entrypoint ---- #


@app.local_entrypoint()
def main(
    recipe: str = "",
    model_id: str = "",
    gpu: str = "",
    args: str = "",
    args_file: str = "",
    extra_args: str = "",
    extra_args_file: str = "",
    custom_config: str = "",
    list_recipes: bool = False,
    dry_run: bool = False,
    allow_cluster_mismatch: bool = False,
    run_name: str = "",
    colocate: bool = True,
    actor_nodes: int = 0,
    rollout_num_gpus: int = 0,
):
    if list_recipes:
        _print_recipe_table()
        return

    selected_recipe = get_optional_recipe(recipe)
    resolved_model_id = _resolve_model_id(selected_recipe, model_id)
    selected_gpu = gpu or (selected_recipe.gpu if selected_recipe else DEFAULT_GPU)
    gpus_per_node = _parse_gpus_per_node(selected_gpu)
    resolved_actor_nodes = _resolve_actor_nodes(
        CLUSTER_NODES,
        colocate=colocate,
        actor_nodes=actor_nodes,
    )
    resolved_rollout_num_gpus = _resolve_rollout_num_gpus(
        CLUSTER_NODES,
        actor_nodes=resolved_actor_nodes,
        gpus_per_node=gpus_per_node,
        colocate=colocate,
        rollout_num_gpus=rollout_num_gpus,
    )

    if (
        selected_recipe
        and not allow_cluster_mismatch
        and CLUSTER_NODES != selected_recipe.recommended_nodes
    ):
        raise ValueError(
            f"Recipe {selected_recipe.name} expects MILES_N_NODES={selected_recipe.recommended_nodes}, "
            f"but this process was started with MILES_N_NODES={CLUSTER_NODES}. "
            f"Rerun with the recommended value or pass --allow-cluster-mismatch."
        )

    base_args_text = merge_arg_texts(args, read_arg_file(args_file))
    resolved_base_args_text = _resolve_base_args_text(
        selected_recipe,
        args_text=base_args_text,
        recipes_dir=here / "recipes",
    )
    if not resolved_base_args_text:
        raise ValueError(
            "No training args were provided. Choose a recipe or pass --args/--args-file."
        )
    merged_extra_args = merge_arg_texts(extra_args, read_arg_file(extra_args_file))
    custom_config_yaml = read_arg_file(custom_config)
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    resolved_run_label = _resolve_run_label(
        selected_recipe,
        model_id=resolved_model_id,
        run_name=run_name,
    )
    checkpoint_dir = CHECKPOINTS_PATH / resolved_run_label / "DRY_RUN"

    if dry_run:
        argv = _build_miles_argv(
            base_args_text=resolved_base_args_text,
            model_path="$MODEL_PATH",
            actor_nodes=resolved_actor_nodes,
            gpus_per_node=gpus_per_node,
            checkpoint_dir=checkpoint_dir,
            extra_args_text=merged_extra_args,
            custom_config_path="/tmp/custom-config.yaml"
            if custom_config_yaml
            else None,
            wandb_key="$WANDB_API_KEY" if wandb_key else None,
            colocate=colocate,
            rollout_num_gpus=resolved_rollout_num_gpus,
        )
        print(f"Recipe: {selected_recipe.name if selected_recipe else '<none>'}")
        print(f"Model: {resolved_model_id}")
        print(f"Run label: {resolved_run_label}")
        print(f"Cluster nodes: {CLUSTER_NODES}")
        print(f"Actor nodes: {resolved_actor_nodes}")
        print(f"Colocate: {colocate}")
        if not colocate:
            print(f"Rollout GPUs: {resolved_rollout_num_gpus}")
        print(f"GPU: {selected_gpu}")
        print(shlex.join(argv))
        return

    print(f"Recipe: {selected_recipe.name if selected_recipe else '<none>'}")
    print(f"Model: {resolved_model_id}")
    print(f"Run label: {resolved_run_label}")
    print(f"Cluster nodes: {CLUSTER_NODES}")
    print(f"Actor nodes: {resolved_actor_nodes}")
    print(f"Colocate: {colocate}")
    if not colocate:
        print(f"Rollout GPUs: {resolved_rollout_num_gpus}")
    print(f"GPU: {selected_gpu}")

    try:
        cluster_cls = modal.Cls.from_name(APP_NAME, "MilesCluster")
        print(f"Using deployed Modal class: {APP_NAME}.MilesCluster")
    except modal.exception.NotFoundError:
        cluster_cls = MilesCluster
        print(
            f"No deployed Modal class found for {APP_NAME}.MilesCluster; using an ephemeral app."
        )

    cluster = cluster_cls.with_options(gpu=selected_gpu)()
    result = cluster.submit_training.remote(
        recipe_name=selected_recipe.name if selected_recipe else "",
        model_id=resolved_model_id,
        base_args_text=base_args_text,
        gpus_per_node=gpus_per_node,
        extra_args_text=merged_extra_args,
        custom_config_yaml=custom_config_yaml,
        wandb_key=wandb_key,
        run_name=resolved_run_label,
        actor_nodes=resolved_actor_nodes,
        colocate=colocate,
        rollout_num_gpus=resolved_rollout_num_gpus,
    )
    print(result)
