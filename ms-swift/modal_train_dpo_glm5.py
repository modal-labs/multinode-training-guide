"""
GLM-5 DPO training via ms-swift Megatron.
"""

import json
import os
import time
from typing import Optional

import modal
import modal.experimental

HF_MODEL = "zai-org/GLM-5"
MODEL_NAME = "GLM-5"
DEFAULT_PREFERENCE_DATASET = "argilla/distilabel-math-preference-dpo"

# Pin exact upstream revisions so Modal image caching cannot silently reuse an
# older ms-swift main branch when validating recent GLM-5 support.
MSSWIFT_PR_8085_COMMIT = "4f4a0640be350eb1d1cb68d2a92b880f08930098"
TRANSFORMERS_VERSION = "5.2.0"
# peft 0.18.1 imports HybridCache, which transformers 5.2.0 does not export.
PEFT_VERSION = "0.18.0"
MEGATRON_LM_COMMIT = "f8becec65f47982c80c3d397bef7c3fba65f9efc"
FAST_HADAMARD_TRANSFORM_COMMIT = "e7706faf8d1c3b9f241e36860640ad1dac644ede"

DEFAULT_MAX_EPOCHS = 1
DEFAULT_MAX_LENGTH = 2048
DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 16
DEFAULT_BETA = 0.1
DEFAULT_RECOMPUTE_GRANULARITY = "none"
DEFAULT_PADDING_FREE = False
# ms-swift/GLM-5 currently forwards None into Megatron's DSA indexer-loss path,
# which then crashes on the first train step when it multiplies by the coeff.
DEFAULT_DSA_INDEXER_LOSS_COEFF = 0.0
# Keep GLM-5 on MCore checkpoint saves only for now; the HF/safetensors export
# path currently crashes in swift.megatron.model.gpt_bridge.GPTBridge.
DEFAULT_SAVE_SAFETENSORS = False

TP_SIZE = 2
PP_SIZE = 4
EP_SIZE = 4
CP_SIZE = 1

WANDB_PROJECT = "glm-5-dpo"

app = modal.App("example-msswift-glm_5-dpo")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name(
    "example-msswift-glm-5-dpo-data",
    create_if_missing=True,
    version=2,
)
checkpoints_volume = modal.Volume.from_name(
    "example-msswift-glm-5-dpo-checkpoints",
    create_if_missing=True,
    version=2,
)

HF_CACHE = "/root/.cache/huggingface"
DATA_DIR = "/data"
CHECKPOINTS_DIR = "/checkpoints"

# Default to 4 nodes (32x B200), matching TP=2 x EP=4 x PP=4 for GLM-5.
N_NODES = int(os.environ.get("N_NODES", "4"))


download_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    # Keep the prep/download image minimal. It only needs HF Hub + datasets, and
    # pinning a modern hub release avoids conflicts with recent GLM-5 support.
    "huggingface_hub>=1.3.0,<2.0",
    "datasets>=2.14.0",
    "safetensors==0.4.5",
).env(
    {
        # Use the Hub's supported high-performance Xet mode for very large
        # checkpoint shards; GLM-5 downloads are otherwise painfully slow.
        "HF_XET_HIGH_PERFORMANCE": "1",
    }
)

msswift_glm5_image = (
    modal.Image.from_registry(
        "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-modelscope1.33.0-swift3.12.5"
    )
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc-dev",
        "libnl-route-3-200",
    )
    .run_commands(
        "pip uninstall -y transformers ms-swift swift 2>/dev/null; true",
    )
    .uv_pip_install(
        "huggingface_hub>=1.3.0,<2.0",
        f"megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@{MEGATRON_LM_COMMIT}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"peft=={PEFT_VERSION}",
        f"ms-swift @ git+https://github.com/modelscope/ms-swift.git@{MSSWIFT_PR_8085_COMMIT}",
        "einops==0.8.2",
        "wandb==0.19.1",
        "jieba",
    )
    .uv_pip_install(
        # fast-hadamard-transform imports torch in setup.py but does not declare
        # it as a build dependency, so build isolation fails unless we reuse the
        # already-installed torch from the base image. Install from git because
        # the PyPI sdist is missing the C++ source files needed to compile.
        f"fast-hadamard-transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@{FAST_HADAMARD_TRANSFORM_COMMIT}",
        extra_options="--no-build-isolation",
        env={"FAST_HADAMARD_TRANSFORM_FORCE_BUILD": "TRUE"},
    )
    .env(
        {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    )
)


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _default_pipeline_layer_split(num_hidden_layers: int, pp_size: int) -> tuple[int, int]:
    base_layers = num_hidden_layers // pp_size
    remainder = num_hidden_layers % pp_size
    extra_first = (remainder + 1) // 2
    extra_last = remainder // 2
    return base_layers + extra_first, base_layers + extra_last


@app.function(
    image=download_image,
    volumes={HF_CACHE: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,
)
def download_model(force: bool = False):
    from huggingface_hub import snapshot_download

    hf_cache_vol.reload()

    print(f"Downloading {HF_MODEL}{'  [force]' if force else ''}...")

    path = snapshot_download(
        HF_MODEL,
        token=os.environ.get("HF_TOKEN"),
        force_download=force,
    )

    print(f"Downloaded to: {path}")
    hf_cache_vol.commit()
    return {"path": path}


@app.function(
    image=download_image,
    volumes={DATA_DIR: data_volume},
    timeout=3600,
)
def prepare_preference_dataset(
    data_folder: str = "distilabel-math-preference-dpo",
    hf_dataset: str = DEFAULT_PREFERENCE_DATASET,
    split: str = "train",
    prompt_col: str = "instruction",
    chosen_col: str = "chosen_response",
    rejected_col: str = "rejected_response",
    system_message: str = "You are a careful math tutor. Provide correct, concise reasoning.",
    max_samples: Optional[int] = 256,
    shuffle: bool = True,
    seed: int = 42,
):
    """Download a preference dataset and convert it to ms-swift DPO JSONL."""
    from datasets import load_dataset

    data_volume.reload()

    print(f"Downloading {hf_dataset} (split={split})...")
    ds = load_dataset(hf_dataset, split=split)

    columns = ds.column_names
    print(f"  Columns: {columns}")

    for required_col in (prompt_col, chosen_col, rejected_col):
        if required_col not in columns:
            raise ValueError(
                f"Column {required_col} not found in dataset columns: {columns}"
            )

    if shuffle:
        ds = ds.shuffle(seed=seed)
    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(len(ds), max_samples)))

    output_dir = f"{DATA_DIR}/{data_folder}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/training.jsonl"
    metadata_path = f"{output_dir}/metadata.json"

    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            prompt = _normalize_text(row[prompt_col])
            chosen = _normalize_text(row[chosen_col])
            rejected = _normalize_text(row[rejected_col])

            if not prompt or not chosen or not rejected:
                skipped += 1
                continue

            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ]
            )

            record = {
                "messages": messages,
                "rejected_response": rejected,
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    metadata = {
        "hf_dataset": hf_dataset,
        "split": split,
        "prompt_col": prompt_col,
        "chosen_col": chosen_col,
        "rejected_col": rejected_col,
        "system_message": system_message,
        "written": written,
        "skipped": skipped,
        "max_samples": max_samples,
        "shuffle": shuffle,
        "seed": seed,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {written} preference pairs to {output_path}")
    if skipped:
        print(f"Skipped {skipped} rows with empty prompt/chosen/rejected fields")
    data_volume.commit()
    return {"path": output_path, "count": written, "skipped": skipped}


@app.function(
    image=msswift_glm5_image,
    gpu="B200:8",
    volumes={
        HF_CACHE: hf_cache_vol,
        DATA_DIR: data_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=86400,
    retries=0,
    memory=1048576,
    ephemeral_disk=2048000,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train_dpo(
    run_id: Optional[str] = None,
    data_folder: str = "distilabel-math-preference-dpo",
    merge_lora: bool = False,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    max_length: int = DEFAULT_MAX_LENGTH,
    beta: float = DEFAULT_BETA,
    recompute_granularity: str = DEFAULT_RECOMPUTE_GRANULARITY,
    padding_free: bool = DEFAULT_PADDING_FREE,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
    global_batch_size: int = 8,
    lr: float = 1e-5,
    moe_aux_loss_coeff: float = 1e-3,
    dsa_indexer_loss_coeff: float = DEFAULT_DSA_INDEXER_LOSS_COEFF,
    save_safetensors: bool = DEFAULT_SAVE_SAFETENSORS,
    save_steps: int = 25,
    logging_steps: int = 1,
    eval_iters: int = 5,
    eval_steps: int = 25,
    disable_packing: bool = True,
    split_dataset_ratio: float = 0.05,
    loss_type: str = "sigmoid",
    reference_free: bool = False,
):
    """Train GLM-5 with DPO using ms-swift Megatron."""
    import subprocess

    if run_id is None:
        run_id = f"train_glm_5_dpo_{int(time.time())}"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else 1
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )

    if tp_size <= 0 or ep_size <= 0 or pp_size <= 0 or cp_size <= 0:
        raise ValueError(
            f"TP/EP/PP/CP must be positive, got TP={tp_size}, EP={ep_size}, PP={pp_size}, CP={cp_size}"
        )

    total_gpus = n_nodes * 8
    model_parallel_size = tp_size * ep_size * pp_size * cp_size
    if total_gpus % model_parallel_size != 0:
        raise ValueError(
            f"Total GPUs ({total_gpus}) must be divisible by TP*EP*PP*CP ({model_parallel_size})"
        )

    print(f"Node {node_rank}/{n_nodes}, Master: {master_addr}")
    print(f"Model: {HF_MODEL}")
    print(
        f"Parallelism: TP={tp_size}, EP={ep_size}, PP={pp_size}, CP={cp_size}, DP={total_gpus // model_parallel_size}"
    )

    from huggingface_hub import snapshot_download

    try:
        model_dir = snapshot_download(HF_MODEL, local_files_only=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model {HF_MODEL} not found in HF cache. "
            f"Download first: modal run modal_train_dpo_glm5.py::download_model"
        ) from exc

    config_path = os.path.join(model_dir, "config.json")
    num_hidden_layers = None
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        num_hidden_layers = model_config.get("num_hidden_layers")
    if isinstance(num_hidden_layers, int) and num_hidden_layers > 0:
        if num_hidden_layers % pp_size != 0:
            if (
                decoder_first_pipeline_num_layers is None
                and decoder_last_pipeline_num_layers is None
            ):
                (
                    decoder_first_pipeline_num_layers,
                    decoder_last_pipeline_num_layers,
                ) = _default_pipeline_layer_split(num_hidden_layers, pp_size)
                print(
                    "Auto-selected uneven pipeline split for "
                    f"{MODEL_NAME}: first={decoder_first_pipeline_num_layers}, "
                    f"last={decoder_last_pipeline_num_layers}, total_layers={num_hidden_layers}, PP={pp_size}"
                )
            elif (
                decoder_first_pipeline_num_layers is None
                or decoder_last_pipeline_num_layers is None
            ):
                raise ValueError(
                    "Set both decoder_first_pipeline_num_layers and decoder_last_pipeline_num_layers "
                    "when overriding the default uneven pipeline split."
                )

            middle_stages = pp_size - 2
            remaining_layers = (
                num_hidden_layers
                - decoder_first_pipeline_num_layers
                - decoder_last_pipeline_num_layers
            )
            if remaining_layers < 0:
                raise ValueError(
                    "decoder_first_pipeline_num_layers + decoder_last_pipeline_num_layers "
                    f"exceeds num_hidden_layers={num_hidden_layers}"
                )
            if middle_stages > 0 and remaining_layers % middle_stages != 0:
                raise ValueError(
                    "Remaining decoder layers must divide evenly across middle pipeline stages. "
                    f"Got num_hidden_layers={num_hidden_layers}, first={decoder_first_pipeline_num_layers}, "
                    f"last={decoder_last_pipeline_num_layers}, PP={pp_size}"
                )

    dataset_path = f"{DATA_DIR}/{data_folder}/training.jsonl"
    if not os.path.exists(dataset_path):
        raise RuntimeError(
            f"No DPO training data found at {dataset_path}. "
            f"Prepare first: modal run modal_train_dpo_glm5.py::prepare_preference_dataset"
        )

    checkpoint_dir = f"{CHECKPOINTS_DIR}/train_glm_5_dpo_{run_id}"
    packing_enabled = not disable_packing

    resuming = False
    if os.path.exists(checkpoint_dir):
        iter_dirs = sorted(
            d for d in os.listdir(checkpoint_dir) if d.startswith("iter_")
        )
        if iter_dirs:
            resuming = True
            print(f"Resuming from existing checkpoint ({iter_dirs[-1]})")

    os.makedirs(checkpoint_dir, exist_ok=True)
    args_json_path = os.path.join(checkpoint_dir, "args.json")
    if not os.path.exists(args_json_path):
        with open(args_json_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "placeholder": True}, f)

    megatron_cmd = [
        "megatron",
        "rlhf",
        "--rlhf_type",
        "dpo",
        "--model",
        model_dir,
        "--output_dir",
        checkpoint_dir,
        "--dataset",
        dataset_path,
        "--tuner_type",
        "lora",
        "--perform_initialization",
        "--split_dataset_ratio",
        str(split_dataset_ratio),
        "--tensor_model_parallel_size",
        str(tp_size),
        "--expert_model_parallel_size",
        str(ep_size),
        "--pipeline_model_parallel_size",
        str(pp_size),
        "--context_parallel_size",
        str(cp_size),
        "--sequence_parallel",
        "true",
        "--moe_permute_fusion",
        "true",
        "--moe_grouped_gemm",
        "true",
        "--moe_shared_expert_overlap",
        "true",
        "--moe_aux_loss_coeff",
        str(moe_aux_loss_coeff),
        "--dsa_indexer_loss_coeff",
        str(dsa_indexer_loss_coeff),
        "--global_batch_size",
        str(global_batch_size),
        "--recompute_granularity",
        recompute_granularity,
        "--packing",
        str(packing_enabled).lower(),
        "--padding_free",
        str(padding_free).lower(),
        "--use_precision_aware_optimizer",
        "true",
        "--num_train_epochs",
        str(max_epochs),
        "--lr",
        str(lr),
        "--lr_warmup_fraction",
        "0.05",
        "--lr_decay_iters",
        "100000",
        "--min_lr",
        str(lr / 10),
        "--max_length",
        str(max_length),
        "--attention_backend",
        "flash",
        "--dataset_num_proc",
        "8",
        "--save_steps",
        str(save_steps),
        "--save_safetensors",
        str(save_safetensors).lower(),
        "--no_save_optim",
        "true",
        "--no_save_rng",
        "true",
        "--use_hf",
        "1",
        "--add_version",
        "false",
        "--report_to",
        "wandb",
        "--wandb_project",
        WANDB_PROJECT,
        "--wandb_exp_name",
        run_id,
        "--logging_steps",
        str(logging_steps),
        "--eval_iters",
        str(eval_iters),
        "--beta",
        str(beta),
        "--loss_type",
        loss_type,
        "--reference_free",
        str(reference_free).lower(),
    ]
    if decoder_first_pipeline_num_layers is not None:
        megatron_cmd.extend(
            [
                "--decoder_first_pipeline_num_layers",
                str(decoder_first_pipeline_num_layers),
            ]
        )
    if decoder_last_pipeline_num_layers is not None:
        megatron_cmd.extend(
            [
                "--decoder_last_pipeline_num_layers",
                str(decoder_last_pipeline_num_layers),
            ]
        )
    if eval_iters > 0:
        megatron_cmd.extend(["--eval_steps", str(eval_steps)])

    if resuming:
        megatron_cmd.extend(["--load", checkpoint_dir])

    megatron_cmd.extend(
        [
            "--target_modules",
            "all-linear",
            "--lora_rank",
            str(lora_rank),
            "--lora_alpha",
            str(lora_alpha),
            "--merge_lora",
            str(merge_lora).lower(),
        ]
    )

    cmd = [
        "torchrun",
        "--no_python",
        "--nproc_per_node",
        "8",
        "--nnodes",
        str(n_nodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        "29500",
        *megatron_cmd,
    ]

    print(f"Running megatron command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ms-swift DPO failed with code {result.returncode}")

    checkpoints_volume.commit()
    print(
        f"Training {run_id} completed successfully with return code {result.returncode}"
    )
    print(f"Results saved to {checkpoint_dir}")
