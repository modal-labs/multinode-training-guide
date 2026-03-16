"""
GLM-5 DPO training via ms-swift Megatron.
"""

import json
import os
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Optional

import modal
import modal.experimental

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR.parent
    if (SCRIPT_DIR.parent / "msswift_eval_helpers.py").exists()
    else SCRIPT_DIR
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep these imports lazy so Modal can import the service module before the
# function image has copied the helper files into place.
TRAIN_FILENAME = "train.jsonl"
EVAL_FILENAME = "eval.jsonl"
METADATA_FILENAME = "metadata.json"

_EVAL_HELPERS = None


def _eval_helpers():
    global _EVAL_HELPERS
    if _EVAL_HELPERS is None:
        _EVAL_HELPERS = import_module("msswift_eval_helpers")
    return _EVAL_HELPERS


def build_eval_record(*args, **kwargs):
    return _eval_helpers().build_eval_record(*args, **kwargs)


def build_train_record(*args, **kwargs):
    return _eval_helpers().build_train_record(*args, **kwargs)


def eval_dir(*args, **kwargs):
    return _eval_helpers().eval_dir(*args, **kwargs)


def export_dir_name(*args, **kwargs):
    return _eval_helpers().export_dir_name(*args, **kwargs)


def latest_checkpoint_dir(*args, **kwargs):
    return _eval_helpers().latest_checkpoint_dir(*args, **kwargs)


def normalize_text(*args, **kwargs):
    return _eval_helpers().normalize_text(*args, **kwargs)


def run_root(*args, **kwargs):
    return _eval_helpers().run_root(*args, **kwargs)

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
SGLANG_COMMIT = "e2be31824fb10df9a003cf9752c87e3678db1550"

DEFAULT_MAX_EPOCHS = 1
DEFAULT_MAX_LENGTH = 2048
DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 16
DEFAULT_BETA = 0.1
DEFAULT_EVAL_SIZE = 64
DEFAULT_EVAL_MAX_NEW_TOKENS = 256
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
RUN_PREFIX = "train_glm_5_dpo"

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
GPU_CONFIG = os.environ.get("GPU_TYPE", "B200:8")
CLUSTER_RDMA = os.environ.get("CLUSTER_RDMA", "true").lower() == "true"
EFA_ENABLED = os.environ.get("EFA_ENABLED", "true").lower() == "true"
EXPERIMENTAL_OPTIONS = {"efa_enabled": True} if EFA_ENABLED else {}


def _with_shared_eval_files(image: modal.Image) -> modal.Image:
    return (
        image.add_local_file(
            REPO_ROOT / "msswift_eval_helpers.py",
            "/root/msswift_eval_helpers.py",
            copy=True,
        )
        .add_local_file(
            REPO_ROOT / "msswift_eval_runtime.py",
            "/root/msswift_eval_runtime.py",
            copy=True,
        )
        .add_local_file(
            REPO_ROOT / "msswift_megatron_logprob_eval.py",
            "/root/msswift_megatron_logprob_eval.py",
            copy=True,
        )
        .add_local_file(
            REPO_ROOT / "msswift_mcore_workarounds.py",
            "/root/msswift_mcore_workarounds.py",
            copy=True,
        )
        .add_local_file(
            REPO_ROOT / "msswift_custom_export.py",
            "/root/msswift_custom_export.py",
            copy=True,
        )
    )


download_image = _with_shared_eval_files(
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        # Keep the prep/download image minimal. It only needs HF Hub + datasets, and
        # pinning a modern hub release avoids conflicts with recent GLM-5 support.
        "huggingface_hub>=1.3.0,<2.0",
        "datasets>=2.14.0",
        "safetensors==0.4.5",
        "requests>=2.32.0",
    )
    .env(
        {
            # Use the Hub's supported high-performance Xet mode for very large
            # checkpoint shards; GLM-5 downloads are otherwise painfully slow.
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

msswift_glm5_image = _with_shared_eval_files(
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
        "requests>=2.32.0",
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

sglang_eval_image = _with_shared_eval_files(
    modal.Image.from_registry(
        "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-modelscope1.33.0-swift3.12.5"
    )
    .uv_pip_install(
        f"sglang @ git+https://github.com/sgl-project/sglang.git@{SGLANG_COMMIT}#subdirectory=python",
        "transformers==4.57.1",
        "requests>=2.32.0",
    )
    .env({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
)

MEGATRON_EVAL_SCRIPT = str(REPO_ROOT / "msswift_megatron_logprob_eval.py")
CUSTOM_EXPORT_SCRIPT = str(REPO_ROOT / "msswift_custom_export.py")


def _run_dir(run_id: str) -> str:
    return run_root(CHECKPOINTS_DIR, RUN_PREFIX, run_id)


def _checkpoint_dir(run_id: str, checkpoint_name: Optional[str]) -> str:
    root_dir = _run_dir(run_id)
    if checkpoint_name:
        return os.path.join(root_dir, checkpoint_name)
    return latest_checkpoint_dir(root_dir)


def _checkpoint_args_payload(
    merge_lora: bool = False,
    mcore_model: Optional[str] = None,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
) -> dict:
    return {
        "model": HF_MODEL,
        "task_type": "causal_lm",
        "tuner_type": "lora",
        "target_modules": ["all-linear"],
        "merge_lora": merge_lora,
        "lora_rank": DEFAULT_LORA_RANK,
        "lora_alpha": DEFAULT_LORA_ALPHA,
        "mcore_model": mcore_model,
        "tensor_model_parallel_size": TP_SIZE,
        "pipeline_model_parallel_size": PP_SIZE,
        "expert_model_parallel_size": EP_SIZE,
        "context_parallel_size": CP_SIZE,
        "sequence_parallel": True,
        "decoder_first_pipeline_num_layers": decoder_first_pipeline_num_layers,
        "decoder_last_pipeline_num_layers": decoder_last_pipeline_num_layers,
    }


def _ensure_checkpoint_args_json(checkpoint_dir: str, payload: dict) -> None:
    args_path = os.path.join(checkpoint_dir, "args.json")
    existing = {}
    if os.path.exists(args_path):
        with open(args_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    updated = existing.copy()
    updated.update({k: v for k, v in payload.items() if v is not None})
    if updated != existing:
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(updated, f)


def _default_pipeline_layer_split(num_hidden_layers: int, pp_size: int) -> tuple[int, int]:
    # Megatron only needs explicit overrides for the edge pipeline stages; the
    # middle stages get an even share of whatever layers remain.
    base_layers = num_hidden_layers // pp_size
    remainder = num_hidden_layers % pp_size
    extra_first = (remainder + 1) // 2
    extra_last = remainder // 2
    return base_layers + extra_first, base_layers + extra_last


def _resolve_pipeline_layer_split(
    model_dir: str,
    pp_size: int,
    decoder_first_pipeline_num_layers: Optional[int],
    decoder_last_pipeline_num_layers: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return decoder_first_pipeline_num_layers, decoder_last_pipeline_num_layers

    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    num_hidden_layers = model_config.get("num_hidden_layers")
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        return decoder_first_pipeline_num_layers, decoder_last_pipeline_num_layers
    if num_hidden_layers % pp_size == 0:
        return decoder_first_pipeline_num_layers, decoder_last_pipeline_num_layers

    if (
        decoder_first_pipeline_num_layers is None
        and decoder_last_pipeline_num_layers is None
    ):
        return _default_pipeline_layer_split(num_hidden_layers, pp_size)
    return decoder_first_pipeline_num_layers, decoder_last_pipeline_num_layers


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
    eval_size: int = DEFAULT_EVAL_SIZE,
    shuffle: bool = True,
    seed: int = 42,
):
    """Download a preference dataset and split it into train/eval JSONL files."""
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
    train_path = os.path.join(output_dir, TRAIN_FILENAME)
    eval_path = os.path.join(output_dir, EVAL_FILENAME)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)

    filtered_rows = []
    skipped = 0
    for idx, row in enumerate(ds):
        prompt = normalize_text(row[prompt_col])
        chosen = normalize_text(row[chosen_col])
        rejected = normalize_text(row[rejected_col])
        if not prompt or not chosen or not rejected:
            skipped += 1
            continue
        filtered_rows.append(
            {
                "id": f"{data_folder}-{idx}",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    eval_count = min(max(eval_size, 0), len(filtered_rows))
    eval_rows = filtered_rows[:eval_count]
    train_rows = filtered_rows[eval_count:]

    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_rows:
            record = build_train_record(
                system_message, row["prompt"], row["chosen"], row["rejected"]
            )
            f.write(json.dumps(record) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for row in eval_rows:
            record = build_eval_record(
                row["id"],
                system_message,
                row["prompt"],
                row["chosen"],
                row["rejected"],
            )
            f.write(json.dumps(record) + "\n")

    metadata = {
        "hf_dataset": hf_dataset,
        "split": split,
        "prompt_col": prompt_col,
        "chosen_col": chosen_col,
        "rejected_col": rejected_col,
        "system_message": system_message,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "skipped": skipped,
        "max_samples": max_samples,
        "eval_size": eval_size,
        "shuffle": shuffle,
        "seed": seed,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(train_rows)} train pairs to {train_path}")
    print(f"Wrote {len(eval_rows)} eval pairs to {eval_path}")
    if skipped:
        print(f"Skipped {skipped} rows with empty prompt/chosen/rejected fields")
    data_volume.commit()
    return {
        "train_path": train_path,
        "eval_path": eval_path,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "skipped": skipped,
    }


@app.function(
    image=msswift_glm5_image,
    timeout=3600,
)
def inspect_swift_megatron_export_arguments():
    import inspect

    from swift.megatron.arguments import MegatronExportArguments

    print(inspect.signature(MegatronExportArguments))
    return {"signature": str(inspect.signature(MegatronExportArguments))}


@app.function(
    image=msswift_glm5_image,
    gpu=GPU_CONFIG,
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
    experimental_options=EXPERIMENTAL_OPTIONS,
)
@modal.experimental.clustered(size=N_NODES, rdma=CLUSTER_RDMA)
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

    # Each clustered task is one 8-GPU node on Modal, so total world size here
    # is just 8 * n_nodes.
    total_gpus = n_nodes * 8
    model_parallel_size = tp_size * ep_size * pp_size * cp_size
    canonical_dp_size = total_gpus // (tp_size * pp_size * cp_size)
    if total_gpus % model_parallel_size != 0:
        raise ValueError(
            f"Total GPUs ({total_gpus}) must be divisible by TP*EP*PP*CP ({model_parallel_size})"
        )

    print(f"Node {node_rank}/{n_nodes}, Master: {master_addr}")
    print(f"Model: {HF_MODEL}")
    print(
        f"Parallelism: TP={tp_size}, EP={ep_size}, PP={pp_size}, CP={cp_size}, Megatron_DP={canonical_dp_size}"
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
            # GLM-5 has 78 decoder layers, so PP=4 cannot be expressed as an
            # even split. ms-swift exposes first/last-stage overrides, so we
            # derive those from config.json instead of hard-coding model internals.
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

    dataset_path = f"{DATA_DIR}/{data_folder}/{TRAIN_FILENAME}"
    if not os.path.exists(dataset_path):
        raise RuntimeError(
            f"No DPO training data found at {dataset_path}. "
            f"Prepare first: modal run modal_train_dpo_glm5.py::prepare_preference_dataset"
        )

    checkpoint_dir = _run_dir(run_id)
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
        # ms-swift's checkpoint saver expects an args.json to already exist next
        # to output_dir so it can copy it into each checkpoint directory.
        with open(args_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "model": HF_MODEL,
                    "task_type": "causal_lm",
                    "tuner_type": "lora",
                    "merge_lora": merge_lora,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "decoder_first_pipeline_num_layers": decoder_first_pipeline_num_layers,
                    "decoder_last_pipeline_num_layers": decoder_last_pipeline_num_layers,
                },
                f,
            )

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
        # GLM-5's DSA attention path currently breaks with padding_free=True in
        # this stack, so keep the switch explicit in the generated command.
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
        # HF/safetensors export currently crashes for GLM-5, so default to
        # Megatron-Core checkpoints unless the user opts back in explicitly.
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


def _export_for_inference_impl(
    run_id: str,
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
    peft_format: bool = False,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
    *,
    node_rank: int,
    n_nodes: int,
    master_addr: str,
):
    import subprocess
    from huggingface_hub import snapshot_download

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    export_dir = (
        os.path.join(_run_dir(run_id), export_name)
        if export_name
        else export_dir_name(checkpoint_dir)
    )

    try:
        model_dir = snapshot_download(HF_MODEL, local_files_only=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model {HF_MODEL} not found in HF cache. "
            f"Download first: modal run modal_train_dpo_glm5.py::download_model"
        ) from exc

    (
        decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers,
    ) = _resolve_pipeline_layer_split(
        model_dir,
        pp_size,
        decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers,
    )
    _ensure_checkpoint_args_json(
        checkpoint_dir,
        _checkpoint_args_payload(
            merge_lora=False,
            mcore_model=checkpoint_dir,
            decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
            decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
        ),
    )

    export_cmd = [
        "--base-model-dir",
        model_dir,
        "--checkpoint-dir",
        checkpoint_dir,
        "--output-dir",
        export_dir,
        "--tp-size",
        str(tp_size),
        "--ep-size",
        str(ep_size),
        "--pp-size",
        str(pp_size),
        "--cp-size",
        str(cp_size),
        "--sequence-parallel",
    ]
    if peft_format:
        export_cmd.append("--peft-format")
    if decoder_first_pipeline_num_layers is not None:
        export_cmd.extend(
            [
                "--decoder-first-pipeline-num-layers",
                str(decoder_first_pipeline_num_layers),
            ]
        )
    if decoder_last_pipeline_num_layers is not None:
        export_cmd.extend(
            [
                "--decoder-last-pipeline-num-layers",
                str(decoder_last_pipeline_num_layers),
            ]
        )

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "8",
        "--nnodes",
        str(n_nodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        "29511",
        CUSTOM_EXPORT_SCRIPT,
        *export_cmd,
    ]

    print(f"Running export command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ms-swift export failed with code {result.returncode}")

    checkpoints_volume.commit()
    return {"checkpoint_dir": checkpoint_dir, "export_dir": export_dir}


@app.function(
    image=msswift_glm5_image,
    gpu=GPU_CONFIG,
    volumes={
        HF_CACHE: hf_cache_vol,
        DATA_DIR: data_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
    retries=0,
    memory=1048576,
    ephemeral_disk=2048000,
    experimental_options=EXPERIMENTAL_OPTIONS,
)
@modal.experimental.clustered(size=N_NODES, rdma=CLUSTER_RDMA)
def export_for_inference(
    run_id: str,
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
    peft_format: bool = False,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
):
    cluster_info = modal.experimental.get_cluster_info()
    return _export_for_inference_impl(
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        export_name=export_name,
        peft_format=peft_format,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        cp_size=cp_size,
        decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
        node_rank=cluster_info.rank,
        n_nodes=len(cluster_info.container_ips) if cluster_info.container_ips else 1,
        master_addr=cluster_info.container_ips[0]
        if cluster_info.container_ips
        else "localhost",
    )


@app.function(
    image=msswift_glm5_image,
    gpu=GPU_CONFIG,
    volumes={
        HF_CACHE: hf_cache_vol,
        DATA_DIR: data_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
    retries=0,
    memory=1048576,
    ephemeral_disk=2048000,
)
def export_for_inference_single_node(
    run_id: str,
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
    peft_format: bool = False,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
):
    return _export_for_inference_impl(
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        export_name=export_name,
        peft_format=peft_format,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        cp_size=cp_size,
        decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
        node_rank=0,
        n_nodes=1,
        master_addr="localhost",
    )


def _evaluate_megatron_native_impl(
    run_id: str,
    data_folder: str = "distilabel-math-preference-dpo",
    checkpoint_name: Optional[str] = None,
    max_eval_samples: Optional[int] = DEFAULT_EVAL_SIZE,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
    *,
    node_rank: int,
    n_nodes: int,
    master_addr: str,
):
    import subprocess
    from huggingface_hub import snapshot_download

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    eval_dataset = os.path.join(DATA_DIR, data_folder, EVAL_FILENAME)
    output_dir = eval_dir(checkpoint_dir, "megatron-native")

    try:
        model_dir = snapshot_download(HF_MODEL, local_files_only=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model {HF_MODEL} not found in HF cache. "
            f"Download first: modal run modal_train_dpo_glm5.py::download_model"
        ) from exc

    (
        decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers,
    ) = _resolve_pipeline_layer_split(
        model_dir,
        pp_size,
        decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers,
    )
    _ensure_checkpoint_args_json(
        checkpoint_dir,
        _checkpoint_args_payload(
            merge_lora=False,
            mcore_model=checkpoint_dir,
            decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
            decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
        ),
    )

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "8",
        "--nnodes",
        str(n_nodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        "29512",
        MEGATRON_EVAL_SCRIPT,
        "--base-model-dir",
        model_dir,
        "--checkpoint-dir",
        checkpoint_dir,
        "--eval-dataset",
        eval_dataset,
        "--output-dir",
        output_dir,
        "--tp-size",
        str(tp_size),
        "--ep-size",
        str(ep_size),
        "--pp-size",
        str(pp_size),
        "--cp-size",
        str(cp_size),
        "--sequence-parallel",
    ]
    if decoder_first_pipeline_num_layers is not None:
        cmd.extend(
            [
                "--decoder-first-pipeline-num-layers",
                str(decoder_first_pipeline_num_layers),
            ]
        )
    if decoder_last_pipeline_num_layers is not None:
        cmd.extend(
            [
                "--decoder-last-pipeline-num-layers",
                str(decoder_last_pipeline_num_layers),
            ]
        )
    if max_eval_samples is not None:
        cmd.extend(["--max-samples", str(max_eval_samples)])

    print(f"Running Megatron-native eval command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Megatron-native eval failed with code {result.returncode}"
        )

    checkpoints_volume.commit()
    return {
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "results_path": os.path.join(output_dir, "per_example.jsonl"),
    }


@app.function(
    image=msswift_glm5_image,
    gpu=GPU_CONFIG,
    volumes={
        HF_CACHE: hf_cache_vol,
        DATA_DIR: data_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
    retries=0,
    memory=1048576,
    ephemeral_disk=2048000,
    experimental_options=EXPERIMENTAL_OPTIONS,
)
@modal.experimental.clustered(size=N_NODES, rdma=CLUSTER_RDMA)
def evaluate_megatron_native(
    run_id: str,
    data_folder: str = "distilabel-math-preference-dpo",
    checkpoint_name: Optional[str] = None,
    max_eval_samples: Optional[int] = DEFAULT_EVAL_SIZE,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
):
    cluster_info = modal.experimental.get_cluster_info()
    return _evaluate_megatron_native_impl(
        run_id=run_id,
        data_folder=data_folder,
        checkpoint_name=checkpoint_name,
        max_eval_samples=max_eval_samples,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        cp_size=cp_size,
        decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
        node_rank=cluster_info.rank,
        n_nodes=len(cluster_info.container_ips) if cluster_info.container_ips else 1,
        master_addr=cluster_info.container_ips[0]
        if cluster_info.container_ips
        else "localhost",
    )


@app.function(
    image=msswift_glm5_image,
    gpu=GPU_CONFIG,
    volumes={
        HF_CACHE: hf_cache_vol,
        DATA_DIR: data_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
    retries=0,
    memory=1048576,
    ephemeral_disk=2048000,
)
def evaluate_megatron_native_single_node(
    run_id: str,
    data_folder: str = "distilabel-math-preference-dpo",
    checkpoint_name: Optional[str] = None,
    max_eval_samples: Optional[int] = DEFAULT_EVAL_SIZE,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    decoder_first_pipeline_num_layers: Optional[int] = None,
    decoder_last_pipeline_num_layers: Optional[int] = None,
):
    return _evaluate_megatron_native_impl(
        run_id=run_id,
        data_folder=data_folder,
        checkpoint_name=checkpoint_name,
        max_eval_samples=max_eval_samples,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        cp_size=cp_size,
        decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
        node_rank=0,
        n_nodes=1,
        master_addr="localhost",
    )


@app.function(
    image=msswift_glm5_image,
    gpu=GPU_CONFIG,
    volumes={DATA_DIR: data_volume, CHECKPOINTS_DIR: checkpoints_volume},
    timeout=43200,
    retries=0,
    memory=1048576,
    ephemeral_disk=2048000,
)
def evaluate_hf_native(
    run_id: str,
    data_folder: str = "distilabel-math-preference-dpo",
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
    max_eval_samples: Optional[int] = DEFAULT_EVAL_SIZE,
    max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS,
):
    from huggingface_hub import snapshot_download
    from msswift_eval_runtime import hf_score_and_generate

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    export_dir = (
        os.path.join(_run_dir(run_id), export_name)
        if export_name
        else export_dir_name(checkpoint_dir)
    )
    eval_dataset = os.path.join(DATA_DIR, data_folder, EVAL_FILENAME)
    output_dir = eval_dir(export_dir, "hf-native")
    adapter_config_path = os.path.join(export_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        model_dir = snapshot_download(HF_MODEL, local_files_only=True)
        tokenizer_dir = model_dir
        adapter_dir = export_dir
    else:
        model_dir = export_dir
        tokenizer_dir = export_dir
        adapter_dir = None

    result = hf_score_and_generate(
        model_dir,
        tokenizer_dir,
        eval_dataset,
        output_dir,
        max_eval_samples,
        max_new_tokens,
        adapter_dir=adapter_dir,
    )
    checkpoints_volume.commit()
    return {
        "checkpoint_dir": checkpoint_dir,
        "export_dir": export_dir,
        "output_dir": output_dir,
        **result,
    }


@app.function(
    image=sglang_eval_image,
    gpu=GPU_CONFIG,
    volumes={DATA_DIR: data_volume, CHECKPOINTS_DIR: checkpoints_volume},
    timeout=43200,
    retries=0,
    memory=1048576,
    ephemeral_disk=2048000,
)
def evaluate_sglang(
    run_id: str,
    data_folder: str = "distilabel-math-preference-dpo",
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
    max_eval_samples: Optional[int] = DEFAULT_EVAL_SIZE,
    max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS,
    sglang_tp_size: int = 8,
):
    from huggingface_hub import snapshot_download
    from msswift_eval_runtime import sglang_score_and_generate

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    export_dir = (
        os.path.join(_run_dir(run_id), export_name)
        if export_name
        else export_dir_name(checkpoint_dir)
    )
    eval_dataset = os.path.join(DATA_DIR, data_folder, EVAL_FILENAME)
    output_dir = eval_dir(export_dir, "sglang")
    adapter_config_path = os.path.join(export_dir, "adapter_config.json")
    server_extra_args = [
        "--max-total-tokens",
        "1000000",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--skip-server-warmup",
    ]
    lora_name = None
    model_dir = export_dir
    tokenizer_dir = export_dir
    if os.path.exists(adapter_config_path):
        model_dir = snapshot_download(HF_MODEL, local_files_only=True)
        tokenizer_dir = model_dir
        lora_name = "finetune"
        server_extra_args.extend(
            [
                "--enable-lora",
                "--lora-paths",
                f"{lora_name}={export_dir}",
            ]
        )

    result = sglang_score_and_generate(
        model_dir,
        tokenizer_dir,
        eval_dataset,
        output_dir,
        max_eval_samples,
        max_new_tokens,
        tp_size=sglang_tp_size,
        startup_timeout_s=1800,
        server_extra_args=server_extra_args,
        lora_name=lora_name,
    )
    checkpoints_volume.commit()
    return {
        "checkpoint_dir": checkpoint_dir,
        "export_dir": export_dir,
        "output_dir": output_dir,
        **result,
    }


@app.function(
    image=download_image,
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
    timeout=3600,
)
def write_parity_report(
    run_id: str,
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
):
    from msswift_eval_runtime import write_parity_report_from_paths

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    export_dir = (
        os.path.join(_run_dir(run_id), export_name)
        if export_name
        else export_dir_name(checkpoint_dir)
    )
    output_path = os.path.join(export_dir, "eval", "parity_report.json")
    report = write_parity_report_from_paths(
        os.path.join(eval_dir(checkpoint_dir, "megatron-native"), "per_example.jsonl"),
        os.path.join(eval_dir(export_dir, "hf-native"), "per_example.jsonl"),
        os.path.join(eval_dir(export_dir, "sglang"), "per_example.jsonl"),
        output_path,
    )
    checkpoints_volume.commit()
    return {"output_path": output_path, "summary": report["summary"]}


@app.local_entrypoint()
def evaluate_all(
    run_id: str,
    data_folder: str = "distilabel-math-preference-dpo",
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
    max_eval_samples: int = DEFAULT_EVAL_SIZE,
    max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS,
    sglang_tp_size: int = 8,
):
    export_result = export_for_inference.remote(
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        export_name=export_name,
    )
    megatron_result = evaluate_megatron_native.remote(
        run_id=run_id,
        data_folder=data_folder,
        checkpoint_name=checkpoint_name,
        max_eval_samples=max_eval_samples,
    )
    hf_result = evaluate_hf_native.remote(
        run_id=run_id,
        data_folder=data_folder,
        checkpoint_name=checkpoint_name,
        export_name=export_name,
        max_eval_samples=max_eval_samples,
        max_new_tokens=max_new_tokens,
    )
    sglang_result = evaluate_sglang.remote(
        run_id=run_id,
        data_folder=data_folder,
        checkpoint_name=checkpoint_name,
        export_name=export_name,
        max_eval_samples=max_eval_samples,
        max_new_tokens=max_new_tokens,
        sglang_tp_size=sglang_tp_size,
    )
    parity_result = write_parity_report.remote(
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        export_name=export_name,
    )
    print(
        json.dumps(
            {
                "export": export_result,
                "megatron": megatron_result,
                "hf": hf_result,
                "sglang": sglang_result,
                "parity": parity_result,
            },
            indent=2,
        )
    )
