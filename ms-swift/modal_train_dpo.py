"""
GLM-4.7 DPO training via ms-swift Megatron.
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

HF_MODEL = "zai-org/GLM-4.7"
MODEL_NAME = "GLM-4.7"
DEFAULT_PREFERENCE_DATASET = "argilla/distilabel-math-preference-dpo"
DEFAULT_DATA_FOLDER = "distilabel-math-dpo-256"

DEFAULT_MAX_EPOCHS = 1
DEFAULT_MAX_LENGTH = 2048
DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 16
DEFAULT_BETA = 0.1
DEFAULT_EVAL_SIZE = 64
DEFAULT_EVAL_MAX_NEW_TOKENS = 256
RUN_PREFIX = "train_glm_4_7_dpo"
SGLANG_COMMIT = "e2be31824fb10df9a003cf9752c87e3678db1550"

TP_SIZE = 2
PP_SIZE = 2
EP_SIZE = 4
CP_SIZE = 1

WANDB_PROJECT = "glm-4-7-dpo"

app = modal.App("example-msswift-glm_4_7-dpo")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name(
    "example-msswift-glm-4-7-dpo-data",
    create_if_missing=True,
    version=2,
)
checkpoints_volume = modal.Volume.from_name(
    "example-msswift-glm-4-7-dpo-checkpoints",
    create_if_missing=True,
    version=2,
)

HF_CACHE = "/root/.cache/huggingface"
DATA_DIR = "/data"
CHECKPOINTS_DIR = "/checkpoints"

# GLM-4.7 DPO should be treated as a multinode workload. Default to 2 nodes
# (16x B200), which matches TP=2 x EP=4 x PP=2.
N_NODES = int(os.environ.get("N_NODES", "2"))


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
    modal.Image.debian_slim(python_version="3.11").uv_pip_install(
        "huggingface_hub==0.27.1",
        "transformers>=4.50",
        "torch==2.5.1",
        "safetensors==0.4.5",
        "datasets>=2.14.0",
        "requests>=2.32.0",
    )
)

msswift_v4_image = _with_shared_eval_files(
    modal.Image.from_registry(
        "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.3"
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
        "transformers==4.57.3",
        "ms-swift @ git+https://github.com/modelscope/ms-swift.git@main",
        "einops==0.8.2",
        "wandb==0.19.1",
        "requests>=2.32.0",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
    data_folder: str = DEFAULT_DATA_FOLDER,
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
    image=msswift_v4_image,
    timeout=3600,
)
def inspect_swift_megatron_export_arguments():
    import inspect

    from swift.megatron.arguments import MegatronExportArguments

    print(inspect.signature(MegatronExportArguments))
    return {"signature": str(inspect.signature(MegatronExportArguments))}


@app.function(
    image=msswift_v4_image,
    volumes={HF_CACHE: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def inspect_swift_megatron_export_runtime():
    import inspect
    import pkgutil
    import textwrap

    from swift.megatron.convert import convert_mcore2hf
    import swift

    source = inspect.getsource(convert_mcore2hf)
    print(f"swift_module={swift.__file__}")
    print(
        "swift_submodules="
        + ",".join(sorted(module.name for module in pkgutil.iter_modules(swift.__path__)))
    )
    print("convert_mcore2hf:")
    print(textwrap.shorten(source, width=12000, placeholder=" ..."))
    return {
        "swift_module": swift.__file__,
    }


@app.function(
    image=msswift_v4_image,
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
    timeout=3600,
)
def inspect_megatron_load_args(path: str):
    from swift.megatron.arguments import MegatronArguments

    config = MegatronArguments.load_args_config(path)
    print(config)
    return config


@app.function(
    image=msswift_v4_image,
    timeout=3600,
)
def inspect_load_mcore_checkpoint():
    import inspect
    import textwrap

    from swift.megatron.utils import load_mcore_checkpoint

    print(inspect.signature(load_mcore_checkpoint))
    print(textwrap.shorten(inspect.getsource(load_mcore_checkpoint), width=12000, placeholder=" ..."))
    return {"signature": str(inspect.signature(load_mcore_checkpoint))}


@app.function(
    image=download_image,
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
    timeout=3600,
)
def inspect_checkpoint_tree(path: str, max_entries: int = 200):
    checkpoints_volume.reload()

    entries = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        files.sort()
        rel_root = os.path.relpath(root, path)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        if depth > 4:
            dirs[:] = []
            continue
        entries.append({"type": "dir", "path": "." if rel_root == "." else rel_root})
        for filename in files:
            full_path = os.path.join(root, filename)
            entries.append(
                {
                    "type": "file",
                    "path": os.path.relpath(full_path, path),
                    "size": os.path.getsize(full_path),
                }
            )
        if len(entries) >= max_entries:
            break

    trimmed_entries = entries[:max_entries]
    print(json.dumps(trimmed_entries, indent=2))
    return {"path": path, "entries": trimmed_entries}


@app.function(
    image=msswift_v4_image,
    timeout=3600,
)
def inspect_swift_megatron_utils_surface():
    import inspect

    from swift.megatron import utils as mg_utils

    publicish = sorted(name for name in dir(mg_utils) if "state_dict" in name or "checkpoint" in name or "load" in name)
    print("swift.megatron.utils:", mg_utils.__file__)
    print("surface:")
    for name in publicish:
        print(name)
    for name in (
        "load_mcore_checkpoint",
        "_generate_state_dict",
        "_filter_adapter_state_dict",
        "get_default_load_sharded_strategy",
        "FullyParallelLoadStrategyWrapper",
        "patch_merge_fn",
    ):
        print(f"{name}: {hasattr(mg_utils, name)}")
    if hasattr(mg_utils, "load_mcore_checkpoint"):
        print("load_mcore_checkpoint source:")
        print(inspect.getsource(mg_utils.load_mcore_checkpoint))
    return {"module": mg_utils.__file__, "surface": publicish}


@app.function(
    image=msswift_v4_image,
    timeout=3600,
)
def inspect_swift_megatron_utils_file():
    from swift.megatron import utils as mg_utils

    print("file:", mg_utils.__file__)
    with open(mg_utils.__file__, "r", encoding="utf-8") as f:
        print(f.read())
    return {"module": mg_utils.__file__}


@app.function(
    image=msswift_v4_image,
    timeout=3600,
)
def inspect_swift_megatron_lm_utils_file():
    import inspect

    from swift.megatron.utils import megatron_lm_utils

    print("file:", megatron_lm_utils.__file__)
    with open(megatron_lm_utils.__file__, "r", encoding="utf-8") as f:
        print(f.read())
    print("globals:")
    for name in (
        "_generate_state_dict",
        "_filter_adapter_state_dict",
        "get_default_load_sharded_strategy",
        "FullyParallelLoadStrategyWrapper",
        "dist_checkpointing",
    ):
        obj = megatron_lm_utils.__dict__.get(name)
        print(name, bool(obj), type(obj).__name__ if obj is not None else None)
        if obj is not None and inspect.isfunction(obj):
            print(inspect.getsource(obj))
    return {"module": megatron_lm_utils.__file__}


@app.function(
    image=msswift_v4_image,
    timeout=3600,
)
def inspect_swift_megatron_tuner_helpers():
    import inspect

    from swift.megatron.utils import prepare_mcore_model, tuners_sharded_state_dict

    print("prepare_mcore_model:")
    print(inspect.getsource(prepare_mcore_model))
    print("tuners_sharded_state_dict:")
    print(inspect.getsource(tuners_sharded_state_dict))
    return {"ok": True}


@app.function(
    image=msswift_v4_image,
    timeout=3600,
)
def search_swift_source(pattern: str, max_hits: int = 200):
    import os
    import swift

    hits = []
    root_dir = os.path.dirname(swift.__file__)
    for root, _, files in os.walk(root_dir):
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            full_path = os.path.join(root, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    if pattern in line:
                        hits.append(
                            {
                                "path": full_path,
                                "lineno": lineno,
                                "line": line.rstrip(),
                            }
                        )
                        if len(hits) >= max_hits:
                            print(json.dumps(hits, indent=2))
                            return {"hits": hits}
    print(json.dumps(hits, indent=2))
    return {"hits": hits}


@app.function(
    image=msswift_v4_image,
    timeout=3600,
)
def inspect_swift_file(path: str, start_line: int = 1, end_line: int = 200):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    excerpt = "".join(
        f"{lineno:04d}: {line}"
        for lineno, line in enumerate(lines[start_idx:end_idx], start=start_line)
    )
    print(excerpt)
    return {"path": path, "start_line": start_line, "end_line": end_idx}


@app.function(
    image=download_image,
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
    timeout=3600,
)
def inspect_adapter_safetensors_keys(run_id: str, checkpoint_name: Optional[str] = None):
    from safetensors import safe_open

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    with safe_open(adapter_path, framework="pt") as f:
        keys = list(f.keys())
    print(f"adapter_path={adapter_path}")
    print(f"num_keys={len(keys)}")
    for key in keys[:200]:
        print(key)
    return {"adapter_path": adapter_path, "num_keys": len(keys), "sample_keys": keys[:50]}


@app.function(
    image=download_image,
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
    timeout=3600,
)
def inspect_adapter_bridge_compatibility(
    run_id: str,
    checkpoint_name: Optional[str] = None,
    max_layers: int = 8,
):
    from safetensors import safe_open

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    qkv_results = []
    mlp_results = []
    with safe_open(adapter_path, framework="pt") as f:
        keys = set(f.keys())
        for layer_idx in range(max_layers):
            prefix = f"base_model.model.model.layers.{layer_idx}"
            q_key = f"{prefix}.self_attn.q_proj.lora_A.weight"
            k_key = f"{prefix}.self_attn.k_proj.lora_A.weight"
            v_key = f"{prefix}.self_attn.v_proj.lora_A.weight"
            if q_key in keys and k_key in keys and v_key in keys:
                q = f.get_tensor(q_key)
                k = f.get_tensor(k_key)
                v = f.get_tensor(v_key)
                qkv_results.append(
                    {
                        "layer": layer_idx,
                        "qk_equal": bool((q == k).all().item()),
                        "qv_equal": bool((q == v).all().item()),
                        "shape": list(q.shape),
                    }
                )
            gate_key = f"{prefix}.mlp.gate_proj.lora_A.weight"
            up_key = f"{prefix}.mlp.up_proj.lora_A.weight"
            if gate_key in keys and up_key in keys:
                gate = f.get_tensor(gate_key)
                up = f.get_tensor(up_key)
                mlp_results.append(
                    {
                        "layer": layer_idx,
                        "gate_up_equal": bool((gate == up).all().item()),
                        "shape": list(gate.shape),
                    }
                )
    payload = {
        "adapter_path": adapter_path,
        "qkv_lora_a": qkv_results,
        "gate_up_lora_a": mlp_results,
    }
    print(json.dumps(payload, indent=2))
    return payload


@app.function(
    image=msswift_v4_image,
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
    timeout=3600,
)
def inspect_dist_checkpoint_metadata(run_id: str, checkpoint_name: Optional[str] = None):
    import pickle

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    meta_path = os.path.join(checkpoint_dir, "iter_0000030", ".metadata")
    with open(meta_path, "rb") as f:
        obj = pickle.load(f)
    print(type(obj))
    print("attrs:", [a for a in dir(obj) if not a.startswith("_")])
    state_dict_metadata = getattr(obj, "state_dict_metadata", None)
    if isinstance(state_dict_metadata, dict):
        print("state_dict_metadata size:", len(state_dict_metadata))
        for i, key in enumerate(state_dict_metadata):
            print(key)
            if i >= 200:
                break
    all_local_plans = getattr(getattr(obj, "mcore_data", {}), "get", lambda *_: None)("all_local_plans")
    print("all_local_plans type:", type(all_local_plans))
    return {"meta_path": meta_path}


@app.function(
    image=msswift_v4_image,
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
    retries=2,
    memory=1048576,
    ephemeral_disk=2048000,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train_dpo(
    run_id: Optional[str] = None,
    data_folder: str = DEFAULT_DATA_FOLDER,
    merge_lora: bool = False,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    max_length: int = DEFAULT_MAX_LENGTH,
    beta: float = DEFAULT_BETA,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
    global_batch_size: int = 8,
    lr: float = 1e-5,
    moe_aux_loss_coeff: float = 1e-3,
    save_interval: int = 25,
    eval_iters: int = 5,
    eval_interval: int = 25,
    disable_packing: bool = True,
    split_dataset_ratio: float = 0.05,
    loss_type: str = "sigmoid",
    reference_free: bool = False,
):
    """Train GLM-4.7 with DPO using ms-swift Megatron."""
    import subprocess

    if run_id is None:
        run_id = f"train_glm_4_7_dpo_{int(time.time())}"

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
            f"Download first: modal run modal_train_dpo.py::download_model"
        ) from exc

    dataset_path = f"{DATA_DIR}/{data_folder}/{TRAIN_FILENAME}"
    if not os.path.exists(dataset_path):
        raise RuntimeError(
            f"No DPO training data found at {dataset_path}. "
            f"Prepare first: modal run modal_train_dpo.py::prepare_preference_dataset"
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
        "--global_batch_size",
        str(global_batch_size),
        "--packing",
        str(packing_enabled).lower(),
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
        "--save_interval",
        str(save_interval),
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
        "--log_interval",
        "1",
        "--eval_iters",
        str(eval_iters),
        "--beta",
        str(beta),
        "--loss_type",
        loss_type,
        "--reference_free",
        str(reference_free).lower(),
    ]
    if eval_iters > 0:
        megatron_cmd.extend(["--eval_interval", str(eval_interval)])

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


@app.function(
    image=msswift_v4_image,
    gpu="B200:8",
    volumes={
        HF_CACHE: hf_cache_vol,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=43200,
    retries=0,
    memory=1048576,
    ephemeral_disk=2048000,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def export_for_inference(
    run_id: str,
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
):
    import subprocess

    from huggingface_hub import snapshot_download

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else 1
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )

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
            f"Download first: modal run modal_train_dpo.py::download_model"
        ) from exc

    os.makedirs(export_dir, exist_ok=True)
    print(
        "Exporting HF artifact via distributed GPT bridge: "
        f"checkpoint={checkpoint_dir} output={export_dir}"
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
        "29503",
        CUSTOM_EXPORT_SCRIPT,
        "--base-model-dir",
        model_dir,
        "--checkpoint-dir",
        checkpoint_dir,
        "--output-dir",
        export_dir,
        "--tp-size",
        str(TP_SIZE),
        "--ep-size",
        str(EP_SIZE),
        "--pp-size",
        str(PP_SIZE),
        "--cp-size",
        str(CP_SIZE),
        "--sequence-parallel",
    ]
    print(f"Running export command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HF export failed with code {result.returncode}")

    checkpoints_volume.commit()
    return {"checkpoint_dir": checkpoint_dir, "export_dir": export_dir}


@app.function(
    image=download_image,
    volumes={HF_CACHE: hf_cache_vol, CHECKPOINTS_DIR: checkpoints_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def finalize_partial_export(
    run_id: str,
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
    finalized_export_name: Optional[str] = None,
):
    import glob
    import shutil

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    partial_export_dir = (
        os.path.join(_run_dir(run_id), export_name)
        if export_name
        else export_dir_name(checkpoint_dir)
    )
    finalized_dir = (
        os.path.join(_run_dir(run_id), finalized_export_name)
        if finalized_export_name
        else f"{partial_export_dir}-finalized"
    )

    try:
        model_dir = snapshot_download(HF_MODEL, local_files_only=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model {HF_MODEL} not found in HF cache. "
            f"Download first: modal run modal_train_dpo.py::download_model"
        ) from exc

    shard_paths = sorted(
        glob.glob(os.path.join(partial_export_dir, "model-*-of-?????.safetensors"))
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"No partial model shards found under {partial_export_dir}"
        )

    if os.path.exists(finalized_dir):
        shutil.rmtree(finalized_dir)
    os.makedirs(finalized_dir, exist_ok=True)

    total_shards = len(shard_paths)
    weight_map = {}
    total_size = 0

    for shard_index, src_path in enumerate(shard_paths, start=1):
        dst_name = f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        dst_path = os.path.join(finalized_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        total_size += os.path.getsize(dst_path)
        with safe_open(dst_path, framework="pt") as shard_file:
            for key in shard_file.keys():
                weight_map[key] = dst_name

    index_payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(
        os.path.join(finalized_dir, "model.safetensors.index.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(index_payload, f, indent=2, sort_keys=True)

    for root, dirs, files in os.walk(model_dir):
        rel_root = os.path.relpath(root, model_dir)
        target_root = finalized_dir if rel_root == "." else os.path.join(finalized_dir, rel_root)
        os.makedirs(target_root, exist_ok=True)
        for dirname in dirs:
            os.makedirs(os.path.join(target_root, dirname), exist_ok=True)
        for filename in files:
            if filename.endswith((".safetensors", ".bin", ".pt")):
                continue
            if filename == "model.safetensors.index.json":
                continue
            shutil.copy2(os.path.join(root, filename), os.path.join(target_root, filename))

    for filename in ("config.json", "tokenizer_config.json"):
        config_path = os.path.join(finalized_dir, filename)
        if not os.path.exists(config_path):
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        changed = False
        if not config.get("_name_or_path"):
            config["_name_or_path"] = HF_MODEL
            changed = True
        if filename == "tokenizer_config.json" and not config.get("name_or_path"):
            config["name_or_path"] = HF_MODEL
            changed = True
        if changed:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, sort_keys=True)

    for filename in ("args.json", "adapter_config.json", "additional_config.json"):
        src = os.path.join(checkpoint_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(finalized_dir, filename))

    print(
        json.dumps(
            {
                "partial_export_dir": partial_export_dir,
                "finalized_dir": finalized_dir,
                "num_shards": total_shards,
                "num_weights": len(weight_map),
            },
            indent=2,
        )
    )
    checkpoints_volume.commit()
    return {
        "partial_export_dir": partial_export_dir,
        "export_dir": finalized_dir,
        "num_shards": total_shards,
        "num_weights": len(weight_map),
    }


@app.function(
    image=download_image,
    volumes={HF_CACHE: hf_cache_vol, CHECKPOINTS_DIR: checkpoints_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def finalize_export_in_place(
    run_id: str,
    checkpoint_name: Optional[str] = None,
    export_name: Optional[str] = None,
):
    import glob
    import shutil

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

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
            f"Download first: modal run modal_train_dpo.py::download_model"
        ) from exc

    shard_paths = sorted(
        glob.glob(os.path.join(export_dir, "model-*-of-?????.safetensors"))
    )
    if not shard_paths:
        raise FileNotFoundError(f"No partial model shards found under {export_dir}")

    total_shards = len(shard_paths)
    renamed_paths = []
    for shard_index, src_path in enumerate(shard_paths, start=1):
        dst_path = os.path.join(
            export_dir, f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        )
        if src_path != dst_path:
            os.replace(src_path, dst_path)
        renamed_paths.append(dst_path)

    weight_map = {}
    total_size = 0
    for shard_path in renamed_paths:
        shard_name = os.path.basename(shard_path)
        total_size += os.path.getsize(shard_path)
        with safe_open(shard_path, framework="pt") as shard_file:
            for key in shard_file.keys():
                weight_map[key] = shard_name

    with open(
        os.path.join(export_dir, "model.safetensors.index.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            f,
            indent=2,
            sort_keys=True,
        )

    for root, dirs, files in os.walk(model_dir):
        rel_root = os.path.relpath(root, model_dir)
        target_root = export_dir if rel_root == "." else os.path.join(export_dir, rel_root)
        os.makedirs(target_root, exist_ok=True)
        for dirname in dirs:
            os.makedirs(os.path.join(target_root, dirname), exist_ok=True)
        for filename in files:
            if filename.endswith((".safetensors", ".bin", ".pt")):
                continue
            if filename == "model.safetensors.index.json":
                continue
            shutil.copy2(os.path.join(root, filename), os.path.join(target_root, filename))

    for filename in ("config.json", "tokenizer_config.json"):
        config_path = os.path.join(export_dir, filename)
        if not os.path.exists(config_path):
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        changed = False
        if not config.get("_name_or_path"):
            config["_name_or_path"] = HF_MODEL
            changed = True
        if filename == "tokenizer_config.json" and not config.get("name_or_path"):
            config["name_or_path"] = HF_MODEL
            changed = True
        if changed:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, sort_keys=True)

    for filename in ("args.json", "adapter_config.json", "additional_config.json"):
        src = os.path.join(checkpoint_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(export_dir, filename))

    print(
        json.dumps(
            {
                "export_dir": export_dir,
                "num_shards": total_shards,
                "num_weights": len(weight_map),
            },
            indent=2,
        )
    )
    checkpoints_volume.commit()
    return {
        "export_dir": export_dir,
        "num_shards": total_shards,
        "num_weights": len(weight_map),
    }


@app.function(
    image=msswift_v4_image,
    gpu="B200:8",
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
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def evaluate_megatron_native(
    run_id: str,
    data_folder: str = DEFAULT_DATA_FOLDER,
    checkpoint_name: Optional[str] = None,
    max_eval_samples: Optional[int] = DEFAULT_EVAL_SIZE,
    tp_size: int = TP_SIZE,
    ep_size: int = EP_SIZE,
    pp_size: int = PP_SIZE,
    cp_size: int = CP_SIZE,
):
    import subprocess
    from huggingface_hub import snapshot_download

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    n_nodes = len(cluster_info.container_ips) if cluster_info.container_ips else 1
    master_addr = (
        cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"
    )

    checkpoint_dir = _checkpoint_dir(run_id, checkpoint_name)
    _ensure_checkpoint_args_json(
        checkpoint_dir,
        _checkpoint_args_payload(
            merge_lora=False,
            mcore_model=checkpoint_dir,
        ),
    )
    eval_dataset = os.path.join(DATA_DIR, data_folder, EVAL_FILENAME)
    output_dir = eval_dir(checkpoint_dir, "megatron-native")

    try:
        model_dir = snapshot_download(HF_MODEL, local_files_only=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model {HF_MODEL} not found in HF cache. "
            f"Download first: modal run modal_train_dpo.py::download_model"
        ) from exc

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
        "29502",
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
        "--merge-lora",
    ]
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
    image=msswift_v4_image,
    gpu="B200:8",
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
def evaluate_hf_native(
    run_id: str,
    data_folder: str = DEFAULT_DATA_FOLDER,
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
    base_model_dir = snapshot_download(HF_MODEL, local_files_only=True)

    result = hf_score_and_generate(
        base_model_dir,
        base_model_dir,
        eval_dataset,
        output_dir,
        max_eval_samples,
        max_new_tokens,
        adapter_dir=checkpoint_dir,
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
    gpu="B200:8",
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
def evaluate_sglang(
    run_id: str,
    data_folder: str = DEFAULT_DATA_FOLDER,
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
    tokenizer_dir = snapshot_download(HF_MODEL, local_files_only=True)

    result = sglang_score_and_generate(
        export_dir,
        tokenizer_dir,
        eval_dataset,
        output_dir,
        max_eval_samples,
        max_new_tokens,
        tp_size=sglang_tp_size,
        startup_timeout_s=1800,
        server_extra_args=[
            "--max-total-tokens",
            "1000000",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--skip-server-warmup",
        ],
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
    data_folder: str = DEFAULT_DATA_FOLDER,
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
