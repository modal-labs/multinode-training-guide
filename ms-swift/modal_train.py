"""
GLM-4.7 SFT with ms-swift on Modal.

Usage:
    modal run ms-swift/modal_train.py::download_model
    modal run ms-swift/modal_train.py::prep_dataset
    modal run --detach ms-swift/modal_train.py::train_sft
"""

import json
import os
from typing import Any, Dict, Tuple

import modal
import modal.experimental

app = modal.App("glm47-ms-swift-sft")

# Volumes
models_volume = modal.Volume.from_name("big-model-hfcache", create_if_missing=True)
data_volume = modal.Volume.from_name("glm47-training-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("glm47-ms-swift-checkpoints", create_if_missing=True)

# Paths
HF_CACHE = "/root/.cache/huggingface"
DATA_DIR = "/data"
CHECKPOINTS_DIR = "/checkpoints"
PREPROCESSED_DIR = f"{DATA_DIR}/longmit-128k"
TRAIN_JSONL = f"{PREPROCESSED_DIR}/training.jsonl"

# Models/Datasets
HF_MODEL = "zai-org/GLM-4.7"
HF_DATASET = "donmaclean/LongMIT-128K"

# Training/Dataset defaults
MAX_SFT_TOKENS = 131_072
DEFAULT_MAX_LENGTH = 16_384
PREP_CPU = 32

# Distributed training defaults
N_NODES = 4
GPUS_PER_NODE = 8
MASTER_PORT = 29500

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface_hub==0.36.0",
        "transformers==4.57.4",
        "torch==2.9.1",
        "safetensors==0.7.0",
        "sentencepiece==0.2.1",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

prep_image = download_image.uv_pip_install("datasets==3.1.0")

swift_image = (
    modal.Image.from_registry("nvcr.io/nvidia/nemo:25.11")
    .entrypoint([])
    .uv_pip_install(
        "ms-swift==3.12.4",
        "deepspeed==0.17.6",
    )
    .run_commands(f"rm -Rf {HF_CACHE}")
)


def _build_longmit_prompt(example: Dict[str, Any]) -> Tuple[str, str]:
    passages = "\n".join(
        [f"Passage {i + 1}:\n{doc['content']}" for i, doc in enumerate(example["all_docs"])]
    )
    prompt = (
        "Answer the question based on the given passages.\n\n"
        f"{passages}\n\n"
        f"Question: {example['question']}\nAnswer:"
    )
    return prompt, example["answer"]


@app.function(
    image=download_image,
    volumes={HF_CACHE: models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=14400,
)
def download_model():
    """Download GLM-4.7 weights to the HuggingFace cache volume."""
    from huggingface_hub import snapshot_download

    models_volume.reload()

    print(f"Downloading model: {HF_MODEL}")
    path = snapshot_download(HF_MODEL, token=os.environ.get("HF_TOKEN"))
    print(f"Model downloaded to: {path}")

    models_volume.commit()
    return {"model_path": path}


@app.function(
    image=prep_image,
    volumes={DATA_DIR: data_volume, HF_CACHE: models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
    cpu=PREP_CPU,
)
def prep_dataset():
    """
    Prepare LongMIT-128K exactly like the Megatron example:
    - source dataset: donmaclean/LongMIT-128K
    - same prompt format
    - same <=131072 token filtering
    - same output path: /data/longmit-128k/training.jsonl
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    data_volume.reload()
    models_volume.reload()
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    print(f"Loading dataset from Hugging Face: {HF_DATASET}")
    dataset = load_dataset(HF_DATASET, split="train", trust_remote_code=True)
    print(f"Loaded {len(dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL,
        use_fast=True,
        trust_remote_code=True,
    )

    def format_longmit(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt, answer = _build_longmit_prompt(example)
        token_count = len(tokenizer(prompt + answer).input_ids)
        return {"input": prompt, "output": answer, "n_tokens": token_count}

    print(f"Formatting and filtering to <= {MAX_SFT_TOKENS} tokens")
    original_len = len(dataset)
    dataset = dataset.map(
        format_longmit,
        remove_columns=dataset.column_names,
        num_proc=PREP_CPU,
    )
    dataset = dataset.filter(
        lambda ex: ex["n_tokens"] <= MAX_SFT_TOKENS,
        num_proc=PREP_CPU,
    )
    filtered_len = len(dataset)
    print(f"Filtered dataset size: {filtered_len} (from {original_len})")

    print(f"Writing JSONL: {TRAIN_JSONL}")
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for example in dataset:
            # Keep input/output schema aligned with the Megatron example.
            json.dump({"input": example["input"], "output": example["output"]}, f)
            f.write("\n")

    fsize = os.path.getsize(TRAIN_JSONL)
    print(f"Created {TRAIN_JSONL} ({fsize:,} bytes)")

    data_volume.commit()
    return {"dataset_path": TRAIN_JSONL, "examples": filtered_len}


@app.function(
    image=swift_image,
    gpu="H100:8",
    volumes={
        HF_CACHE: models_volume,
        DATA_DIR: data_volume,
        CHECKPOINTS_DIR: checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=N_NODES, rdma=True)
def train_sft(
    run_name: str = "glm47-ms-swift-lora",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_length: int = DEFAULT_MAX_LENGTH,
):
    """
    Multi-node SFT for GLM-4.7 with ms-swift.

    This uses the same preprocessed dataset path as the Megatron example:
    /data/longmit-128k/training.jsonl
    """
    import subprocess

    data_volume.reload()
    models_volume.reload()

    if not os.path.exists(TRAIN_JSONL):
        raise RuntimeError(
            f"Dataset missing at {TRAIN_JSONL}. "
            "Run `modal run ms-swift/modal_train.py::prep_dataset` first."
        )

    output_dir = f"{CHECKPOINTS_DIR}/{run_name}"

    cluster_info = modal.experimental.get_cluster_info()
    node_rank = cluster_info.rank
    master_addr = cluster_info.container_ips[0] if cluster_info.container_ips else "localhost"

    # `swift` auto-switches to torchrun when these env vars are present.
    os.environ["NNODES"] = str(N_NODES)
    os.environ["NPROC_PER_NODE"] = str(GPUS_PER_NODE)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(MASTER_PORT)

    cmd = [
        "swift",
        "sft",
        "--model",
        HF_MODEL,
        "--dataset",
        TRAIN_JSONL,
        "--use_hf",
        "true",
        "--train_type",
        "lora",
        "--deepspeed",
        "zero3",
        "--torch_dtype",
        "bfloat16",
        "--target_modules",
        "all-linear",
        "--lora_rank",
        "128",
        "--lora_alpha",
        "32",
        "--lora_dropout",
        "0.05",
        "--num_train_epochs",
        str(num_train_epochs),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--learning_rate",
        "1e-4",
        "--save_strategy",
        "steps",
        "--save_steps",
        "50",
        "--save_total_limit",
        "2",
        "--logging_steps",
        "5",
        "--max_length",
        str(max_length),
        "--truncation_strategy",
        "left",
        "--split_dataset_ratio",
        "0",
        "--dataloader_num_workers",
        "4",
        "--gradient_checkpointing",
        "true",
        "--output_dir",
        output_dir,
    ]

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        cmd.extend(["--hub_token", hf_token])

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        cmd.extend(["--report_to", "wandb", "--run_name", run_name])
    else:
        cmd.extend(["--report_to", "tensorboard"])

    print(f"Starting rank {node_rank}/{N_NODES - 1} with master {master_addr}:{MASTER_PORT}")
    print("Running command:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"swift sft exited with code {result.returncode}")

    checkpoints_volume.commit()
    return {"status": "complete", "output_dir": output_dir}
