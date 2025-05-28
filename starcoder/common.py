import modal
import os

DATASET_ID = "bigcode/starcoderdata"
DATASET_VOLUME_NAME = f"{DATASET_ID.replace('/', '-')}-dataset"
DATASET_MOUNT_PATH = "/data"

MODEL_MOUNT_PATH = "/model"
MODEL_CACHE_DIR = f"{MODEL_MOUNT_PATH}/model_cache"
MODEL_VOLUME_NAME = "starcoder-model-v2"
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libibverbs-dev", "libibverbs1")
    .pip_install(
        "datasets>=2.19",
        "sympy",
        "transformers",
        "trl",
        "wandb",
        "huggingface_hub",
        "torch==2.6.0",
        "accelerate",
    )
    .add_local_python_source("common")
)

LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_CODE_DIR = "/root/"
REMOTE_TRAIN_SCRIPT_PATH = "/root/train.py"

train_image = base_image.add_local_dir(
    LOCAL_CODE_DIR,
    remote_path=REMOTE_CODE_DIR,
).add_local_file("../utils/mlx_monitor.py", remote_path="/root/mlx_monitor.py")
