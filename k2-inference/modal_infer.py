import pathlib
import modal
import modal.experimental
from inference import K2Inference, app, hf_cache_volume, image, vllm_cache_volume

local_path = pathlib.Path(__file__).parent / "inference.py"
image = image.add_local_file(local_path, "/root/inference.py")


@app.cls(
    image=image,
    gpu="H100:8",
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    min_containers=1,
    timeout=60 * 60 * 1,
    experimental_options={"flash": "us-east"},
)
@modal.experimental.clustered(size=4, rdma=True)
class K2Tp8Pp4Ep(K2Inference):
    # RECOMMENDED
    # single request decodes at ~40 tokens/s
    # 4x8H100
    # tp=ep=8,pp=4,dp=1
    tp_size = 8
    pp_size = 4
    dp_size = 1
    nodes = 4
    max_seqs = None
    max_model_len = 128000
    enable_expert_parallel = True
