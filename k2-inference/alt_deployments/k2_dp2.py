import pathlib
import modal
import modal.experimental
from inference import K2Inference, app, hf_cache_volume, image, vllm_cache_volume

local_path = pathlib.Path(__file__).parent.parent / "inference.py"
image = image.add_local_file(local_path, "/root/inference.py")


@app.cls(
    image=image,
    gpu="H100:8",
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    timeout=60 * 60 * 1,
    min_containers=1,
    experimental_options={"flash": "us-east"},
)
@modal.experimental.clustered(size=4, rdma=True)
class K2Tp8Dp2Ep(K2Inference):
    # 4x8H100
    # tp=8,pp=1,dp=2,ep=tp*dp=16
    # single request decodes at ~20 tokens/s
    # trading more comm for less risk of pipeline bubbles
    tp_size = 8
    pp_size = 1
    dp_size = 2
    nodes = 4
    max_seqs = 8
    max_model_len = 64000
    enable_expert_parallel = True
