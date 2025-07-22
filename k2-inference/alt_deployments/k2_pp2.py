import modal
import modal.experimental
from modal_infer import K2Inference, app, hf_cache_volume, image, vllm_cache_volume


@app.cls(
    image=image,
    gpu="H100:8",
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    timeout=60 * 60 * 1,
    experimental_options={"flash": "us-east"},
)
@modal.experimental.clustered(size=4, rdma=True)
class K2Tp16Pp2Ep(K2Inference):
    # 4x8H100
    # tp=ep=16,pp=2,dp=1
    # single request decodes at ~20 tokens/s
    # trading more comm for less risk of pipeline bubbles
    tp_size = 16
    pp_size = 2
    dp_size = 1
    nodes = 4
    max_seqs = 256
    max_model_len = 128000
    enable_expert_parallel = True
