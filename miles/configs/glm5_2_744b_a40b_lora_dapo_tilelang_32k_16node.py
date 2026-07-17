"""GLM-5.2 LoRA DAPO 32k padded — 16 nodes, NO activation recompute, bf16 rollouts.
tp 8 x cp 4 -> 32 gpu replicas, dp 4, ep 32 unchanged 

"""

from configs import glm5_2_744b_a40b_lora_dapo_tilelang_32k as _base
from configs.base import CHECKPOINTS_PATH, DATA_PATH, ModalConfig

modal = ModalConfig(
    docker_image=_base.modal.docker_image,
    gpu=_base.modal.gpu,
    memory=_base.modal.memory,
    cloud=_base.modal.cloud,
    region=_base.modal.region,
    patch_files=[*_base.modal.patch_files, "miles/megatron_dsa_cp_assert_fix.py"],
    image_run_commands=[
        *_base.modal.image_run_commands,
        # megatron-core blanket-refuses CP for DSA; the TileLang bridge path
        # has its own CP collectives (see the patch docstring), so gate the
        # assert on the backend instead.
        "python /tmp/megatron_dsa_cp_assert_fix.py",
    ],
    image_env=dict(_base.modal.image_env),
)


# filler tokens to mimic 32k context length 
# 26k filler + header + question (~0.3-1.5k) ≈ 26.5-27.5k prompt tokens,
# + 4096 response ≤ ~31.7k, under the 32767 limit.
_FILLER_TOKENS = 26000


class _Miles(_base._Miles):
    actor_num_nodes = 16

    # TP8 x CP4 -> 32-GPU replicas, DP 4. EP32 unchanged (128 % 32 == 0).
    context_parallel_size = 4
    allgather_cp = True

    # The whole point of this config: no activation recompute.
    recompute_granularity = None
    recompute_method = None
    recompute_num_layers = None

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k-pad26k.jsonl"
    rollout_max_response_len = 4096

    # 2 gradient steps per rollout (128 samples / global_batch_size 64).
    num_rollout = 40
    save = f"{CHECKPOINTS_PATH}/GLM-5.2-lora-dapo-32k-16node-ckpt"
    save_interval = 10

    wandb_group = "glm5.2-744B-16node-tilelang-dapo-32k-norecompute"

    def download_data(self) -> None:
        """Generate the padded dapo set (same as the retired padstress config)."""
        import json
        import os
        import random

        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        os.makedirs(f"{DATA_PATH}/dapo-math-17k", exist_ok=True)
        snapshot_download(
            repo_id="zhuzilin/dapo-math-17k",
            repo_type="dataset",
            local_dir=f"{DATA_PATH}/dapo-math-17k",
        )

        dst = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k-pad26k.jsonl"
        if os.path.exists(dst):
            print(f"{dst} already exists, skipping generation")
            return

        tokenizer = AutoTokenizer.from_pretrained(
            "zai-org/GLM-5.2", trust_remote_code=True
        )
        words = (
            "system model tensor kernel matrix vector gradient layer token "
            "attention memory cache buffer stream block thread warp shard "
            "sequence batch epoch metric loss reward policy value state action"
        ).split()

      
        rng = random.Random(0)
        base = " ".join(rng.choice(words) for _ in range(2 * _FILLER_TOKENS))
        filler = tokenizer.decode(
            tokenizer.encode(base, add_special_tokens=False)[:_FILLER_TOKENS]
        )

        src = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
        n_written = 0
        with open(src) as fin, open(dst, "w") as fout:
            for i, line in enumerate(fin):
                sample = json.loads(line)
                question = sample["prompt"][0]["content"]
                sample["prompt"][0]["content"] = (
                    f"Reference log #{i}-{random.Random(i).getrandbits(64):x} "
                    "follows. It is not relevant to the question; ignore it and "
                    "solve the question at the end.\n\n"
                    f"{filler}\n\n{question}"
                )
                fout.write(json.dumps(sample) + "\n")
                n_written += 1

        total = len(
            tokenizer.encode(json.loads(open(dst).readline())["prompt"][0]["content"])
        )
        print(f"Wrote {n_written} padded samples to {dst}; sample 0 = {total} tokens")


miles = _Miles()
