from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from constants import NUM_NODES

import torch

def reward_len(completions, **kwargs):
    """
    A simple reward function that rewards completions closer to a target length of 20.
    """
    target_length = 20
    if not isinstance(completions, list) or not all(
        isinstance(c, str) for c in completions
    ):
        print(
            f"Warning: Unexpected format for completions in reward_len: {completions}"
        )
        return [-1000.0] * (len(completions) if isinstance(completions, list) else 1)

    return [-float(abs(target_length - len(completion))) for completion in completions]

dataset = load_dataset("trl-lib/tldr", split="train")

def main():

    model_name = "Qwen/Qwen3-30B-A3B"
    try:
        model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Can be helpful for large models if CPU memory is a constraint during loading
                )
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    run_name = "multinode-rl_" + model_name.split("/")[-1].lower()


    training_args = GRPOConfig(
        run_name=run_name,
        num_iterations=50,
        max_steps=50,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=16,
        learning_rate=1e-3,
        logging_steps=1,
        fp16=True,
        report_to="wandb",
        use_vllm=True,
        vllm_server_host=f"10.100.0.{NUM_NODES}",
        vllm_server_port=8000,
        vllm_gpu_memory_utilization=0.9,
        beta=0.0,  # skip kl divergence in loss calculation
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_len],
        args=training_args,
        train_dataset=dataset,
    )
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()