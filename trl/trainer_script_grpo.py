from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
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
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Can be helpful for large models if CPU memory is a constraint during loading
        )
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    print("Attempting to freeze MoE routing layer...")
    num_frozen_gate_layers = 0
    for name, module in model.named_modules():
        # Check if the module is an instance of your MoE block class
        if isinstance(module, Qwen3MoeSparseMoeBlock):
            print(f"Found Qwen3MoeSparseMoeBlock: {name}")

            # The gate is an nn.Linear layer named 'gate' within Qwen3MoeSparseMoeBlock
            if hasattr(module, "gate") and isinstance(module.gate, torch.nn.Linear):
                print(f"  Freezing parameters of 'gate' in {name}...")
                for param_name, param in module.gate.named_parameters():
                    param.requires_grad = False
                    print(
                        f"Froze {param_name} (shape: {param.shape}). requires_grad: {param.requires_grad}"
                    )
                num_frozen_gate_layers += 1
            else:
                print(
                    f"WARNING: Module {name} is a Qwen3MoeSparseMoeBlock but doesn't have a 'gate' attribute "
                    "of type nn.Linear as expected."
                )

    if num_frozen_gate_layers > 0:
        print(
            f"\nSuccessfully froze the gate parameters for {num_frozen_gate_layers} MoE blocks."
        )
    else:
        print(
            "\nNo Qwen3MoeSparseMoeBlock gate layers were found or frozen. "
            "Please check the model structure or the class name used for type checking."
        )
    run_name = "multinode-rl_" + model_name.split("/")[-1].lower()

    training_args = GRPOConfig(
        run_name=run_name,
        num_iterations=50,
        max_steps=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=1,
        learning_rate=1e-3,
        logging_steps=1,
        bf16=True,
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
