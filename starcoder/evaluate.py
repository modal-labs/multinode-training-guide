import modal
import os

from common import (
    MODEL_MOUNT_PATH,
    MODEL_CACHE_DIR,
    MODEL_VOLUME_NAME,
    MODEL_NAME,
    train_image,
)

app = modal.App("starcoder-evaluate")

model_vol = modal.Volume.from_name(
    MODEL_VOLUME_NAME,
    create_if_missing=True,
)

hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=train_image,
    volumes={
        MODEL_MOUNT_PATH: model_vol,
    },
    secrets=[hf_secret],
    gpu="H100",
    timeout=60 * 10,
)
def generate_text(prompts: list[str], checkpoint_dir: str | None = None):
    """
    Generates text using the fine-tuned model from a specified checkpoint for a list of prompts.
    If checkpoint_dir is None, it tries to use the base model.
    Returns a list of generated responses corresponding to the input prompts.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import os

    model_load_path = MODEL_NAME
    if checkpoint_dir:
        full_checkpoint_path = os.path.join(MODEL_MOUNT_PATH, checkpoint_dir)
        if not os.path.isdir(full_checkpoint_path):
            raise ValueError(
                f"Checkpoint directory {full_checkpoint_path} not found. "
                "Ensure it exists in the model volume."
            )
        print(f"Loading model from checkpoint: {full_checkpoint_path}")
        model_load_path = full_checkpoint_path
    else:
        print(f"No checkpoint_dir provided. Loading base model: {MODEL_NAME}")

    print(f"Loading tokenizer from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=os.environ["HF_TOKEN"],
        padding_side="left",
        cache_dir=MODEL_CACHE_DIR,
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {model_load_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        cache_dir=MODEL_CACHE_DIR,
    )
    model.eval()

    print(f"Tokenizing {len(prompts)} prompts in a batch...")
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    ).to(model.device)

    # Set a seed for reproducibility
    torch.manual_seed(42)

    print(f"Generating text for {len(prompts)} prompts in a batch...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
    )

    print(f"Decoding {len(outputs)} generated sequences...")
    responses = [
        tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in outputs
    ]

    # Truncate responses at the first occurrence of "\n}\n" since this is the end of the function
    stop_sequence = "\n}\n"
    processed_responses = []
    for res in responses:
        stop_index = res.find(stop_sequence)
        if stop_index != -1:
            processed_responses.append(res[: stop_index + len(stop_sequence)])
        else:
            processed_responses.append(res)
    return processed_responses


def aggregate_evaluation_results(
    prompts: list[str],
    model_identifiers: list[str],
    all_generated_outputs_lists: list[list[str]],
) -> list[tuple[str, str, str]]:
    """
    Aggregates evaluation results from model generations or an exception.
    """
    results_data = []

    print("Starmap execution completed. Processing results...")
    for i, model_identifier in enumerate(model_identifiers):
        print(f"Processing results for: {model_identifier}")
        if i < len(all_generated_outputs_lists):
            generated_responses_for_model = all_generated_outputs_lists[i]
            if len(generated_responses_for_model) == len(prompts):
                for prompt_idx, prompt_text in enumerate(prompts):
                    output_text = generated_responses_for_model[prompt_idx]
                    results_data.append((model_identifier, prompt_text, output_text))
            else:
                error_msg = "Error: Response count mismatch"
                print(f"  {error_msg} for {model_identifier}")
                for prompt_text in prompts:
                    results_data.append((model_identifier, prompt_text, error_msg))
        else:
            # Should not happen if starmap returns a result for every input
            error_msg = "Error: Missing result from starmap"
            print(f"  {error_msg} for {model_identifier}")
            for prompt_text in prompts:
                results_data.append((model_identifier, prompt_text, error_msg))
    return results_data


def print_eval_results(results_data: list[tuple[str, str, str]]):
    """
    Prints the evaluation results in a structured format.
    """
    print("\n" + "#" * 30 + " Evaluation Results " + "#" * 30)
    if not results_data:
        print("No evaluation results to display.")
        return

    current_model_identifier = None
    for model_identifier, prompt_text, output_text in results_data:
        if model_identifier != current_model_identifier:
            if current_model_identifier is not None:
                print("\n" + "-" * 70)  # Separator between different models
            print("\n" + "=" * 70)
            print(f"MODEL/CHECKPOINT: {model_identifier}")
            print("=" * 70)
            current_model_identifier = model_identifier

        print("\n---")
        print("PROMPT:")
        print(prompt_text)
        print("---")
        print("GENERATED OUTPUT:")
        print(output_text)
        print("---")

    print("\n" + "#" * 30 + " End of Evaluation " + "#" * 29 + "\n")


@app.function(
    image=train_image,
    volumes={MODEL_MOUNT_PATH: model_vol},
    secrets=[hf_secret],
    gpu="H100",
    timeout=60 * 30,
)
def eval_model(run_name: str):
    prompts = [
        # Go prompts
        """
// Fib takes a number n and returns the nth Fibonacci number using
// the naive recursive algorithm.
func Fib(n int) int {""",
        """
// Fib takes a number n and returns the nth Fibonacci number using
// the efficient iterative algorithm.
func Fib(n int) int {""",
        """
// CheckRotation takes two strings and returns true if one is a rotation of the other.
func CheckRotation(s1 string, s2 string) bool {""",
        """
// FirstNonRepeatingCharacter takes a string and returns the first non-repeating character.
func FirstNonRepeatingCharacter(s string) string {""",
        # Rust prompts
        """
/// Returns the nth Fibonacci number using the naive recursive algorithm.
fn fib(n: u32) -> u32 {""",
        """
/// Returns the nth Fibonacci number using the efficient iterative algorithm.
fn fib(n: u32) -> u32 {""",
        """
/// Returns true if one string is a rotation of the other.
fn check_rotation(s1: &str, s2: &str) -> bool {""",
        """
/// Returns the first non-repeating character in a string.
fn first_non_repeating_character(s: &str) -> char {""",
    ]

    checkpoint_dirs = []
    # Construct the path to the specific model's checkpoint directory
    specific_model_checkpoint_base_path = os.path.join(MODEL_MOUNT_PATH, run_name)

    if os.path.exists(specific_model_checkpoint_base_path) and os.path.isdir(
        specific_model_checkpoint_base_path
    ):
        for item in os.listdir(specific_model_checkpoint_base_path):
            if os.path.isdir(
                os.path.join(specific_model_checkpoint_base_path, item)
            ) and item.startswith("checkpoint-"):
                # Store the path relative to MODEL_MOUNT_PATH for generate_text
                checkpoint_dirs.append(os.path.join(run_name, item))
    checkpoint_dirs.sort()

    results_data = []
    starmap_call_args = []
    model_identifiers = []

    # Build starmap call arguments and model identifiers
    starmap_call_args.append((prompts, None))
    model_identifiers.append("Base Model")

    for ckpt in checkpoint_dirs:
        starmap_call_args.append((prompts, ckpt))
        model_identifiers.append(ckpt)

    if not starmap_call_args:
        print("No models or checkpoints found to evaluate.")
        return

    print(
        f"Starting parallel generation for {len(starmap_call_args)} model configurations using starmap..."
    )

    # Each item in all_generated_outputs_lists will be a list of responses
    # from one call to generate_text (i.e., for one model/checkpoint).
    all_generated_outputs_lists = list(generate_text.starmap(starmap_call_args))

    results_data = aggregate_evaluation_results(
        prompts,
        model_identifiers,
        all_generated_outputs_lists,
    )

    # Print results in sections
    print_eval_results(results_data)
