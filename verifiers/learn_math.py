# # Training a mathematical reasoning model using the verifiers library with sandboxed code execution

# This example demonstrates how to train mathematical reasoning models on Modal using the [verifiers library](https://github.com/willccbb/verifiers) with [Modal Sandboxes](https://modal.com/docs/guide/sandbox) for executing generated code.
# The [verifiers library](https://github.com/willccbb/verifiers) is a set of tools and abstractions for training LLMs with reinforcement learning in verifiable multi-turn environments via [GRPO](https://arxiv.org/abs/2402.03300).

# This example demonstrates how to:
# - Launch a distributed GRPO training job on Modal with 4 nodes, each with 8x H100 GPUs.
# - Use vLLM for inference during training.
# - Cache HuggingFace, vLLM, and store the model weights in [Volumes](https://modal.com/docs/guide/volumes).
# - Run inference by loading the trained model from Volumes.

# ## Setup
# We start by importing modal and the dependencies from the verifiers library. Then, we create a Modal App and an image with a NVIDIA CUDA base image.
# We install the dependencies for the `verifiers` and `flash-attn` libraries, following the verifiers [README](https://github.com/willccbb/verifiers?tab=readme-ov-file#getting-started).

import modal
import modal.experimental

app = modal.App(name="learn-math")
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/"
    "flash_attn-2.7.1.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)  # We use a pre-built binary for flash-attn to install it in the image.

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "clang")
    .pip_install(
        "setuptools==69.0.3",
        "wheel==0.45.1",
        "ninja==1.11.1.4",
        "packaging==25.0",
        "hf_transfer",
        "huggingface_hub[hf_xet]==0.32.4",
        flash_attn_release,
    )
    .run_commands("pip install 'verifiers[all]==0.1.1'")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "HF_XET_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/root/.cache/huggingface",
            "NCCL_DEBUG": "INFO",
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        }
    )
    .apt_install("iproute2")
)

# ## Caching HuggingFace, vLLM, and storing model weights
# We create Modal Volumes to persist:
# - HuggingFace downloads
# - vLLM cache
# - Model weights

# We define the model name and a tool that the model can use to execute Python code that it generates.
# See this [this training script](/docs/examples/trainer_script_grpo) for more details.

HF_CACHE_DIR = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

VLLM_CACHE_DIR = "/root/.cache/vllm"
VLLM_CACHE_VOL = modal.Volume.from_name("vllm-cache", create_if_missing=True)

WEIGHTS_DIR = "/root/math_weights"
WEIGHTS_VOL = modal.Volume.from_name("math-rl-weights", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"

TOOL_DESCRIPTIONS = """
- sandbox_exec: Execute Python code to perform calculations.
"""

# ## Training
# Following the [verifiers example](https://github.com/willccbb/verifiers/blob/main/verifiers/examples/math_python.py), we will need a training script and a config file.
# For sandboxed code execution, we will use [this training script](/docs/examples/trainer_script_grpo) and the config file defined [here](https://github.com/willccbb/verifiers/blob/main/configs/zero3.yaml).

# We create a function that uses 4 nodes, each with 8 H100 GPUs and mounts the defined Volumes. Then, we write the training script and the config file to the root directory.
# We use the `Qwen/Qwen3-235B-A22B` model from HuggingFace, setting up inference via a vLLM server. Once, the model is served, we will launch the training script using `accelerate`.
# When the training is complete, we will run a single inference from the training set to test our training run.


def run_vllm_server():
    import subprocess
    cmd = [
        "trl",
        "vllm-serve",
        "--model",
        MODEL_NAME,
        "--port",
        "8000",
        "--log-level",
        "info",
        "--tensor-parallel-size",
        "8",
        "--host",
        "0.0.0.0",
    ]
    vllm_process = subprocess.Popen(
        ["/bin/bash", "-c", " ".join(cmd)],
        stderr=subprocess.STDOUT,
    )
    import subprocess
    print("Overlay0 IP address info:")
    subprocess.run(["ip", "address", "show", "overlay0"])
    vllm_process.wait()

def _generate_node_config(
    machine_rank, leader_ip, leader_port, template_path, target_path
):
    try:
        accelerate_config_template = template_path.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: Accelerate template file not found at {template_path.as_posix()}"
        ) from None

    formatted_config_content = accelerate_config_template.format(
        rank=machine_rank, leader_ip=leader_ip, leader_port=leader_port
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(formatted_config_content)


def run_trainer(cluster_info):
    """Runs the training process on rank 0, waiting for the vLLM server."""
    import requests
    import pathlib
    import time
    import subprocess
    from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE
    from verifiers.utils import load_example_dataset    
    from requests.exceptions import ConnectionError

    parent_dir = pathlib.Path(__file__).parent
    accelerate_config_path = parent_dir / "config_grpo_multinode.yaml"
    generated_accelerate_config_path = parent_dir / "zero3.yaml"
    trainer_script = parent_dir / "trainer_script_grpo.py"

    rank = cluster_info.rank
    vllm_server_host_ip = "10.100.0.4" # TODO: get this from the cluster info: 10.100.0.{rank + 1}
    vllm_server_url = f"http://{vllm_server_host_ip}:8000"

    accelerate_machine_rank = rank
    accelerate_leader_ip = "10.100.0.1"
    accelerate_leader_port = 29500

    _generate_node_config(
        accelerate_machine_rank,
        accelerate_leader_ip,
        accelerate_leader_port,
        accelerate_config_path,
        generated_accelerate_config_path,
    )
    print(
        f"Container Modal rank {rank} (Trainer): Generated accelerate "
        f"config at {generated_accelerate_config_path.as_posix()}"
    )

    print(
        f"Container rank {rank} (Trainer) waiting for vLLM server at {vllm_server_url}..."
    )

    while True:
        try:
            response = requests.get(f"{vllm_server_url}/health")
            if response.status_code:  # Any status code means it's up
                print(f"Container rank {rank} (Trainer): vLLM server is ready!")
                break
        except ConnectionError:
            pass
        time.sleep(5)

    train_cmd = [
        "accelerate",
        "launch",
        "--config-file",
        f"{generated_accelerate_config_path.as_posix()}",
        f"{trainer_script.as_posix()}",
        "--vllm_server_host",
        f"{vllm_server_host_ip}",
    ]

    # Close the communicator
    requests.post(f"{vllm_server_url}/close_communicator")

    print(
        f"Container Modal rank {rank} (Trainer): "
        f"Launching train command: {' '.join(train_cmd)}"
    )

    # Launch the training process
    # Using Popen directly with a list of arguments is generally safer than shell=True
    train_process = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout pipe
        text=True,  # Decode output as text
        bufsize=1,  # Line buffered
        universal_newlines=True,  # Ensure cross-platform newline handling
    )
    # Stream output from the subprocess
    if train_process.stdout:
        for line in iter(train_process.stdout.readline, ""):
            print(f"[Trainer Rank {rank} Output]: {line.strip()}", flush=True)
        train_process.stdout.close()

    train_process.wait()
    if train_process.returncode == 0:
        print(f"Container Modal rank {rank} (Trainer) process exited successfully")
    else:
        print(
            f"Container Modal rank {rank} (Trainer) process exited with code {train_process.returncode}."
        )

    print("Training completed! Running a single inference from test set...")

    dataset = load_example_dataset(
        "math", split="train"
    ).select(
        range(1)
    )  # We use the first example from the training set for inference to test our training run.

    example = dataset[0]
    question = example["question"]
    prompt = (
        DEFAULT_TOOL_PROMPT_TEMPLATE.format(tool_descriptions=TOOL_DESCRIPTIONS)
        + "\n\nProblem: "
        + question
        + "\n\n<think>\n\n<answer>"
    )

    result = inference.remote(prompt)
    print(result)        

@app.function(
    gpu="H100:8",
    image=image,
    volumes={
        HF_CACHE_DIR: HF_CACHE_VOL,
        VLLM_CACHE_DIR: VLLM_CACHE_VOL,
        WEIGHTS_DIR: WEIGHTS_VOL,
    },
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
@modal.experimental.clustered(4, rdma=True)
def math_group_verifier(trainer_script: str, config_file: str):    
    import wandb

    with open("/root/trainer_script_grpo.py", "w") as f:
        f.write(trainer_script)
    with open("/root/config_grpo_multinode.yaml", "w") as f:
        f.write(config_file)

    wandb.init(project="math-rl")
    wandb.config = {"epochs": 10}

    cluster_info = modal.experimental.get_cluster_info()
    rank = cluster_info.rank

    if rank == 3:
        run_vllm_server()
    elif rank <= 2:
        run_trainer(cluster_info)
    else:
        print(f"Container rank {rank} has no specific role in this setup.") 


# ## Inference
# We define an `inference` Modal function that runs on a single GPU and mounts the weights volume.
# Then, we load the trained model from the volume, falling back to the base model if necessary.
# To build the prompt, we apply `DEFAULT_TOOL_PROMPT_TEMPLATE` with `TOOL_DESCRIPTIONS` and the problem text.
# Finally, we tokenize the prompt, generate a response with sampling (temperature, top-p, repetition penalty), then decode and return the answer.

@app.function(
    gpu="H100",
    image=image,
    volumes={
        HF_CACHE_DIR: HF_CACHE_VOL,
        WEIGHTS_DIR: WEIGHTS_VOL,
    },
    timeout=60 * 10,
)
def inference(prompt: str):
    """Test the trained model with the same format as training"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE

    prompt = (
        DEFAULT_TOOL_PROMPT_TEMPLATE.format(tool_descriptions=TOOL_DESCRIPTIONS)
        + "\n\nProblem: "
        + prompt
        + "\n\n<think>\n\n<answer>"
    )

    print("Loading model from weights volume...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{WEIGHTS_DIR}", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"{WEIGHTS_DIR}",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Loaded trained model from weights volume")
    except Exception as e:
        print(f"Could not load trained model: {e}")
        print("Loading base model instead...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, cache_dir=HF_CACHE_DIR, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_response(prompt_text):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt_text) :].strip()

    model_response = generate_response(prompt + "\n\n<think>\n\n<answer>")
    return model_response


# ## Usage
# We create a main function that serves as the entrypoint for the app.
# It supports two modes:
# - train: kick off math_group_verifier with the provided training script and config file
# - inference: invoke inference with prompt string or prompt file

# To run the training, we can use the following command:
# ```bash
# modal run learn_math.py --mode=train --trainer-script=trainer_script_grpo.py --config-file=config_grpo.yaml
# ```
# To run the inference with a custom prompt, we can use the following command:
# ```bash
# modal run learn_math.py --mode=inference --prompt "Find the value of x that satisfies the equation: 2x + 5 = 17"
# ```
# To run the inference with a custom prompt from a file, we can use the following command:
# ```bash
# modal run learn_math.py --mode=inference --prompt-file "prompt.txt"
# ```

@app.local_entrypoint()
def main(
    mode: str = "train",
    prompt: str = None,
    prompt_file: str = None,
    trainer_script: str = "trainer_script_grpo.py",
    config_file: str = "config_grpo.yaml",
):
    if mode == "inference":
        if prompt_file:
            try:
                with open(prompt_file, "r") as f:
                    prompt_text = f.read().strip()
                print(f"Using prompt from file: {prompt_file}")
            except FileNotFoundError:
                print(f"Error: File {prompt_file} not found")
                return
        elif prompt:
            prompt_text = prompt
            print("Using prompt from command line argument")
        else:
            prompt_text = "Find the value of x that satisfies the equation: 2x + 5 = 17"

        print("=" * 50)
        print("Running inference...")
        print("=" * 50)
        print("PROMPT:")
        print(prompt_text)
        print("-" * 30)
        model_response = inference.remote(prompt_text)
        print("MODEL RESPONSE:")
        print(model_response)
        print("-" * 30)

    elif mode == "train":
        print(
            f"Training with trainer script: {trainer_script} and config file: {config_file}"
        )
        with open(trainer_script, "r") as f:
            trainer_content = f.read()
        with open(config_file, "r") as f:
            config_content = f.read()

        math_group_verifier.remote(trainer_content, config_content)

