# K2 Inference: Kimi K2 Multinode Inference

This example demonstrates distributed vLLM inference for Moonshot AI's Kimi-K2-Instruct model across multiple GPU nodes using Ray orchestration on Modal.

## Overview

The setup runs Kimi-K2-Instruct with:
- 4 nodes with 8x H100 GPUs each (32 H100s total)
- Tensor parallel size: 16, Pipeline parallel size: 2
- RDMA networking for high-performance inter-node communication
- Ray for distributed orchestration
- vLLM nightly build for Kimi-K2-Instruct pipeline parallelism support

## Usage

**Run the inference server:**

```bash
modal deploy main.py::K2Tp8Pp4Ep
```

This will start a vLLM server accessible at port 8000 on the head node, exposed via the Flash URL reported by the CLI output. This particular configuration shards the model with 8-way tensor parallelism and 4-way pipeline parallelism, see `main.py` for other options.

**Curl the web endpoint:**

```bash
curl -X POST https://{WORKSPACE}-{ENVIRONMENT}--k2-multinode-inference-k2tp8pp4ep-dev.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello! How are you today?"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

### Configuration

The example is pre-configured for Kimi-K2-Instruct with:
- Model: `moonshotai/Kimi-K2-Instruct`
- Context length: 128,000 tokens
- Max sequences: 256
- Tensor parallel: 8 GPUs
- Pipeline parallel: 4 nodes

To modify these settings, edit the constants in `main.py`:
```python
MODEL = "moonshotai/Kimi-K2-Instruct"
TP_SIZE = 16
PP_SIZE = 2
MAX_MODEL_LEN = 8192 * 2
```
