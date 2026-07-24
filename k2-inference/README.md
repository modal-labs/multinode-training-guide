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
modal deploy modal_infer.py
```

This will start a vLLM server accessible at port 8000 on the head node, exposed via the Flash URL reported by the CLI output. This particular configuration shards the model with 8-way tensor parallelism and 4-way pipeline parallelism, see `main.py` for other options.

**Curl the web endpoint:**

```console
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

To modify these settings, inherit from `K2Inference` and set the `tp_size`, `pp_size`, `dp_size`, `nodes` `max_seqs`, `max_model_len`, and `enable_expert_parallel` class attributes. See `alt_deployments/k2_pp2.py` for an example.

## Load Testing

Test performance and identify bottlenecks using Locust:

```bash
# Basic load test
modal run load_test.py --target-url https://your-deployment.modal.run

# High-load distributed test
modal run load_test.py --target-url https://your-deployment.modal.run \
  --distributed --workers 8 --users 1000 --time 15m
```

**Parameters:**
- `--users`: Concurrent users (default: 100)
- `--spawn-rate`: Users/second spawn rate (default: 10) 
- `--time`: Test duration, e.g. "5m", "2h" (default: 5m)
- `--distributed`: Enable multi-worker testing
- `--workers`: Worker processes for distributed tests (default: 4)

**Endpoints tested:**
- `/v1/chat/completions` (standard + streaming)
- `/v1/models` 
- `/health`

Results auto-saved to Modal volume `k2-loadtest-results` with CSV stats, HTML reports, and logs. Expected baseline: ~40 tokens/s single request, scales with `max_seqs=256`.
