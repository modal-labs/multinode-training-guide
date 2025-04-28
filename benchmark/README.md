# Multi-node NCCL Benchmark

Simple NCCL bandwidth benchmark for Modal's multi-node training infrastructure. This benchmark measures the all-reduce performance between nodes using a large tensor (500000x2000 float32), in order to understand the communication performance of your multi-node setup.

## Usage

**2 x 8 x H100, multi-node:**

```bash
modal run main.py
```

## Performance Metrics

The benchmark reports two key metrics:

- **algbw (Algorithm Bandwidth)**: The effective bandwidth from the application's perspective
- **busbw (Bus Bandwidth)**: The actual hardware bandwidth utilization

For more details on these metrics, see the [NVIDIA NCCL Tests documentation](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth).

## Environment Configuration

The benchmark automatically configures RDMA settings for OCI's infrastructure:

- Uses IPv6 for control plane (TCP) communication
- Uses IPv4 for data plane (RDMA) communication
- Configures optimal NCCL parameters for IB/RDMA
- Sets appropriate HCA device ordering
