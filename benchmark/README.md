# Multi-node NCCL Benchmark

Simple NCCL bandwidth benchmark for Modal's multi-node training infrastructure. This benchmark measures the all-reduce performance between nodes using a large tensor (500000x2000 float32), in order to understand the communication performance of your multi-node setup.

## Usage

**2 x 8 x H100, multi-node:**

```bash
modal run modal_train.py
```

## Performance Metrics

The benchmark reports two key metrics:

- **algbw (Algorithm Bandwidth)**: The effective bandwidth from the application's perspective.
- **busbw (Bus Bandwidth)**: The actual hardware bandwidth utilization.

For example:

```
The average bandwidth of all_reduce with a 4.0GB payload (50 trials, 16 ranks):
 algbw: 239.742 GBps (1917.9 Gbps)
 busbw: 449.515 GBps (3596.1 Gbps)
```

For more details on these metrics, see the [NVIDIA NCCL Tests documentation](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth).

## Environment Configuration

The benchmark automatically configures RDMA settings for Modal's infrastructure:

- Uses IPv6 for control plane (TCP) communication
- Uses IPv4 for data plane (RDMA) communication
- Configures optimal NCCL parameters for IB/RDMA
- Sets appropriate HCA device ordering

## EFA Support

Modal can schedule RDMA-enabled workloads on instances with AWS EFA enabled. Enabling EFA
deepens the pool of available instances for RDMA-enabled workloads, but it requires configuration
at the container level.

`modal_train_efa.py` is configured to work both with regular RoCE-based Infiniband and EFA. The
function logic detects whether EFA is enabled and configures the appropriate libraries.

The Image built from `modal_train_efa.py` will work properly on both Infiniband and EFA instances. This
means there are no downsides to using this image, but we provide it separately to show the difference
between standard Infiniband and EFA.

Note that the image build will take ~15 minutes, but it will be cached permanently once it is built.

```bash
modal run modal_train_efa.py::run_benchmark_efa
```