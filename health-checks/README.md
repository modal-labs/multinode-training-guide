# Multi-node Health Checks

Health checking inter-node communication over GCP's RoCE v2 and AWS's EFA.

## EFA vs InfiniBand

**EFA (Elastic Fabric Adapter)**: AWS-specific RDMA implementation using libfabric API. Available on AWS with Elastic Fabric Adapter (EFA).

**InfiniBand/RoCE**: Standard RDMA using IB Verbs API. Available on GCP with NVIDIA Mellanox ConnectX NICs.

## How It Works

**InfiniBand/RoCE**: Discovers RDMA device IPv4 address for all devices.

**EFA**: Uses the container's IPv4 address directly.

## Infiniband Test

**2 x 8 x H200, multi-node:**

```bash
modal run modal_rdma_bw_ib.py::main
```

### Sample Output
```
[rank 0] Mean BW peak: 734.26 Gb/s, Mean BW avg: 733.92 Gb/s
```
Ensure a mean bi-directional bandwidth close to 800gb/s.

## EFA Test

Ping-pong messages using AWS Elastic Fabric Adapter.

**Usage:**

```bash
modal run modal_pingpong_efa.py
```

### Sample Output
```
[rank 0] Running: /opt/amazon/efa/bin/fi_pingpong -p efa
[rank 1] Running: /opt/amazon/efa/bin/fi_pingpong -p efa 10.100.0.1
[rank 1] bytes   #sent   #ack     total       time     MB/sec    usec/xfer   Mxfers/sec
[rank 1] 64      10      =10      1.2k        0.00s      1.77      36.25       0.03
[rank 1] 256     10      =10      5k          0.00s     14.80      17.30       0.06
[rank 1] 1k      10      =10      20k         0.00s     57.21      17.90       0.06
[rank 1] 4k      10      =10      80k         0.00s    215.58      19.00       0.05
[rank 0] bytes   #sent   #ack     total       time     MB/sec    usec/xfer   Mxfers/sec
[rank 0] 64      10      =10      1.2k        0.00s      1.66      38.50       0.03
[rank 0] 256     10      =10      5k          0.00s     13.26      19.30       0.05
[rank 0] 1k      10      =10      20k         0.00s     52.11      19.65       0.05
[rank 0] 4k      10      =10      80k         0.00s    196.92      20.80       0.05
```

Ensure there are packets being #sent and #ack. 
