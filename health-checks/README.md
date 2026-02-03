# Multi-node Health Checks

Health checking inter-node communication over RoCE v2 and AWS EFA.

## EFA vs InfiniBand

**EFA (Elastic Fabric Adapter)**: AWS-specific RDMA implementation using libfabric API. Available on AWS instances with EFA support.

**InfiniBand/RoCE**: Standard RDMA using IB Verbs API. Available on GCP and other clouds with Mellanox/NVIDIA ConnectX NICs.

## How It Works

**InfiniBand/RoCE**: Discovers RDMA IPv4 addresses by parsing the GID table (`ibv_devinfo`) and extracting RoCE v2 addresses (`::ffff:10.200.0.x`). Ensures connections use the correct RDMA subnet.

**EFA**: Uses Modal's container IPv4 addresses directly with libfabric's EFA provider (`fi_pingpong -p efa`).

## Infiniband Test

**2 x 8 x H200, multi-node:**

```bash
modal run modal_rdma_bw_ib.py::main
```

### Sample Output
```
#bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
65536      5000             740.30             739.87                      1.411195
```
Ensure a peak bi-directional bandwidth near 800gb/s for GCP's ConnectX-7 Mellanox Fabric.

## EFA Test

Ping-pong messages using AWS Elastic Fabric Adapter.

**Usage:**

```bash
modal run modal_pingpong_efa.py
```

### Sample Output
```
bytes   #sent   #ack     total       time     MB/sec    usec/xfer   Mxfers/sec
64      10      =10      1.2k        0.00s      2.52      25.35       0.04
256     10      =10      5k          0.00s     14.38      17.80       0.06
1k      10      =10      20k         0.00s     60.06      17.05       0.06
4k      10      =10      80k         0.00s    222.61      18.40       0.05
```

Ensure bytes are successfully being sent and ACK'd.

## Difference between EFA and IB
