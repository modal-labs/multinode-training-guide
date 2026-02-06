# Multi-node Health Checks

Health checking inter-node communication over EFA and InfiniBand.

## EFA vs InfiniBand

**EFA (Elastic Fabric Adapter)**: Uses Scalable Reliable Datagrams (SRD) as the transport protocol. Ensures delivery but not order.

**InfiniBand**: Uses Reliable Connection (RC) as the transport protocol. Ensures delivery and order.

## How It Works

**InfiniBand**: Discovers the RDMA device's IPv4 address for all devices.

**EFA**: Uses the container's IPv4 address and specifies a device domain.

## InfiniBand Ping-pong Test

**Two 8-GPU H200 nodes**

Ping-pong messages using perftests with verbs as the fabric provider.

```
[rank 1]  #bytes #iterations    t_min[usec]    t_max[usec]
[rank 1]  2       1000          12.12          18.39
[rank 0]  #bytes #iterations    t_min[usec]    t_max[usec]
[rank 0]  2       1000          12.11          18.86
```
## InfiniBand BW Test

**Two 8-GPU H200 nodes**

```bash
modal run modal_bw_ib.py
```

### Sample Output
```
[rank 0] Mean BW peak: 734.26 Gb/s, Mean BW avg: 733.92 Gb/s
```
This InfiniBand test records average bandwidth over 5000 iterations. The underlying network is a rail-optimized [Clos topology](https://docs.nvidia.com/networking-ethernet-software/guides/EVPN-Network-Reference/Introduction/#topology), where each node's GPUs are connected to each corresponding node's GPUs. Each device only needs to hop one network switch to reach its connected device versus ToR topologies where all GPUs must send traffic through a rack-level network switch. You should see speeds near 800 Gb/s, which is the total bidirectional bandwidth per rail. On a 8-GPU node, you get 3.2 Tb/s in one direction.

## EFA Ping-pong Test

**Two 8-GPU H100 nodes**

Ping-pong messages using the libfabric API's EFA provider.

**Usage:**

```bash
modal run modal_pingpong_efa.py
```

### Sample Output
```
[rank 0] Running: /opt/amazon/efa/bin/fi_pingpong -p efa
[rank 1] Running: /opt/amazon/efa/bin/fi_pingpong -p efa 10.100.0.1
[rank 1] bytes   #sent   #ack     total       time     MB/sec
[rank 1] 64      10      =10      1.2k        0.00s      1.77
[rank 1] 256     10      =10      5k          0.00s     14.80
[rank 1] 1k      10      =10      20k         0.00s     57.21
[rank 1] 4k      10      =10      80k         0.00s    215.58
[rank 0] bytes   #sent   #ack     total       time     MB/sec
[rank 0] 64      10      =10      1.2k        0.00s      1.66
[rank 0] 256     10      =10      5k          0.00s     13.26
[rank 0] 1k      10      =10      20k         0.00s     52.11
[rank 0] 4k      10      =10      80k         0.00s    196.92
```

Ensure there are packets being #sent and #ack.

## EFA Bandwidth Test

**Two 8-GPU H200 nodes**

Measures bandwidth over GPU Direct RDMA, where data travels from GPU to NIC without a host memory copy.

**Usage:**

```bash
modal run modal_bw_efa.py
```

### Sample Output
```
[rank 0] Total BW: 278GB/s
```
Expect a total bidirectional bandwidth between 200 and 400 GB/s. EFA also uses a Clos network topology, but it is not rail-optimized. Thus, you will see higher network congestion when analyzing the indiviudal bandwidth across all 16 EFA devices per node.