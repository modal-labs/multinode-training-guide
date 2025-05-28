# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastcore==1.5.29",
#     "markdown-it-py==3.0.0",
#     "mdurl==0.1.2",
#     "nvidia-ml-py3==7.352.0",
#     "packaging==23.2",
#     "Pygments==2.17.2",
#     "rich==13.7.0",
# ]
# ///
#
# This script is used to monitor the network and GPU usage of a machine.
# This can be used to debug issues with a running job. Run it with:
#
#     uv run mlx_monitor.py
#
# Thanks to Marcin Zablocki from Oracle for the original script found here:
# https://github.com/MarcinZablocki/mlx_monitor/blob/f8aa8735d866733b18ce531bdf0532e7847c6ea7/monitor.py

import platform
import socket
import fcntl
import struct
import array
from time import sleep

import pynvml as nvidia_smi

from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich import box
from rich.panel import Panel
from fastcore.xtras import sparkline

SIOCETHTOOL = 0x8946
ETHTOOL_GSTRINGS = 0x0000001B
ETHTOOL_GSSET_INFO = 0x00000037
ETHTOOL_GSTATS = 0x0000001D
ETH_SS_STATS = 0x1
ETH_GSTRING_LEN = 32

nvidia_smi.nvmlInit()

gpu_utilization = {}
memory_utilization = {}
deviceCount = nvidia_smi.nvmlDeviceGetCount()

for d in range(deviceCount):
    gpu_utilization[d] = [0] * 20
    memory_utilization[d] = [0] * 20


class Ethtool(object):
    """
    A class for interacting with the ethtool API to retrieve network interface card (NIC) statistics.
    """

    def __init__(self, ifname):
        """
        Initializes an Ethtool object.

        Args:
            ifname (str): The name of the network interface.

        """
        self.ifname = ifname
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)

    def _send_ioctl(self, data):
        """
        Sends an ioctl request to the network interface.

        Args:
            data (bytes): The data to be sent.

        Returns:
            bytes: The response from the ioctl request.

        """
        ifr = struct.pack("16sP", self.ifname.encode("utf-8"), data.buffer_info()[0])
        return fcntl.ioctl(self._sock.fileno(), SIOCETHTOOL, ifr)

    def get_gstringset(self, set_id):
        """
        Retrieves the set of strings associated with a given set ID.

        Args:
            set_id (int): The ID of the set.

        Yields:
            str: The strings associated with the set.

        """
        sset_info = array.array(
            "B", struct.pack("IIQI", ETHTOOL_GSSET_INFO, 0, 1 << set_id, 0)
        )
        self._send_ioctl(sset_info)
        sset_mask, sset_len = struct.unpack("8xQI", sset_info)
        if sset_mask == 0:
            sset_len = 0

        strings = array.array(
            "B", struct.pack("III", ETHTOOL_GSTRINGS, ETH_SS_STATS, sset_len)
        )
        strings.extend(b"\x00" * sset_len * ETH_GSTRING_LEN)
        self._send_ioctl(strings)
        for i in range(sset_len):
            offset = 12 + ETH_GSTRING_LEN * i
            s = (
                strings[offset : offset + ETH_GSTRING_LEN]
                .tobytes()
                .partition(b"\x00")[0]
                .decode("utf-8")
            )
            yield s

    def get_nic_stats(self):
        """
        Retrieves the NIC statistics.

        Yields:
            tuple: A tuple containing the statistic name and its corresponding value.

        """
        strings = list(self.get_gstringset(ETH_SS_STATS))
        n_stats = len(strings)

        stats = array.array("B", struct.pack("II", ETHTOOL_GSTATS, n_stats))
        stats.extend(struct.pack("Q", 0) * n_stats)
        self._send_ioctl(stats)
        for i in range(n_stats):
            offset = 8 + 8 * i
            value = struct.unpack("Q", stats[offset : offset + 8])[0]
            yield (strings[i], value)


def make_layout() -> Layout:
    layout = Layout(name="root")
    if deviceCount > 0:
        layout.split(
            Layout(name="header", size=3),
            Layout(name="gpu", size=11),
            Layout(name="main", ratio=1),
        )
    else:
        layout.split(
            Layout(name="header", size=3),
            Layout(name="gpu", size=4),
            Layout(name="main"),
        )

    return layout


class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        grid = Table.grid()
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right", ratio=1)

        return Panel(grid, box=box.SIMPLE)


class Footer:
    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)

        grid.add_row(
            platform.node(),
        )
        return Panel(grid)


# TODO(pawalt): Get this data dynamically.
ibd = [
    {
        "mlx": "mlx5_0",
        "net": "rdma0",
    },
    {
        "mlx": "mlx5_1",
        "net": "rdma1",
    },
    {
        "mlx": "mlx5_3",
        "net": "rdma2",
    },
    {
        "mlx": "mlx5_4",
        "net": "rdma3",
    },
    {
        "mlx": "mlx5_5",
        "net": "rdma4",
    },
    {
        "mlx": "mlx5_6",
        "net": "rdma5",
    },
    {
        "mlx": "mlx5_7",
        "net": "rdma6",
    },
    {
        "mlx": "mlx5_8",
        "net": "rdma7",
    },
    {
        "mlx": "mlx5_9",
        "net": "rdma8",
    },
    {
        "mlx": "mlx5_10",
        "net": "rdma9",
    },
    {
        "mlx": "mlx5_12",
        "net": "rdma10",
    },
    {
        "mlx": "mlx5_13",
        "net": "rdma11",
    },
    {
        "mlx": "mlx5_14",
        "net": "rdma12",
    },
    {
        "mlx": "mlx5_15",
        "net": "rdma13",
    },
    {
        "mlx": "mlx5_16",
        "net": "rdma14",
    },
    {
        "mlx": "mlx5_17",
        "net": "rdma15",
    },
]
stats = {}

for device in ibd:
    # initialize stats
    d = Ethtool(device["net"])
    ethtool_data = {k: v for k, v in d.get_nic_stats()}
    stats[device["mlx"]] = {}
    stats[device["mlx"]]["rx_bytes_phy"] = [ethtool_data["rx_bytes_phy"]] * 20
    stats[device["mlx"]]["tx_bytes_phy"] = [ethtool_data["tx_bytes_phy"]] * 20


def update_stats():
    for device in ibd:
        d = Ethtool(device["net"])
        ethtool_data = {k: v for k, v in d.get_nic_stats()}

        stats[device["mlx"]]["rx_bytes_phy"].append(ethtool_data["rx_bytes_phy"])
        stats[device["mlx"]]["tx_bytes_phy"].append(ethtool_data["tx_bytes_phy"])
        stats[device["mlx"]]["rx_bytes_phy"].pop(0)
        stats[device["mlx"]]["tx_bytes_phy"].pop(0)
    return stats


def generate_table() -> Table:
    # Generate rich table

    stats = update_stats()

    table = Table(expand=False, box=box.SIMPLE_HEAD, padding=(0, 0, 0, 1))
    table.add_column("Device", justify="left", style="dark_orange", no_wrap=True)
    table.add_column("Net", justify="left", style="dark_orange", no_wrap=True)
    table.add_column("TX", justify="left", min_width=20, max_width=22)
    table.add_column("RX", justify="left", min_width=20, max_width=22)
    table.add_column("Throughput", justify="left", min_width=3)

    # TODO: Fix the list comprehension to be more understandable

    for device in sorted(ibd, key=lambda x: x["net"]):
        table.add_row(
            device["mlx"],
            device["net"],
            sparkline(
                [
                    (
                        stats[device["mlx"]]["rx_bytes_phy"][i]
                        - stats[device["mlx"]]["rx_bytes_phy"][i - 1]
                    )
                    // 1000
                    for i in range(1, len(stats[device["mlx"]]["rx_bytes_phy"]))
                ]
            ),
            sparkline(
                [
                    (
                        stats[device["mlx"]]["tx_bytes_phy"][i]
                        - stats[device["mlx"]]["tx_bytes_phy"][i - 1]
                    )
                    // 1000
                    for i in range(1, len(stats[device["mlx"]]["tx_bytes_phy"]))
                ]
            ),
            str(
                f"{(stats[device['mlx']]['rx_bytes_phy'][-1] - stats[device['mlx']]['rx_bytes_phy'][-2]) / 1000000:.2f} / {(stats[device['mlx']]['tx_bytes_phy'][-1] - stats[device['mlx']]['tx_bytes_phy'][-2]) / 1000000:.2f} Mbps"
            ),
        )
    if len(ibd) == 0:
        table.add_row("No InfiniBand Devices FOUND", "N/A", "N/A", "N/A", "N/A")
    return table


def gpu_table() -> Table:
    # Generate rich table

    table = Table(expand=False, box=box.SIMPLE_HEAD, padding=(0, 0, 0, 1))
    table.add_column("Device", justify="left", style="dark_orange", no_wrap=True)
    table.add_column(
        "GPU Utilization",
        justify="left",
    )
    table.add_column(
        "GPU %",
        justify="left",
    )
    table.add_column(
        "MEM %",
        justify="left",
    )
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        gpu_utilization[i].append(nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu)
        mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        memory_utilization = mem_info.used / mem_info.total * 100
        gpu_utilization[i].pop(0)
        power_management_limit = int(
            nvidia_smi.nvmlDeviceGetPowerManagementLimit(
                nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            )
            / 1000
        )
        power_usage = int(
            nvidia_smi.nvmlDeviceGetPowerUsage(nvidia_smi.nvmlDeviceGetHandleByIndex(i))
            / 1000
        )
        table.add_row(
            f"GPU {i} ({nvidia_smi.nvmlDeviceGetName(handle)})",
            sparkline(gpu_utilization[i]),
            f"{gpu_utilization[i][-1]}%",
            f"{memory_utilization:.0f}%"
            f" ({mem_info.used // 1024**2} / {mem_info.total // 1024**2} MB)"
            f" (Busy: {nvidia_smi.nvmlDeviceGetUtilizationRates(nvidia_smi.nvmlDeviceGetHandleByIndex(i)).memory}%)"
            f" (Temp: {nvidia_smi.nvmlDeviceGetTemperature(nvidia_smi.nvmlDeviceGetHandleByIndex(i), nvidia_smi.NVML_TEMPERATURE_GPU)}C)"
            f" (Power: {power_usage}W / {power_management_limit}W)",
        )

    if deviceCount == 0:
        table.add_row("No GPUs FOUND", "N/A", "N/A", "N/A")

    return table


layout = make_layout()
layout["header"].update(Header())
layout["main"].update(generate_table())
layout["gpu"].update(gpu_table())

with Live(layout, refresh_per_second=10, screen=True):
    while True:
        layout["main"].update(generate_table())
        if deviceCount > 0:
            layout["gpu"].update(gpu_table())
        sleep(0.1)
