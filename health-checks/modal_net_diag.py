import os
import subprocess

import modal
import modal.experimental


image = modal.Image.from_registry(
    "slimerl/slime:nightly-dev-20260629a"
).entrypoint([]).apt_install("iproute2")
app = modal.App("modal-net-diag-glm52", image=image)


def run(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, end="", flush=True)
    print(f"[exit {proc.returncode}]", flush=True)


@app.function(
    gpu="H100:8",
    cloud="aws",
    timeout=15 * 60,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(2, rdma=True)
def diag():
    info = modal.experimental.get_cluster_info()
    print(f"rank={info.rank}", flush=True)
    print(f"container_ipv4_ips={info.container_ipv4_ips}", flush=True)
    print(f"container_ips={getattr(info, 'container_ips', None)}", flush=True)
    print(f"task_id={os.environ.get('MODAL_TASK_ID')}", flush=True)

    run("hostname -I || true")
    run("ip -o -4 addr show || true")
    run("ip route || true")
    for ip in info.container_ipv4_ips:
        run(f"ip route get {ip} || true")
    run(
        "for n in /sys/class/net/*; do "
        "i=$(basename $n); "
        "printf '%s mtu=' $i; cat $n/mtu 2>/dev/null; "
        "printf '%s oper=' $i; cat $n/operstate 2>/dev/null; "
        "done"
    )
    run("command -v ucx_info && ucx_info -d | sed -n '1,220p' || true")
    run(
        "UCX_TLS=tcp,cuda_copy,cuda_ipc UCX_NET_DEVICES=all "
        "ucx_info -d 2>&1 | sed -n '1,220p' || true"
    )
    run(
        "UCX_TLS=tcp,cuda_copy,cuda_ipc UCX_NET_DEVICES=eth0 "
        "ucx_info -d 2>&1 | sed -n '1,160p' || true"
    )
    run(
        "UCX_TLS=tcp,cuda_copy,cuda_ipc UCX_NET_DEVICES=overlay0 "
        "ucx_info -d 2>&1 | sed -n '1,160p' || true"
    )
    run("/opt/amazon/efa/bin/fi_info -p tcp 2>/dev/null | head -80 || true")
    run("/opt/amazon/efa/bin/fi_info -p sockets 2>/dev/null | head -80 || true")
    run("FI_PROVIDER=efa /opt/amazon/efa/bin/fi_info -p efa 2>/dev/null | head -80 || true")
    run(
        "python3 - <<'PY'\n"
        "import importlib\n"
        "for name in ('nixl_cu12._api', 'nixl._api'):\n"
        "    try:\n"
        "        mod = importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        print(name, 'IMPORT_ERROR', repr(exc))\n"
        "        continue\n"
        "    print(name, 'OK')\n"
        "    print([x for x in dir(mod) if 'Agent' in x or 'agent' in x or 'backend' in x.lower()][:80])\n"
        "PY"
    )


@app.local_entrypoint()
def main():
    diag.remote()
