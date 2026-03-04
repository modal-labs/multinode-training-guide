import os
import pathlib
import subprocess
import threading

import modal


APP_NAME = "ssh-h100-workspace"
CUDA_TAG = "12.6.0-devel-ubuntu22.04"
LOCAL_SSH_PUBKEY_PATH = pathlib.Path.home() / ".ssh" / "id_ed25519.pub"
LOCAL_SSHD_CONFIG_PATH = pathlib.Path(__file__).resolve().parent / "sshd_config"
REMOTE_WORKSPACE_DIR = "/root/workspace"
VOLUME_ROOT = "/vol"
VOLUME_WORKSPACE_DIR = f"{VOLUME_ROOT}/workspace"
SYNC_INTERVAL_SECONDS = 30


app = modal.App(APP_NAME)
workspace_volume = modal.Volume.from_name(
    "ssh-h100-workspace-volume", create_if_missing=True
)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("openssh-server", "rsync")
    .add_local_file(str(LOCAL_SSHD_CONFIG_PATH), "/etc/ssh/sshd_config", copy=True)
    .run_commands(
        "mkdir -p /root/workspace",
        "mkdir -p /var/run/sshd /root/.ssh",
        "chmod 700 /root/.ssh",
    )
)


def _rsync_directory(src_dir: str, dst_dir: str) -> None:
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude",
            ".git/",
            f"{src_dir.rstrip('/')}/",
            f"{dst_dir.rstrip('/')}/",
        ],
        check=True,
    )


def _initial_sync_from_volume() -> None:
    os.makedirs(VOLUME_WORKSPACE_DIR, exist_ok=True)
    os.makedirs(REMOTE_WORKSPACE_DIR, exist_ok=True)
    workspace_volume.reload()
    _rsync_directory(VOLUME_WORKSPACE_DIR, REMOTE_WORKSPACE_DIR)


def _background_sync_workspace_to_volume(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        _rsync_directory(REMOTE_WORKSPACE_DIR, VOLUME_WORKSPACE_DIR)
        workspace_volume.commit()
        stop_event.wait(SYNC_INTERVAL_SECONDS)


def _configure_ssh_authorized_keys(ssh_public_key: str) -> None:
    key = ssh_public_key.strip()
    if not key:
        raise ValueError("ssh_public_key must be non-empty")

    authorized_keys_path = "/root/.ssh/authorized_keys"
    with open(authorized_keys_path, "w", encoding="utf-8") as f:
        f.write(f"{key}\n")
    os.chmod(authorized_keys_path, 0o600)


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={VOLUME_ROOT: workspace_volume},
)
def ssh_h100_workspace(ssh_public_key: str) -> None:
    _configure_ssh_authorized_keys(ssh_public_key)
    _initial_sync_from_volume()

    stop_event = threading.Event()
    sync_thread = threading.Thread(
        target=_background_sync_workspace_to_volume,
        args=(stop_event,),
        daemon=True,
    )
    sync_thread.start()

    sshd_process = subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    try:
        with modal.forward(22, unencrypted=True) as tunnel:
            host, port = tunnel.tcp_socket
            print(f"SSH available at: {host}:{port}")
            print(f"Connect with: ssh root@{host} -p {port}")
            sshd_process.wait()
    finally:
        stop_event.set()
        sync_thread.join(timeout=5)
        _rsync_directory(REMOTE_WORKSPACE_DIR, VOLUME_WORKSPACE_DIR)
        workspace_volume.commit()
        if sshd_process.poll() is None:
            sshd_process.terminate()


@app.local_entrypoint()
def main(key_path: str = "") -> None:
    key_path_value = key_path.strip() or str(LOCAL_SSH_PUBKEY_PATH)
    local_key_path = pathlib.Path(key_path_value).expanduser()
    if not local_key_path.exists():
        raise ValueError(f"SSH public key file not found: {local_key_path}")
    ssh_public_key = local_key_path.read_text(encoding="utf-8").strip()
    if not ssh_public_key:
        raise ValueError(f"SSH public key file is empty: {local_key_path}")
    ssh_h100_workspace.remote(ssh_public_key=ssh_public_key)
