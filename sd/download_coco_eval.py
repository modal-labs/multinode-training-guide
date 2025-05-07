# ---
# deploy: true
# lambda-test: false
# ---
#
# This script demonstrates ingestion of the [COCO](https://cocodataset.org/#download) (Common Objects in Context)
# dataset.
#
# It is recommended to iterate on this code from a modal.Function running Jupyter server.
# This better supports experimentation and maintains state in the face of errors:
# 11_notebooks/jupyter_inside_modal.py
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import modal

volume = modal.Volume.from_name(
    "sd-coco-volumefs1",
    version=1,
)
image = modal.Image.debian_slim().apt_install("wget").pip_install("tqdm")
app = modal.App(
    "example-coco-dataset-import",
    image=image,
)


script_path = pathlib.Path(__file__).resolve()
coco_script_path = script_path.parent / "diffusion/scripts/convert_coco.py"


def start_monitoring_disk_space(interval: int = 120) -> None:
    """Start monitoring the disk space in a separate thread."""
    task_id = os.environ["MODAL_TASK_ID"]

    def log_disk_space(interval: int) -> None:
        while True:
            statvfs = os.statvfs("/")
            free_space = statvfs.f_frsize * statvfs.f_bavail
            print(
                f"{task_id} free disk space: {free_space / (1024**3):.2f} GB",
                file=sys.stderr,
            )
            time.sleep(interval)

    monitoring_thread = threading.Thread(
        target=log_disk_space, args=(interval,)
    )
    monitoring_thread.daemon = True
    monitoring_thread.start()


def extractall(fzip, dest, desc="Extracting"):
    from tqdm.auto import tqdm
    from tqdm.utils import CallbackIOWrapper

    dest = pathlib.Path(dest).expanduser()
    with (
        zipfile.ZipFile(fzip) as zipf,
        tqdm(
            desc=desc,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=sum(getattr(i, "file_size", 0) for i in zipf.infolist()),
        ) as pbar,
    ):
        for i in zipf.infolist():
            if not getattr(i, "file_size", 0):  # directory
                zipf.extract(i, os.fspath(dest))
            else:
                full_path = dest / i.filename
                full_path.parent.mkdir(exist_ok=True, parents=True)
                with zipf.open(i) as fi, open(full_path, "wb") as fo:
                    shutil.copyfileobj(CallbackIOWrapper(pbar.update, fi), fo)


def copy_concurrent(src: pathlib.Path, dest: pathlib.Path) -> None:
    from multiprocessing.pool import ThreadPool

    class MultithreadedCopier:
        def __init__(self, max_threads):
            self.pool = ThreadPool(max_threads)

        def copy(self, source, dest):
            self.pool.apply_async(shutil.copy2, args=(source, dest))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pool.close()
            self.pool.join()

    with MultithreadedCopier(max_threads=48) as copier:
        shutil.copytree(
            src, dest, copy_function=copier.copy, dirs_exist_ok=True
        )


# This script uses wget to download ZIP files over HTTP because while the official
# website recommends using gsutil to download from a bucket (https://cocodataset.org/#download)
# that bucket no longer exists.


@app.function(
    volumes={"/vol/": volume},
    timeout=60 * 60 * 5,  # 5 hours
    ephemeral_disk=600 * 1024,  # 600 GiB,
)
def _do_part(url: str) -> None:
    start_monitoring_disk_space()
    part = url.replace("http://images.cocodataset.org/", "")
    name = pathlib.Path(part).name.replace(".zip", "")
    zip_path = pathlib.Path("/tmp/") / pathlib.Path(part).name
    extract_tmp_path = pathlib.Path("/tmp", name)
    dest_path = pathlib.Path("/vol/coco/", name)

    print(f"Downloading {name} from {url}")
    command = f"wget {url} -O {zip_path}"
    subprocess.run(command, shell=True, check=True)
    print(f"Download of {name} completed successfully.")
    extract_tmp_path.mkdir()
    extractall(
        zip_path, extract_tmp_path, desc=f"Extracting {name}"
    )  # extract into /tmp/
    zip_path.unlink()  # free up disk space by deleting the zip
    print(f"Copying extract {name} data to volume.")
    copy_concurrent(
        extract_tmp_path, dest_path
    )  # copy from /tmp/ into mounted volume


# We can process each part of the dataset in parallel, using a 'parent' Function just to execute
# the map and wait on completion of all children.


@app.function(
    timeout=60 * 60 * 5,  # 5 hours
)
def import_transform_load() -> None:
    # Using 2014 dataset as directed by https://github.com/mosaicml/diffusion/blob/main/scripts/README.md#coco-dataset.
    print("Starting import, transform, and load of COCO dataset 2014")
    files = [
        "http://images.cocodataset.org/zips/val2014.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    ]
    list(
        _do_part.map(files)
    )
    print("✅ Done")


@app.function(
    timeout=60 * 45,
    image=(
      image.apt_install("git")
        .run_commands(
            "git clone https://github.com/mosaicml/diffusion.git",
            "cd diffusion && pip install -e .",
        )  
        .add_local_file(coco_script_path, remote_path="/root/convert_coco.py")
    ),
    volumes={"/vol/": volume},
)
def convert_coco_to_mds() -> None:
    import convert_coco
    remote_path = "/vol/coco/mds/"
    args = convert_coco.args.parse_args([
        "--data_path", "/vol/coco/val2014/",
        "--annotation_path", "/vol/coco/annotations_trainval2014/annotations/captions_val2014.json",
        "--remote", remote_path
    ])
    import shutil
    if os.path.exists(remote_path):
        shutil.rmtree(remote_path)
    os.makedirs(remote_path, exist_ok=True)
    convert_coco.main(args)
