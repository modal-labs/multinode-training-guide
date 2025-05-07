"""Downloads the LAION dataset with `img2dataset`.

On 10/23/24 this was used to download the `aesthetics_v2_4.5` dataset on 128 nodes with 64 CPUs each.

- 1.33 billion URL's, of which ~67% are working
- >78TB of data downloaded
- Peak download speeds of 55,000 images / second
- Completed in ~8 hours
- Cost $7,845
- https://wandb.ai/nathan-modal-labs/laion-img2dataset-fast?nw=nwusernathanmodal

img2dataset did their run on 10 nodes with 16 CPUs each and achieved ~8782 images/second.
See https://wandb.ai/rom1504/img2dataset/reports/Laion2B-en-download--VmlldzoxNTM3MTQ3.

The last few parquet files took a long time to process. It's possible to speed this up further by better
sharding the parquet files or better optimizing `img2dataset`.

It's possible to optimize the cost by requesting fewer CPUs. We were mostly network bound, not compute bound.

`--processes_count` and `--thread_count` are quite sensitive and worth tuning. We found that 64x64 was
fastest, achieving ~10 images/second per worker and ~550 images/second per node. img2dataset reports
~1000 images/second per node, but we weren't able to achieve this.

Note that there is a bug in `img2dataset` that can cause it to hang indefinitely on certain URL's.
If you observe that some containers never exit, even though their parquet file seems mostly complete
and the container isn't using much CPU or network anymore, it's possible you're runnning into this bug.
I experienced this for *one* URL in the dataset, so I just manually killed the container and ignored
the 10,000 URL's in that shard. See https://github.com/rom1504/img2dataset/issues/437 for more details.
"""

import modal
import subprocess
import os

# dataset = "relaion2B-en-research-safe"
# dataset = "laion2B-en-aesthetic"
dataset = "aesthetics_v2_4.5"

vol = modal.Volume.from_name(
    f"laion-{dataset}-img2dataset-fast",
    create_if_missing=True,
    version=2,
)
app = modal.App(
    "laion-download-fast",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/data": vol,
    },
)

hf_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[cli]", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

img2dataset_image = (
    modal.Image.debian_slim()
    .pip_install("img2dataset")
    .apt_install("htop", "iotop", "bwm-ng")
    .pip_install("mosaicml-streaming")
)


@app.function(image=hf_image)
def download_laion(file: str):
    print(f"Downloading {file}...")
    os.makedirs("/data/laion", exist_ok=True)
    out = subprocess.check_output(
        [
            "huggingface-cli",
            "download",
            f"laion/{dataset}",
            file,
            "--repo-type",
            "dataset",
        ]
    )
    file_path = out.decode().strip()
    os.system(f"cp {file_path} /data/laion/{file}")
    vol.commit()
    print(f"Done with {file}")


@app.function(image=hf_image)
def fetch_files_to_download(path: str):
    from huggingface_hub import HfFileSystem

    hf = HfFileSystem()
    files = hf.glob(path + "/*")
    return [file.replace(path + "/", "") for file in files]


# I think the cpu, memory, process count, and thread count settings can be tweaked.
# In particular wandb reports 10 mages / second when img2dataset claims it can
# reach 80 images / second.
@app.function(
    timeout=60 * 60 * 24,
    cpu=64,
    ephemeral_disk=3 * 1024 * 1024,
    image=img2dataset_image,
)
def run_img2dataset(file: str):
    print(f"Running img2dataset on {file}...")
    vol.reload()
    assert os.path.exists(f"/data/laion/{file}"), f"File {file} does not exist"

    part = file.split("-")[1]
    if part == "00010":
        return
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    subprocess.run(
        [
            "img2dataset",
            "--url_list",
            f"/data/laion/{file}",
            "--input_format",
            "parquet",
            "--url_col",
            "URL",
            "--caption_col",
            "TEXT",
            "--output_format",
            "parquet",
            "--output_folder",
            f"/data/img2dataset/{part}",
            "--processes_count",
            "64",
            "--thread_count",
            "64",
            "--resize_mode",
            "no",
            "--save_additional_columns",
            # Note: AESTHETIC_SCORE is "aesthetic" in some datasets.
            "['punsafe','pwatermark','similarity','hash','AESTHETIC_SCORE']",
            "--enable_wandb",
            "True",
            "--wandb_project",
            "laion-img2dataset-fast",
        ],
        check=True,
    )
    vol.commit()
    print(f"Done with {file}")


@app.local_entrypoint()
def main():
    files_to_download = fetch_files_to_download.remote(f"datasets/laion/{dataset}")
    list(download_laion.map(files_to_download))
    print("Done downloading laion repo")

    files_to_download = [file for file in files_to_download if file.startswith("part-")]
    results = list(run_img2dataset.map(files_to_download, return_exceptions=True))
    print("Done with img2dataset")
    print("Results:")
    print(results)
