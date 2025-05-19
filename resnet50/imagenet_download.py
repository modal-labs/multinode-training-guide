import os
from imagenet_classes import IMAGENET2012_CLASS_INDEXES
import modal

app = modal.App(
    "imagenet-download",
    image=(
        modal.Image.debian_slim()
        .pip_install("webdataset", "huggingface_hub[cli]", "hf_transfer")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .add_local_python_source("imagenet_classes")
    ),
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={
        "/data": modal.Volume.from_name("imagenet", create_if_missing=True),
    },
)


def imagenet_samples(dir: str, has_cls: bool = True):
    for root, dirs, files in os.walk(dir, topdown=False):
        for fname in files:
            if not fname.endswith(".JPEG"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "rb") as stream:
                binary_data = stream.read()

            sample = {
                "__key__": os.path.splitext(fname)[0],
                "jpg": binary_data,
            }

            if has_cls:
                _, synset_id = os.path.splitext(fname)[0].rsplit("_", 1)
                sample["cls"] = IMAGENET2012_CLASS_INDEXES[synset_id]

            yield sample


@app.function(timeout=60 * 60 * 24)
def download_imagenet(split: str, file: str, start_shard: int):
    import webdataset as wds
    assert split in ["train", "val", "test"]

    os.makedirs(f"/data/{split}", exist_ok=True)
    os.makedirs("/tmp/out", exist_ok=True)

    # 1. Download the dataset
    print(f"Downloading {file}...")
    os.system(
        f"huggingface-cli download ILSVRC/imagenet-1k data/{file} --repo-type dataset --local-dir /data"
    )

    # 2. Unpack
    print(f"Extracting {file}...")
    os.system(f"tar -xf /data/data/{file} -C /tmp/out --no-same-owner")

    # 3. Create shards
    print(f"Creating shards for {file}...")
    with wds.ShardWriter(
        f"/data/{split}/%06d.tar", maxcount=10000, start_shard=start_shard
    ) as sink:
        for sample in imagenet_samples("/tmp/out", has_cls=split != "test"):
            sink.write(sample)

    # 4. Create indexes
    # print(f"Creating indexes for {file}...")
    # todo fix: only create indexes for files we made
    # os.system(f'for file in /data/{split}/*.tar; do wds2idx "$file"; done')

    print(f"Done with {file}")


@app.local_entrypoint()
def main():
    data_to_download = {
        "train": [
            "train_images_0.tar.gz",
            "train_images_1.tar.gz",
            "train_images_2.tar.gz",
            "train_images_3.tar.gz",
            "train_images_4.tar.gz",
        ],
        "val": [
            "val_images.tar.gz",
        ],
        "test": [
            "test_images.tar.gz",
        ],
    }

    tasks = []
    for split in data_to_download.keys():
        for i, file in enumerate(data_to_download[split]):
            start_shard = i * 10000
            tasks.append((split, file, start_shard))

    list(download_imagenet.starmap(tasks))
