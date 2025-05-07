"""Converts img2dataset shards into MDS shards.

Logic for converting img2dataset shards into MDS shards taken from:
https://github.com/mosaicml/diffusion/blob/main/scripts/laion_cloudwriter.py

Logic for combining multiple MDS output files into one taken from:
https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/pile.py

Warning: this script does _not_ support concurrent workers running the same
input at the same time, which Modal could potentially do if the worker
disappears.
"""
from typing import Optional, Union
import modal
import os

import numpy as np

dataset = "aesthetics_v2_4.5"

img2dataset_vol = modal.Volume.from_name(
    f"laion-{dataset}-img2dataset-fast",
    version=2,
)
mds_vol = modal.Volume.from_name(
    f"laion-{dataset}-mds",
    create_if_missing=True,
    version=2,
)
app = modal.App(
    "laion-convert-to-mds",
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/img2dataset": img2dataset_vol,
        "/mds": mds_vol,
    },
)
mds_conversion_failure_queue = modal.Queue.from_name(f"laion-{dataset}-mds-conversion-failures", create_if_missing=True)

bin_resolutions = [0, 64, 128, 256, 512, 768, 1024, 2**20]

@app.function(image=modal.Image.debian_slim().pip_install("numpy"))
def get_shards():
    import os

    shards = os.listdir("/img2dataset/img2dataset")
    return shards


def get_int(x: Union[float, int]) -> int:
    """Get an int field from pandas.

    Args:
        x (Union[float, int]): The pandas field.

    Returns:
        int: The normalized field.
    """
    if np.isnan(x):
        return 0
    else:
        return int(x)


def get_float(x: float) -> float:
    """Get a float field from pandas.

    Args:
        x (float): The pandas field.

    Returns:
        float: The normalized field.
    """
    return x


def get_bytes(x: Optional[bytes]) -> bytes:
    """Get a bytes field from pandas.

    Args:
        x (bytes, optional): The pandas field.

    Returns:
        float: The normalized field.
    """
    return x or b""


def get_str(x: Optional[str]) -> str:
    """Get a str field from pandas.

    Args:
        x (str, optional): The pandas field.

    Returns:
        str: The normalized field.
    """
    return x or ""


def process_parquet(writer, parquet_filename: str, lower_res: int, upper_res: int) -> bool:
    """Process a parquet file and upload to MDS."""
    import pyarrow.parquet as pq
    from PIL import Image
    from io import BytesIO

    # Open parquet file
    try:
        table = pq.read_table(parquet_filename)
    except Exception as e:
        print(f"Error reading {parquet_filename}: {e}")
        return False

    n_rows = table.num_rows
    table = table.to_pandas()

    # Iterate through rows of parquet file
    for i in range(n_rows):
        x = table.iloc[i]

        # Only write samples that were successfully downloaded
        success = x["status"] == "success"
        width, height = get_int(x["width"]), get_int(x["height"])
        if success:
            try:
                img = Image.open(BytesIO(x["jpg"]))
                width, height = img.size
            except Exception as e:
                print(e)
                # if unable to decode image, set success to false
                success = False
        success &= lower_res <= min(width, height) < upper_res
        if success:
            sample = {
                "punsafe": get_float(x["punsafe"]),
                "pwatermark": get_float(x["pwatermark"]),
                "similarity": get_float(x["similarity"]),
                "caption": get_str(x["caption"]),
                "url": get_str(x["url"]),
                "key": get_str(x["key"]),
                "status": get_str(x["status"]),
                "error_message": get_str(x["error_message"]),
                "width": get_int(x["width"]),
                "height": get_int(x["height"]),
                "original_width": get_int(x["original_width"]),
                "original_height": get_int(x["original_height"]),
                "exif": get_str(x["exif"]),
                "jpg": get_bytes(x["jpg"]),
                "hash": get_int(x["hash"]),
                "aesthetic_score": get_float(x["AESTHETIC_SCORE"])
                if "AESTHETIC_SCORE" in x.keys()
                else 0.0,
            }
            writer.write(sample)
    
    return True


@app.function(image=modal.Image.debian_slim().pip_install("numpy"), timeout=6000)
def verify_shard(shard_path: str):
    import json

    index_filename = os.path.join(shard_path, "index.json")
    try:
        obj = json.load(open(index_filename))
    except Exception as e:
        print(f"Error reading {index_filename}: {e}")
        return False

    for info in obj['shards']:
        basename = info['raw_data']['basename']

        filename = os.path.join(shard_path, basename)

        if not os.path.exists(filename):
            print(f"ERROR: {filename} does not exist")
            return False
        else:
            filesize = os.path.getsize(filename)
            if filesize != int(info['raw_data']['bytes']):
                print(f"ERROR: {filename} has size {filesize} but {info['raw_data']['bytes']} in index.json")
                return False
    return True



@app.function(
    timeout=60 * 60 * 24,
    cpu=2,
    image=modal.Image.debian_slim()
    .pip_install("mosaicml-streaming")
    .pip_install("wandb", "pyarrow", "pandas", "tqdm")
    .pip_install("numpy"),
)
def convert_to_mds(shard: str, bucket_id: int):
    from streaming import MDSWriter
    from tqdm import tqdm
    import shutil

    columns = {
        "punsafe": "float64",
        "pwatermark": "float64",
        "similarity": "float64",
        "caption": "str",
        "url": "str",
        "key": "str",
        "status": "str",
        "error_message": "str",
        "width": "int32",
        "height": "int32",
        "original_width": "int32",
        "original_height": "int32",
        "exif": "str",
        "jpg": "bytes",
        "hash": "int64",
        "aesthetic_score": "float64",
    }

    assert 0 < bucket_id < len(bin_resolutions) - 1
    lower_res, upper_res = bin_resolutions[bucket_id], bin_resolutions[bucket_id + 1]

    target_path = os.path.join("/mds", "tmp", f"{lower_res}-{upper_res}", shard)

    if os.path.exists(os.path.join(target_path, "index.json")):
        if verify_shard.local(target_path):
            # print(f"Skipping {target_path} because it already exists")
            return
        else:
            print(f"Removing {target_path} because it is corrupt")
    else:
        print(f"Missing {target_path}")
    
    if os.path.exists(target_path):
        shutil.rmtree(target_path, ignore_errors=True)

    print(f"Starting uploader processs for {target_path}")
    writer = MDSWriter(
        out=target_path,
        columns=columns,
        compression=None,
        hashes=[],
        # Note: You have to increase this by a lot to get this to work with volumefs1,
        # which has an upper limit of the number of files it can have.
        size_limit=1024 * (2**20),
        max_workers=64,
    )

    subshards_to_process = [
        file.split("_")[0]
        for file in os.listdir(os.path.join("/img2dataset", "img2dataset", shard))
        if file.endswith("_stats.json")
    ]
    n_failures = 0
    for subshard in tqdm(subshards_to_process):
        if not process_parquet(
            writer,
            os.path.join("/img2dataset", "img2dataset", shard, f"{subshard}.parquet"),
            lower_res,
            upper_res,
        ):
            n_failures += 1

    writer.finish()

    os.sync()

    print(f"Finished uploader process for {target_path} with {n_failures} failures")

    # maybe add a time.sleep here to let mds finish? not sure why else we're seeing
    # missing files.

    mds_vol.commit()

@app.function(image=modal.Image.debian_slim().pip_install("numpy"), timeout=6000)
def aggregate_mds_shards(src_dir: str, dst_dir: str):
    import json

    def with_id(basename: str, shard_id: int) -> str:
        """Get a new basename with the given shard_id.

        Args:
            basename (str): Old basename of file.
            shard_id (int): New shard ID.

        Returns:
            str: New basename of file.
        """
        parts = basename.split('.')
        parts[1] = f'{shard_id:05}'
        return '.'.join(parts)

    shard_id = 0
    infos = []
    renames = []
    for shard in os.listdir(src_dir):
        index_filename = os.path.join(src_dir, shard, "index.json")
        obj = json.load(open(index_filename))

        for info in obj['shards']:
            old_basename = info['raw_data']['basename']
            new_basename = with_id(old_basename, shard_id)
            info['raw_data']['basename'] = new_basename

            old_filename = os.path.join(src_dir, shard, old_basename)
            new_filename = os.path.join(dst_dir, new_basename)
            renames.append((old_filename, new_filename))

            if not os.path.exists(old_filename):
                print(f"ERROR: {old_filename} does not exist")
            else:
                filesize = os.path.getsize(old_filename)
                if filesize != int(info['raw_data']['bytes']):
                    print(f"ERROR: {old_filename} has size {filesize} but {info['raw_data']['bytes']} in index.json")

            shard_id += 1
            infos.append(info)


    os.makedirs(dst_dir, exist_ok=True)

    index_filename = os.path.join(dst_dir, 'index.json')
    obj = {
        'version': 2,
        'shards': infos,
    }
    text = json.dumps(obj, sort_keys=True)
    with open(index_filename, 'w') as out:
        out.write(text)
    os.sync()
    
    mds_vol.commit()

    for old_filename, new_filename in renames:
        if not os.path.exists(new_filename):
            os.link(old_filename, new_filename)

    mds_vol.commit()


@app.function(image=modal.Image.debian_slim().pip_install("numpy"), timeout=6000)
def remove_tmp():
    import shutil
    shutil.rmtree("/mds/tmp")


@app.local_entrypoint()
def main():
    shards = get_shards.remote()
    print(f"Found {len(shards)} shards")

    tasks = [(shard, i) for shard in shards for i in range(1, len(bin_resolutions) - 1)]
    print(list(convert_to_mds.starmap(tasks, return_exceptions=True)))
    print("Done converting to MDS")


    handles = []
    for lower_res, upper_res in zip(bin_resolutions[1:], bin_resolutions[2:]):
        print(f"Starting aggregation for {lower_res}-{upper_res}")
        handles.append(aggregate_mds_shards.spawn(
            os.path.join("/mds", "tmp", f"{lower_res}-{upper_res}"),
            os.path.join("/mds", f"{lower_res}-{upper_res}"),
        ))

    for i, handle in enumerate(handles):
        handle.get()
        print(f"Done aggregating {i + 1} of {len(handles)}")


    handles = []
    for lower_res, upper_res in zip(bin_resolutions[1:], bin_resolutions[2:]):
        print(f"Checking {lower_res}-{upper_res}")
        handles.append(verify_shard.spawn(
            os.path.join("/mds", f"{lower_res}-{upper_res}"),
        ))

    for i, handle in enumerate(handles):
        handle.get()
        print(f"Done checking {i + 1} of {len(handles)}")
    
    remove_tmp.remote()
