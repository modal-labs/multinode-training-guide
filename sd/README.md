# Stable Diffusion 2

Code taken from https://github.com/mosaicml/diffusion/tree/b0a094f9209b55659f41fe78764e27bb826863f8 and manually patched to work.

## Dataset

The easiest way to get the training data is to just mount Nathan's existing volume (environment `nathan-dev`). The data already has precomputed latents.

1. In production, the volume is [`laion-aesthetics_v2_4.5-latents`](https://modal.com/storage/modal-labs/nathan-dev/laion-aesthetics_v2_4.5-latents). There are two datasets you can use: `small-256-512` (a smaller ~50GB dataset) and `medium-256-512` (a larger dataset).
2. In dev_cluster, the volume is [`laion-aesthetics_v2_4.5-latents`](https://modal-dev.com/storage/modal-labs/nathan-dev/laion-aesthetics_v2_4.5-latents). There is only one available dataset located at `data`. It is similar in size to the small datast above.

If you want a local copy of the data, you can download it from the volume.

## Configuration

The model configuration is given in `diffusion/yamls/hydra-yamls/SD-2-base-256.yaml`. Refer to that file to examine common flags you may want to evaluate.

Comment out the section on `wandb` if you don't want to configure wandb secrets.

Adjust the number of nodes in `train.py`.

### Running on Modal

```
MODAL_FUNCTION_RUNTIME=runc modal run train.py
```

Use `runc` for optimal performance.

### RDMA

Performance without RDMA is around 2.8-2.95s/ba

```
train 0%|         | 643/550000 [34:15<444:35:17,  2.91ba/s, loss/train/total=0
```

Performance with RDMA is around 4.00ba/s, or about a 35% improvement 🚀.

```
...TODO
```

### Running Locally

There isn't a way to do this yet, but you can probably figure it out by examining what `train.py` is doing.

---

## Appendix

You can stop reading here!

### Check img2dataset output

I found that img2dataset's output was a little buggy: some Parquet files were missing, or were 0 bytes, or were truncated. I manually deleted the corresponding `_stats.json` files and reran img2dataset.

Eventually I gave up and just allowed some parquet files to fail in `convert_to_mds.py`.

### Removing 0 byte files

First, find all 0 byte files:

```
find . -type f -size 0
```

Then, delete them:

```py
import os

# List of bad files
bad_files = [
    "/img2dataset/00003/00004.parquet",
    "./img2dataset/00003/00005.parquet",
    "./img2dataset/00003/00013.parquet",
    "./img2dataset/00003/00019.parquet",
    # ... (add all other paths here)
    "./img2dataset/00124/00590_stats.json",
    "./img2dataset/00127/00255_stats.json"
]

# Process each bad file
for bad_file in bad_files:
    # Normalize file path
    bad_file = os.path.normpath(bad_file)
    
    # Check if the bad file exists
    if os.path.exists(bad_file):
        # Delete the bad file
        os.remove(bad_file)
        print(f"Deleted: {bad_file}")

    # Identify the corresponding pair file (either .parquet or _stats.json)
    if bad_file.endswith(".parquet"):
        pair_file = bad_file.replace(".parquet", "_stats.json")
    elif bad_file.endswith("_stats.json"):
        pair_file = bad_file.replace("_stats.json", ".parquet")
    
    # Check if the pair file exists and delete it
    if os.path.exists(pair_file):
        os.remove(pair_file)
        print(f"Deleted pair file: {pair_file}")
```

### Removing missing pairs

First, find which shards have an odd number of files:

```bash
find . -type d -exec bash -c '(( $(find "$0" -maxdepth 1 -type f | wc -l) % 2 == 0 )) || echo "$0 has an odd number of files"' {} \;
```

Then, for every shard, identify which files are missing:

```bash
find . -type f -name '*.parquet' -o -name '*_stats.json' | sed -E 's/_stats\.json|\.parquet//' | sort | uniq -u
```

Or just do it together (useful if there are an even number of missing files):

```bash
find . -type d -exec sh -c 'find "$1" -type f -name "*.parquet" -o -name "*_stats.json" | sed -E "s/_stats\.json|\.parquet//" | sort | uniq -u' _ {} \;
```

### Small files

```bash
find . -type f -name "*.parquet" -size -30M
```
