import modal
import os

mds_vol = modal.Volume.from_name("laion-aesthetics_v2_4.5-mds", version=2)
latents_vol = modal.Volume.from_name("laion-aesthetics_v2_4.5-latents", version=2, create_if_missing=True)

app = modal.App(
    name="laion-latents",
    image=(
        modal.Image.debian_slim()
        .apt_install("git")
        .run_commands(
            "git clone https://github.com/mosaicml/diffusion.git",
            "cd diffusion && pip install -e .",
        )
    ),
    volumes={
        "/data": mds_vol,
        "/latents": latents_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)


@app.function(
    mounts=[modal.Mount.from_local_dir("diffusion", remote_path="/root/diffusion")],
    gpu="H100:8",
    cpu=32,
    timeout=60 * 60 * 24,
)
def precompute_latents(src_dir: str, dst_dir: str) -> None:
    # Note: with H100's, utilization is not fully saturated. Consider increasing
    # batch size.
    os.system(
        "composer /root/diffusion/scripts/precompute_latents.py "
        "--local /tmp/mds-cache "
        f"--remote_download {src_dir} "
        f"--remote_upload {dst_dir} "
        "--wandb_name precompute-latents"
    )

    import time
    # Give time for volume to update. not sure if this does anything lol
    time.sleep(10)


# TODO: deduplicate this code with convert_to_mds.py
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
    
    latents_vol.commit()

    for old_filename, new_filename in renames:
        if not os.path.exists(new_filename):
            os.link(old_filename, new_filename)

    latents_vol.commit()
    import time
    # Give time for volume to update. not sure if this does anything lol
    time.sleep(10)


@app.local_entrypoint()
def main():
    # precompute_latents.remote(
    #     "/data/256-512",
    #     "/latents/tmp/medium-256-512",
    # )
    list(verify_shard.map(f"/latents/tmp/medium-256-512/{i}" for i in range(8)))

    aggregate_mds_shards.remote(
        "/latents/tmp/medium-256-512",
        "/latents/medium-256-512",
    )
    verify_shard.remote("/latents/medium-256-512")
