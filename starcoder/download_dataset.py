import modal
import os
import shutil

DATASET_ID = "bigcode/starcoderdata"

vol = modal.Volume.from_name(
    f"{DATASET_ID.replace('/', '-')}-dataset",
    create_if_missing=True,
    # version=2,
)

DATASET_MOUNT_PATH = "/data"

hf_secret = modal.Secret.from_name("huggingface-token")

app = modal.App(
    f"{DATASET_ID}-download",
)

hf_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "hf_transfer", "datasets", "tqdm")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=hf_image, secrets=[hf_secret])
def get_dataset_files() -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()

    # List all files (and folder entries) under the root of the dataset
    all_paths = api.list_repo_files(repo_id=DATASET_ID, repo_type="dataset")

    return [entry for entry in all_paths if ".parquet" in entry]


@app.function(
    image=hf_image,
    volumes={DATASET_MOUNT_PATH: vol},
)
def clear_dataset_volume():
    print(f"Clearing all contents from {DATASET_MOUNT_PATH}...")
    vol.reload()
    if os.path.exists(DATASET_MOUNT_PATH):
        for item in os.listdir(DATASET_MOUNT_PATH):
            item_path = os.path.join(DATASET_MOUNT_PATH, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print(f"Successfully cleared {DATASET_MOUNT_PATH}.")
    else:
        print(f"{DATASET_MOUNT_PATH} does not exist, nothing to clear.")
    vol.commit()


@app.function(
    image=hf_image,
    secrets=[hf_secret],
    volumes={DATASET_MOUNT_PATH: vol},
    max_containers=50,
)
def download_dataset(file_path: str):
    import os
    from datasets import load_dataset
    import shutil

    print(f"Downloading {file_path} to {DATASET_MOUNT_PATH}/{file_path}")

    # Load dataset from source
    dataset = load_dataset(
        DATASET_ID,
        data_files=f"{file_path}",
    )

    save_path = f"{DATASET_MOUNT_PATH}/{file_path}"
    print(f"Saving {file_path} to {save_path}...")

    vol.reload()

    # Create parent directory
    parent_dir = os.path.dirname(save_path)
    os.makedirs(parent_dir, exist_ok=True)

    # Save to disk
    dataset.save_to_disk(save_path)
    vol.commit()
    print(f"Saved {file_path} to {save_path}.")

    print(f"Finished downloading and saving {file_path}.")
    return file_path


@app.function(timeout=3600, image=hf_image)
def orchestrate():
    from tqdm import tqdm

    file_paths = get_dataset_files.remote()
    print(f"Identified {len(file_paths)} files/folders for dataset.")

    # Clear the volumes before downloading
    print("Clearing V2 volume...")
    clear_dataset_volume.remote()
    print("Dataset volume cleared.")

    # Download to both volumes using map
    print(f"Starting downloads for {len(file_paths)} items to V2...")
    download_results = []
    for result in tqdm(download_dataset.map(file_paths), total=len(file_paths)):
        print(f"Finished download for: {result}")
        download_results.append(result)
    print(f"Done downloading {len(download_results)} items to V2.")

    # Pass the original file_paths list to validation
    print("Starting dataset structure validation on V2...")
    orchestrate_validation.remote(file_paths)

    print("Done with all operations.")


# New function to validate a single dataset directory
@app.function(
    image=hf_image,  # Use the same image with datasets library
    volumes={DATASET_MOUNT_PATH: vol},
    timeout=300,
)
def validate_single_dataset(file_path: str) -> str | None:
    import os
    from datasets import load_from_disk, Dataset, DatasetDict

    load_path = f"{DATASET_MOUNT_PATH}/{file_path}"
    print(f"Validating dataset at: {load_path}...")
    vol.reload()  # Ensure volume contents are up-to-date

    try:
        if not os.path.exists(load_path):
            print(f"Validation Error: Path does not exist: {load_path}")
            return file_path  # Return path if it doesn't exist

        # print(f"Path {load_path} exists. Attempting load_from_disk...")
        loaded_ds = load_from_disk(load_path)

        # Check if it's a valid Dataset or a non-empty DatasetDict
        is_valid_dataset = isinstance(loaded_ds, Dataset) or (
            isinstance(loaded_ds, DatasetDict) and loaded_ds
        )

        if is_valid_dataset:
            print(f"Successfully validated dataset: {load_path}")
            return None  # Success
        else:
            print(
                f"Validation Error: Loaded unexpected type or empty DatasetDict: {type(loaded_ds)} from {load_path}"
            )
            return file_path  # Failure

    except Exception as e:
        print(f"Validation Exception for {load_path}: {e}")
        # Optionally, list directory contents on error for debugging
        # if os.path.exists(DATASET_MOUNT_PATH):
        #     print(f"Contents of {DATASET_MOUNT_PATH}: {os.listdir(DATASET_MOUNT_PATH)}")
        return file_path  # Failure


# New orchestrator for validation
@app.function(timeout=3600)  # Allow ample time for mapping
def orchestrate_validation(file_paths: list[str]):
    print(f"Starting validation orchestration for {len(file_paths)} paths...")

    failed_validations = []

    # Run validation checks using map
    # Each result is None (success) or the failed file_path (str)
    print("Running validation checks...")
    validation_results = validate_single_dataset.map(file_paths)

    # Process results
    for result in validation_results:
        if result is not None:  # result is the file_path on failure
            failed_validations.append(result)

    # Report results
    print("\n--- Validation Results ---")
    if not failed_validations:
        print("All dataset paths loaded and validated successfully!")
    else:
        print(f"Found {len(failed_validations)} paths that failed validation:")
        for f_path in failed_validations:
            print(f"  - {f_path}")
    print("--------------------------")

    print("Dataset validation orchestration finished.")


@app.local_entrypoint()
def main():
    orchestrate.remote()
