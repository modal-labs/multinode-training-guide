import modal
import os
import shutil
from typing import Union
from collections import defaultdict

from common import DATASET_ID, DATASET_VOLUME_NAME, DATASET_MOUNT_PATH

vol = modal.Volume.from_name(
    DATASET_VOLUME_NAME,
    create_if_missing=True,
)


hf_secret = modal.Secret.from_name("huggingface-token")

app = modal.App(
    f"{DATASET_ID}-download",
)

hf_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "hf_transfer", "datasets")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("common")
)


@app.function(image=hf_image, secrets=[hf_secret])
def get_dataset_files() -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()

    # List all files (and folder entries) under the root of the dataset
    all_paths = api.list_repo_files(repo_id=DATASET_ID, repo_type="dataset")

    # These have different columns, so they break our training runs
    excluded_files = [
        "jupyter-scripts-dedup-filtered",
        "jupyter-structured-clean-dedup",
        "github-issues-filtered-structured",
        "git-commits-cleaned",
    ]

    return [
        entry
        for entry in all_paths
        if ".parquet" in entry
        and not any(excluded in entry for excluded in excluded_files)
    ]


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
    """Download a single file from the dataset and save it to the volume."""
    import os
    from datasets import load_dataset

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


ValidationResponse = Union[int, str]


@app.function(
    image=hf_image,
    volumes={DATASET_MOUNT_PATH: vol},
    timeout=300,
)
def validate_single_dataset(file_path: str) -> ValidationResponse:
    """Validate a single dataset by loading it and checking the length of the train split.

    Returns the number of samples in the train split if validation passes, otherwise returns the file path.
    """
    import os
    from datasets import load_from_disk

    load_path = f"{DATASET_MOUNT_PATH}/{file_path}"
    vol.reload()

    if not os.path.exists(load_path):
        print(f"Validation Error ({file_path}): Path does not exist.")
        return file_path

    try:
        loaded_data = load_from_disk(load_path)

        train_split = loaded_data["train"]

        if len(train_split) == 0:
            print(f"Validation Error ({file_path}): 'train' split is empty.")
            return file_path

        return len(train_split)

    except Exception as e:
        print(f"Validation Exception ({file_path}): {e}")
        return file_path


@app.function(
    image=hf_image,
    volumes={DATASET_MOUNT_PATH: vol},
    timeout=60,
)
def write_metadata_file(total_samples: int):
    import os

    metadata_file_path = os.path.join(DATASET_MOUNT_PATH, "dataset_length.txt")
    print(f"Writing dataset length ({total_samples}) to {metadata_file_path}...")
    vol.reload()
    try:
        with open(metadata_file_path, "w") as f:
            f.write(str(total_samples))
        vol.commit()
        print(f"Successfully wrote {metadata_file_path}.")
    except Exception as e:
        print(f"Error writing metadata file {metadata_file_path}: {e}")


# Helper function to determine the top-level directory for aggregation
def _get_top_level_directory(file_path: str) -> str:
    import os

    parent_dir = os.path.dirname(file_path)

    if not parent_dir or parent_dir == ".":
        return "<root_dataset_dir>"

    normalized_path_parts = parent_dir.strip(os.sep).split(os.sep)

    if not normalized_path_parts or not normalized_path_parts[0]:
        return "<root_dataset_dir>"

    return normalized_path_parts[0]


@app.function(
    image=hf_image,
    timeout=60 * 60 * 24,
)
def orchestrate_validation(file_paths: list[str]):
    """Validate each dataset by loading it and checking the length of the train split.

    This helper function also records the number of samples in each dataset and the number of failed validations.
    """

    print(f"Starting validation orchestration for {len(file_paths)} paths...")

    # For overall aggregation
    successful_lengths = []
    overall_failed_validation_paths = []

    # For per-directory aggregation
    dir_sample_counts = defaultdict(int)
    dir_failed_files = defaultdict(list)

    # Run validation checks using map
    # Each result is int (length) or the failed file_path (str)
    print("Running validation checks...")
    validation_results_iterator = validate_single_dataset.map(file_paths)

    # Process results to populate overall and per-directory aggregates
    for i, result in enumerate(validation_results_iterator):
        original_path = file_paths[i]

        directory = _get_top_level_directory(original_path)

        if isinstance(result, int):
            successful_lengths.append(result)
            dir_sample_counts[directory] += result
        elif isinstance(result, str):
            overall_failed_validation_paths.append(result)
            dir_failed_files[directory].append(result)

    # Report overall results
    print("\n--- Overall Validation Summary ---")
    if not overall_failed_validation_paths:
        total_samples = sum(successful_lengths)
        summary_message = (
            f"All {len(file_paths)} dataset parts loaded and validated successfully!\n"
            f"Total number of samples: {total_samples}"
        )
        print(summary_message)
        write_metadata_file.remote(total_samples)
    else:
        failed_paths_str = "\n".join(
            [f"  - {f_path}" for f_path in overall_failed_validation_paths]
        )
        summary_message = (
            f"Found {len(overall_failed_validation_paths)} paths that failed validation (or were empty):\n"
            f"{failed_paths_str}"
        )
        if successful_lengths:
            summary_message += (
                f"\n{len(successful_lengths)} paths were validated successfully (with >0 samples), "
                f"contributing a total of {sum(successful_lengths)} samples.\n"
                f"Out of {len(file_paths)} total paths, {len(overall_failed_validation_paths)} failed or were empty."
            )
        else:
            summary_message += (
                f"\nNo paths were validated successfully with >0 samples. "
                f"All {len(file_paths)} processed paths failed or were empty."
            )
        print(summary_message)
    print("--------------------------")

    # Report per-directory results
    print("\n--- Per-Directory Row Counts & Validation ---")
    all_involved_dirs = sorted(
        list(set(dir_sample_counts.keys()) | set(dir_failed_files.keys()))
    )

    if not all_involved_dirs:
        if file_paths:
            print(
                "No samples found and no specific directory failures reported. This might mean all files failed validation in a way not captured per directory, or all datasets were empty."
            )
        else:
            print("No files/directories were provided for validation.")
    else:
        for directory in all_involved_dirs:
            dir_summary_lines = [f"\nDirectory: {directory}"]

            samples_in_dir = dir_sample_counts.get(directory, 0)
            failed_files_in_dir = dir_failed_files.get(directory, [])

            dir_summary_lines.append(
                f"  Total row count (validated samples): {samples_in_dir}"
            )

            if failed_files_in_dir:
                dir_summary_lines.append(
                    f"  Failed or empty files ({len(failed_files_in_dir)}):"
                )
                for f_path in failed_files_in_dir:
                    dir_summary_lines.append(f"    - {f_path}")
            elif samples_in_dir > 0:
                dir_summary_lines.append(
                    f"  All processed files in this directory validated successfully with samples."
                )
            print("\n".join(dir_summary_lines))

    print("==========================================")
    print("Dataset validation orchestration finished.")


@app.function(
    image=hf_image,
    timeout=60 * 60 * 24,
)
def ingest_dataset():
    """Ingest the dataset into the volume by downloading the files and validating them."""

    file_paths = get_dataset_files.remote()
    print(f"Identified {len(file_paths)} files/folders for dataset.")

    # Clear the volumes before downloading
    print("Clearing volume...")
    clear_dataset_volume.remote()

    # Download to both volumes using map
    print(f"Starting downloads for {len(file_paths)} items...")
    download_results = []
    for result in download_dataset.map(file_paths):
        download_results.append(result)
    print(f"Done downloading {len(download_results)} items.")

    # Pass the original file_paths list to validation
    print("Starting dataset structure validation...")
    orchestrate_validation.remote(file_paths)

    print("Done with all operations.")
