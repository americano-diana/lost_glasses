# ------------------------------------------------
# Library imports
# ------------------------------------------------
import json
from datetime import datetime
from pathlib import Path

from PIL import Image
from torchvision import transforms

# ------------------------------------------------
# Local imports
# ------------------------------------------------
from config import (
    CAT_DOG_DATA,
    RAW_DATASET,
    RAW_BLUR_2,
    RAW_BLUR_5,
)

# ------------------------------------------------
# Preprocessing run configuration
# ------------------------------------------------

PREPROCESSED_DATA_ROOT = CAT_DOG_DATA / "preprocessed_data"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Adjust name if needed to fit preprocessing changes
PREPROCESSING_NAME = "simple_resize_256"

PREPROCESSED_RUN_DIR = (
    PREPROCESSED_DATA_ROOT
    / f"{TIMESTAMP}_{PREPROCESSING_NAME}"
)

# Each input dataset receives its own output directory
DATASET_PATHS = {
    "clear": {
        "source": RAW_DATASET,
        "output": PREPROCESSED_RUN_DIR / "clear",
    },
    "blur_2": {
        "source": RAW_BLUR_2,
        "output": PREPROCESSED_RUN_DIR / "blur_2",
    },
    "blur_5": {
        "source": RAW_BLUR_5,
        "output": PREPROCESSED_RUN_DIR / "blur_5",
    },
}


# ------------------------------------------------
# Preprocessing settings
# ------------------------------------------------

IMAGE_SIZE = (256, 256) # Standard AlexNet resizing - might need to change for other networks

save_preprocessing = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
])

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ------------------------------------------------
# Preprocessing functions
# ------------------------------------------------

def preprocess_and_save_dataset(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    """
    Resize and save all images while preserving their directory structure.

    Expected source structure:

        source_dir/
            train/
                class_1/
                class_2/
            test/
                class_1/
                class_2/

    Output structure:

        output_dir/
            train/
                class_1/
                class_2/
            test/
                class_1/
                class_2/

    Parameters
    ----------
    source_dir:
        Root directory containing the raw dataset.

    output_dir:
        Directory in which processed images will be saved.

    overwrite:
        Whether existing processed files should be overwritten.

    Returns
    -------
    dict
        Processing statistics.
    """
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist: {source_dir}"
        )

    image_paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported image files were found in: {source_dir}"
        )

    print(
        f"\nProcessing {len(image_paths)} images "
        f"from {source_dir}"
    )

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for image_path in image_paths:
        relative_path = image_path.relative_to(source_dir)

        # Save all output images as PNG.
        output_path = (
            output_dir
            / relative_path.parent
            / f"{image_path.stem}.png"
        )

        if output_path.exists() and not overwrite:
            skipped_count += 1
            continue

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                processed_image = save_preprocessing(image)

                processed_image.save(
                    output_path,
                    format="PNG",
                )

            processed_count += 1

        except (OSError, ValueError) as error:
            failed_count += 1
            print(
                f"Could not process {image_path}: {error}"
            )

    statistics = {
        "source": str(source_dir),
        "output": str(output_dir),
        "found": len(image_paths),
        "processed": processed_count,
        "skipped": skipped_count,
        "failed": failed_count,
    }

    print(f"Saved:   {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed:  {failed_count}")
    print(f"Output:  {output_dir}")

    return statistics


def save_run_metadata(
    run_directory: Path,
    dataset_statistics: dict,
) -> None:
    """
    Save preprocessing settings and source paths as JSON.
    """
    metadata = {
        "timestamp": TIMESTAMP,
        "preprocessing_name": PREPROCESSING_NAME,
        "image_size": list(IMAGE_SIZE),
        "run_directory": str(run_directory.resolve()),
        "datasets": dataset_statistics,
    }

    metadata_path = run_directory / "preprocessing_metadata.json"

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(f"\nMetadata saved to: {metadata_path}")


def save_latest_run_pointer(
    preprocessing_root: Path,
    run_directory: Path,
) -> None:
    """
    Save the path of the most recent preprocessing run.

    The dataloader module can use this file to locate the latest dataset.
    """
    latest_run_file = preprocessing_root / "latest_run.txt"

    latest_run_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_run_file.write_text(
        str(run_directory.resolve()),
        encoding="utf-8",
    )

    print(f"Latest-run pointer saved to: {latest_run_file}")


def main() -> None:
    """
    Run preprocessing for every configured dataset.
    """
    PREPROCESSED_RUN_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("Preprocessing configuration")
    print("---------------------------")
    print(f"Run directory: {PREPROCESSED_RUN_DIR}")

    dataset_statistics = {}

    for dataset_name, paths in DATASET_PATHS.items():
        print(f"\nDataset: {dataset_name}")
        print(f"Source:  {paths['source']}")
        print(f"Output:  {paths['output']}")

        dataset_statistics[dataset_name] = (
            preprocess_and_save_dataset(
                source_dir=paths["source"],
                output_dir=paths["output"],
            )
        )

    save_run_metadata(
        run_directory=PREPROCESSED_RUN_DIR,
        dataset_statistics=dataset_statistics,
    )

    save_latest_run_pointer(
        preprocessing_root=PREPROCESSED_DATA_ROOT,
        run_directory=PREPROCESSED_RUN_DIR,
    )

    print("\nPreprocessing complete")
    print("----------------------")
    print(f"Version: {PREPROCESSED_RUN_DIR.name}")


if __name__ == "__main__":
    main()