"""
Script to create subsets when not using full data for train/val
"""
# ------------------------------------------------
# Imports
# ------------------------------------------------
from pathlib import Path
import h5py
import numpy as np
from torchvision.datasets import ImageFolder


from config import (
    IMAGENET_ROOT,
    SEED,
    TRAIN_SUBSET_SIZE,
    VAL_SUBSET_SIZE,
)

TRAIN_SAMPLES = TRAIN_SUBSET_SIZE
VAL_SAMPLES = VAL_SUBSET_SIZE

OUTPUT_DIRECTORY = Path(
    "splits"
)


def create_random_subset(
    dataset_length: int,
    number_of_samples: int,
    rng: np.random.Generator,
):
    if number_of_samples > dataset_length:
        raise ValueError(
            f"Requested {number_of_samples:,} samples, "
            f"but dataset contains only "
            f"{dataset_length:,}."
        )

    return rng.choice(
        dataset_length,
        size=number_of_samples,
        replace=False,
    ).astype(np.int64)


def main():
    train_root = (
        Path(IMAGENET_ROOT)
        / "train"
    )

    val_root = (
        Path(IMAGENET_ROOT)
        / "val"
    )

    # No transform is needed because images are not loaded.
    train_dataset = ImageFolder(
        root=train_root
    )

    val_dataset = ImageFolder(
        root=val_root
    )

    if (
        train_dataset.class_to_idx
        != val_dataset.class_to_idx
    ):
        raise RuntimeError(
            "Train and validation class mappings "
            "do not match."
        )

    print(
        f"Training images available: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation images available: "
        f"{len(val_dataset):,}"
    )

    rng = np.random.default_rng(
        SEED
    )

    train_indices = create_random_subset(
        dataset_length=len(train_dataset),
        number_of_samples=TRAIN_SAMPLES,
        rng=rng,
    )

    val_indices = create_random_subset(
        dataset_length=len(val_dataset),
        number_of_samples=VAL_SAMPLES,
        rng=rng,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_output = (
        OUTPUT_DIRECTORY
        / (
            f"imagenet_train_"
            f"{TRAIN_SAMPLES}_seed{SEED}.npy"
        )
    )

    val_output = (
        OUTPUT_DIRECTORY
        / (
            f"imagenet_val_"
            f"{VAL_SAMPLES}_seed{SEED}.npy"
        )
    )

    np.save(
        train_output,
        train_indices,
    )

    np.save(
        val_output,
        val_indices,
    )

    print(
        f"Saved training indices: "
        f"{train_output}"
    )

    print(
        f"Saved validation indices: "
        f"{val_output}"
    )


if __name__ == "__main__":
    main()