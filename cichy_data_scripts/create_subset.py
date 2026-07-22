"""
Script to create subsets when not using full data for train/val
"""
# ------------------------------------------------
# Imports
# ------------------------------------------------
from pathlib import Path
import h5py
import numpy as np

from config import (
    TRAIN_DATA
)

OUTPUT = Path("splits")

TRAIN_SAMPLES = 300_000
VAL_SAMPLES = 10_000

SEED = 42

def create_random_subset(
    dataset_length: int,
    number_of_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if number_of_samples > dataset_length:
        raise ValueError(
            f"Requested {number_of_samples:,} samples, "
            f"but the dataset contains only {dataset_length:,}."
        )

    return rng.choice(
        dataset_length,
        size=number_of_samples,
        replace=False,
    ).astype(np.int64)


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(TRAIN_DATA, "r") as file:
        train_length = len(file["train"]["data"])
        val_length = len(file["val"]["data"])

    print(f"Training images available: {train_length:,}")
    print(f"Validation images available: {val_length:,}")

    rng = np.random.default_rng(SEED)

    train_indices = create_random_subset(
        dataset_length=train_length,
        number_of_samples=TRAIN_SAMPLES,
        rng=rng,
    )

    val_indices = create_random_subset(
        dataset_length=val_length,
        number_of_samples=VAL_SAMPLES,
        rng=rng,
    )

    train_output = (
        OUTPUT
        / f"ecoset_train_{TRAIN_SAMPLES}_seed{SEED}.npy"
    )

    val_output = (
        OUTPUT
        / f"ecoset_val_{VAL_SAMPLES}_seed{SEED}.npy"
    )

    np.save(
        train_output,
        train_indices,
    )

    np.save(
        val_output,
        val_indices,
    )

    print(f"Saved training indices to: {train_output}")
    print(f"Saved validation indices to: {val_output}")

    print(
        f"Training subset: {len(train_indices):,} unique indices"
    )
    print(
        f"Validation subset: {len(val_indices):,} unique indices"
    )


if __name__ == "__main__":
    main()