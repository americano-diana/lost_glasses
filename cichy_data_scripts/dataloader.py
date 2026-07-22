"""
dataloader.py

Lazy HDF5 dataloader for Ecoset classification.

Training conditions:
    clear:
        Standard clear-image training.

    mixed:
        Each training image has a configurable probability of being blurred.

The same saved subset indices are used for both conditions so that the
underlying training and validation images remain identical.
"""

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from config import (
    TRAIN_DATA,
    TRAIN_INDICES_PATH,
    VAL_INDICES_PATH,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_VAL,
    NUM_WORKERS,
    PIN_MEMORY,
    IMAGE_SIZE,
    MEAN,
    STD,
    BLUR_PROBABILITY,
    BLUR_KERNEL_SIZE,
    BLUR_SIGMA_MIN,
    BLUR_SIGMA_MAX,
    SEED,
)


TrainingCondition = Literal["clear", "blur", "mixed"]


def preprocess_images(
    split: str,
    condition: TrainingCondition = "clear",
):
    """
    Create image preprocessing for Ecoset.

    Conditions
    ----------
    clear:
        Training images remain clear.

    mixed:
        Each training image is blurred with probability
        BLUR_PROBABILITY.

    blur:
        Every training image is blurred.

    Validation and test images remain clear for all conditions.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Unknown split '{split}'. "
            "Expected 'train', 'val', or 'test'."
        )

    if condition not in {"clear", "mixed", "blur"}:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            "Expected 'clear', 'mixed', or 'blur'."
        )

    operations = []

    if split == "train":
        operations.extend([
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
        ])

        blur_transform = transforms.GaussianBlur(
            kernel_size=BLUR_KERNEL_SIZE,
            sigma=(
                BLUR_SIGMA_MIN,
                BLUR_SIGMA_MAX,
            ),
        )

        if condition == "mixed":
            operations.append(
                transforms.RandomApply(
                    [blur_transform],
                    p=BLUR_PROBABILITY,
                )
            )

        elif condition == "blur":
            operations.append(
                blur_transform
            )

    else:
        operations.append(
            transforms.CenterCrop(IMAGE_SIZE)
        )

    operations.append(
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        )
    )

    return transforms.Compose(operations)

class EcosetDataset(Dataset):
    """
    Lazy loader for the Ecoset HDF5 file.

    The HDF5 file is opened separately inside each DataLoader worker.
    This avoids reopening the file for every image while remaining safer
    with multiprocessing.
    """

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        condition: TrainingCondition = "clear",
    ):
        self.h5_path = str(h5_path)
        self.split = split
        self.condition = condition

        self.transform = preprocess_images(
            split=split,
            condition=condition,
        )

        # These are opened lazily inside each worker.
        self._h5_file = None
        self._images = None
        self._labels = None

        with h5py.File(self.h5_path, "r") as file:
            if split not in file:
                raise KeyError(
                    f"Split '{split}' not found in {self.h5_path}. "
                    f"Available splits: {list(file.keys())}"
                )

            split_group = file[split]

            if "data" not in split_group:
                raise KeyError(
                    f"'data' not found under split '{split}'. "
                    f"Available keys: {list(split_group.keys())}"
                )

            if "labels" not in split_group:
                raise KeyError(
                    f"'labels' not found under split '{split}'. "
                    f"Available keys: {list(split_group.keys())}"
                )

            self.length = len(split_group["data"])

            if len(split_group["labels"]) != self.length:
                raise ValueError(
                    f"Image and label counts differ for split '{split}': "
                    f"{self.length} images versus "
                    f"{len(split_group['labels'])} labels."
                )

    def _open_h5(self):
        """
        Open the HDF5 file lazily.

        Each DataLoader worker receives its own dataset instance and
        therefore opens its own read-only file handle.
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(
                self.h5_path,
                mode="r",
                swmr=True,
            )

            self._images = self._h5_file[self.split]["data"]
            self._labels = self._h5_file[self.split]["labels"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._open_h5()

        image = np.asarray(self._images[idx])
        label = int(self._labels[idx])

        image = torch.from_numpy(image)

        # Ecoset images are stored as HWC: [height, width, channels].
        if image.ndim != 3:
            raise ValueError(
                f"Expected a 3D image at index {idx}, "
                f"but found shape {tuple(image.shape)}."
            )

        if image.shape[-1] in {1, 3, 4}:
            image = image.permute(2, 0, 1)

        # Remove an alpha channel if present.
        if image.shape[0] == 4:
            image = image[:3]

        # Repeat grayscale images across RGB channels.
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        image = image.float() / 255.0
        image = self.transform(image)

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image, label

    def close(self):
        """Close the HDF5 handle if it is open."""
        if self._h5_file is not None:
            self._h5_file.close()

            self._h5_file = None
            self._images = None
            self._labels = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def seed_worker(worker_id):
    """
    Give each DataLoader worker a reproducible NumPy seed.

    PyTorch already assigns a different initial seed to every worker.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def load_subset_indices(
    indices_path: str | Path,
    dataset_length: int,
) -> np.ndarray:
    """
    Load and validate saved subset indices.
    """
    indices_path = Path(indices_path)

    if not indices_path.exists():
        raise FileNotFoundError(
            f"Subset index file not found: {indices_path}"
        )

    indices = np.load(indices_path).astype(np.int64)

    if indices.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional indices in {indices_path}, "
            f"but found shape {indices.shape}."
        )

    if len(indices) == 0:
        raise ValueError(
            f"No indices were found in {indices_path}."
        )

    if indices.min() < 0 or indices.max() >= dataset_length:
        raise IndexError(
            f"Indices in {indices_path} are outside the dataset range. "
            f"Valid range: 0 to {dataset_length - 1}. "
            f"Observed range: {indices.min()} to {indices.max()}."
        )

    if len(np.unique(indices)) != len(indices):
        raise ValueError(
            f"Duplicate indices found in {indices_path}."
        )

    return indices


def create_dataloaders(
    condition: TrainingCondition,
):
    """
    Create Ecoset training and validation DataLoaders.

    Parameters
    ----------
    condition:
        "clear":
            Train only on clear images.

        "mixed":
            Train on a dynamic mixture of clear and blurred images.

    Returns
    -------
    train_loader:
        Shuffled training DataLoader.

    val_loader:
        Deterministic clear-image validation DataLoader.
    """
    if condition not in {"clear", "mixed", "blur"}:
        raise ValueError(
            "condition must be 'clear', 'mixed', or 'blur'."
        )

    train_dataset = EcosetDataset(
        h5_path=TRAIN_DATA,
        split="train",
        condition=condition,
    )
    # Validation remains clear for both training conditions.
    val_dataset = EcosetDataset(
        h5_path=TRAIN_DATA,
        split="val",
        condition="clear",
    )

    train_indices = load_subset_indices(
        indices_path=TRAIN_INDICES_PATH,
        dataset_length=len(train_dataset),
    )

    val_indices = load_subset_indices(
        indices_path=VAL_INDICES_PATH,
        dataset_length=len(val_dataset),
    )

    train_subset = Subset(
        train_dataset,
        train_indices,
    )

    val_subset = Subset(
        val_dataset,
        val_indices,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    common_loader_arguments = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": NUM_WORKERS > 0,
    }

    # prefetch_factor is only valid when num_workers > 0.
    if NUM_WORKERS > 0:
        common_loader_arguments["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        drop_last=False,
        **common_loader_arguments,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    print(
        f"Created Ecoset loaders | "
        f"condition={condition} | "
        f"train={len(train_subset):,} | "
        f"val={len(val_subset):,}"
    )

    return train_loader, val_loader