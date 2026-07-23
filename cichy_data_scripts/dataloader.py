"""
dataloader.py

ImageNet dataloaders for fine-tuning a pretrained AlexNet.

Training conditions:
    clear:
        Standard ImageNet augmentation without blur.

    mixed:
        Each training image is dynamically presented as either
        clear or Gaussian blurred.

    blur:
        Every training image is dynamically blurred.

Validation loaders:
    clear validation
    blurred validation

Optional saved subset indices can be used to shorten training.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    IMAGENET_ROOT,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_VAL,
    NUM_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    IMAGE_SIZE,
    RESIZE_SIZE,
    BLUR_PROBABILITY,
    BLUR_KERNEL_SIZE,
    BLUR_SIGMA_MIN,
    BLUR_SIGMA_MAX,
    TRAIN_INDICES_PATH,
    VAL_INDICES_PATH,
    SEED,
)


TrainingCondition = Literal[
    "clear",
    "mixed",
    "blur",
]


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# Transforms
# ============================================================

def create_transform(
    split: str,
    condition: TrainingCondition = "clear",
):
    """
    Create ImageNet preprocessing and augmentation.

    Training:
        RandomResizedCrop
        RandomHorizontalFlip
        optional dynamic blur
        ToTensor
        ImageNet normalization

    Validation:
        Resize
        CenterCrop
        optional blur
        ToTensor
        ImageNet normalization
    """
    if split not in {
        "train",
        "val",
    }:
        raise ValueError(
            "split must be 'train' or 'val'."
        )

    if condition not in {
        "clear",
        "mixed",
        "blur",
    }:
        raise ValueError(
            "condition must be "
            "'clear', 'mixed', or 'blur'."
        )

    blur_transform = transforms.GaussianBlur(
        kernel_size=BLUR_KERNEL_SIZE,
        sigma=(
            BLUR_SIGMA_MIN,
            BLUR_SIGMA_MAX,
        ),
    )

    operations = []

    if split == "train":
        operations.extend([
            transforms.RandomResizedCrop(
                IMAGE_SIZE
            ),
            transforms.RandomHorizontalFlip(),
        ])

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
        operations.extend([
            transforms.Resize(
                RESIZE_SIZE
            ),
            transforms.CenterCrop(
                IMAGE_SIZE
            ),
        ])

        # A mixed validation set is not recommended because it
        # changes randomly between evaluations.
        if condition == "blur":
            operations.append(
                blur_transform
            )

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return transforms.Compose(
        operations
    )


# ============================================================
# Reproducibility
# ============================================================

def seed_worker(worker_id):
    """
    Give every DataLoader worker a reproducible random seed.
    """
    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    np.random.seed(worker_seed)


# ============================================================
# Optional subset loading
# ============================================================

def apply_saved_subset(
    dataset,
    indices_path,
    split_name: str,
):
    """
    Restrict a dataset using saved NumPy indices.

    If indices_path is None, return the full dataset.
    """
    if indices_path is None:
        print(
            f"{split_name}: using full dataset "
            f"({len(dataset):,} images)"
        )
        return dataset

    indices_path = Path(
        indices_path
    )

    if not indices_path.exists():
        raise FileNotFoundError(
            f"{split_name} subset file not found: "
            f"{indices_path}"
        )

    indices = np.load(
        indices_path
    ).astype(np.int64)

    if indices.ndim != 1:
        raise ValueError(
            f"{split_name} indices must be "
            f"one-dimensional, but found "
            f"shape {indices.shape}."
        )

    if len(indices) == 0:
        raise ValueError(
            f"{split_name} indices are empty."
        )

    if (
        indices.min() < 0
        or indices.max() >= len(dataset)
    ):
        raise IndexError(
            f"{split_name} indices outside dataset "
            f"range 0–{len(dataset) - 1}. "
            f"Observed range: "
            f"{indices.min()}–{indices.max()}."
        )

    if len(np.unique(indices)) != len(indices):
        raise ValueError(
            f"{split_name} subset contains "
            f"duplicate indices."
        )

    subset = Subset(
        dataset,
        indices.tolist(),
    )

    print(
        f"{split_name}: using saved subset "
        f"with {len(subset):,} images"
    )

    print(
        f"{split_name} indices: "
        f"{indices_path}"
    )

    return subset


# ============================================================
# Loader creation
# ============================================================

def create_dataloaders(
    condition: TrainingCondition = "mixed",
):
    """
    Create ImageNet fine-tuning and validation loaders.

    Returns
    -------
    train_loader:
        Clear, mixed, or fully blurred training loader.

    clear_val_loader:
        Deterministic clear validation images.

    blur_val_loader:
        Deterministic validation pipeline with Gaussian blur.
    """
    if condition not in {
        "clear",
        "mixed",
        "blur",
    }:
        raise ValueError(
            "condition must be "
            "'clear', 'mixed', or 'blur'."
        )

    imagenet_root = Path(
        IMAGENET_ROOT
    )

    train_root = (
        imagenet_root / "train"
    )

    val_root = (
        imagenet_root / "val"
    )

    if not train_root.exists():
        raise FileNotFoundError(
            f"ImageNet training directory not found: "
            f"{train_root}"
        )

    if not val_root.exists():
        raise FileNotFoundError(
            f"ImageNet validation directory not found: "
            f"{val_root}"
        )

    # ------------------------------------------------
    # Datasets
    # ------------------------------------------------

    train_dataset = datasets.ImageFolder(
        root=train_root,
        transform=create_transform(
            split="train",
            condition=condition,
        ),
    )

    clear_val_dataset = datasets.ImageFolder(
        root=val_root,
        transform=create_transform(
            split="val",
            condition="clear",
        ),
    )

    blur_val_dataset = datasets.ImageFolder(
        root=val_root,
        transform=create_transform(
            split="val",
            condition="blur",
        ),
    )

    # Ensure class labels correspond across every dataset.
    if (
        train_dataset.class_to_idx
        != clear_val_dataset.class_to_idx
    ):
        raise RuntimeError(
            "Training and validation class mappings "
            "do not match."
        )

    if (
        clear_val_dataset.class_to_idx
        != blur_val_dataset.class_to_idx
    ):
        raise RuntimeError(
            "Clear and blurred validation class "
            "mappings do not match."
        )

    # ------------------------------------------------
    # Optional subsets
    # ------------------------------------------------

    train_dataset = apply_saved_subset(
        dataset=train_dataset,
        indices_path=TRAIN_INDICES_PATH,
        split_name="Training",
    )

    clear_val_dataset = apply_saved_subset(
        dataset=clear_val_dataset,
        indices_path=VAL_INDICES_PATH,
        split_name="Clear validation",
    )

    # Use the exact same image indices for blurred validation.
    blur_val_dataset = apply_saved_subset(
        dataset=blur_val_dataset,
        indices_path=VAL_INDICES_PATH,
        split_name="Blur validation",
    )

    # ------------------------------------------------
    # Random generators
    # ------------------------------------------------

    train_generator = torch.Generator()
    train_generator.manual_seed(
        SEED
    )

    validation_generator = torch.Generator()
    validation_generator.manual_seed(
        SEED + 1
    )

    # ------------------------------------------------
    # Common loader settings
    # ------------------------------------------------

    common_arguments = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "worker_init_fn": seed_worker,
        "persistent_workers": (
            NUM_WORKERS > 0
        ),
    }

    if NUM_WORKERS > 0:
        common_arguments[
            "prefetch_factor"
        ] = PREFETCH_FACTOR

    # ------------------------------------------------
    # DataLoaders
    # ------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        generator=train_generator,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
        prefetch_factor=(
            PREFETCH_FACTOR
            if NUM_WORKERS > 0
            else None
        ),
    )

    clear_val_loader = DataLoader(
        clear_val_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        drop_last=False,
        num_workers=min(NUM_WORKERS, 2),
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
        prefetch_factor=(
            2
            if NUM_WORKERS > 0
            else None
        ),
    )

    blur_val_loader = DataLoader(
        blur_val_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        drop_last=False,
        num_workers=min(NUM_WORKERS, 2),
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
        prefetch_factor=(
            2
            if NUM_WORKERS > 0
            else None
        ),
    )

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    print()
    print("=" * 70)
    print("ImageNet dataloaders")
    print("=" * 70)

    print(
        f"Training condition: {condition}"
    )

    print(
        f"Classes: "
        f"{len(train_dataset.dataset.classes) if isinstance(train_dataset, Subset) else len(train_dataset.classes)}"
    )

    print(
        f"Training images: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Clear validation images: "
        f"{len(clear_val_dataset):,}"
    )

    print(
        f"Blur validation images: "
        f"{len(blur_val_dataset):,}"
    )

    print(
        f"Training batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Clear validation batches: "
        f"{len(clear_val_loader):,}"
    )

    print(
        f"Blur validation batches: "
        f"{len(blur_val_loader):,}"
    )

    print("=" * 70)

    return (
        train_loader,
        clear_val_loader,
        blur_val_loader,
    )