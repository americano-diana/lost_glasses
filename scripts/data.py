""""
Data loader script
"""

# ------------------------------------------------
# Library imports
# ------------------------------------------------
from pathlib import Path
from typing import Optional

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ------------------------------------------------
# Local imports
# ------------------------------------------------
from config import ( IMAGE_SIZE, PIN_MEMORY, RAW_DATASET, RAW_BLUR_2, RAW_BLUR_5)


# Define path to raw datasets
DATASET_PATHS = { "clear": RAW_DATASET, "blur_2": RAW_BLUR_2, "blur_5": RAW_BLUR_5, }

# ------------------------------------------------
# Model preprocessing
# ------------------------------------------------

model_preprocessing = transforms.Compose([ 
    transforms.Resize(IMAGE_SIZE), 
    transforms.ToTensor(), 
    transforms.Normalize( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), 
    ), 
])

# ------------------------------------------------
# Create train-validation indices
# ------------------------------------------------

def create_split_indices(
    targets,
    validation_ratio,
    random_state,
):
    """
    Create reproducible and class-balanced train-validation indices.
    """
    all_indices = list(range(len(targets)))

    train_indices, validation_indices = (
        train_test_split(
            all_indices,
            test_size=validation_ratio,
            stratify=targets,
            random_state=random_state,
        )
    )

    return train_indices, validation_indices


# ------------------------------------------------
# Create DataLoaders
# ------------------------------------------------

def create_dataloaders(
    dataset_names=("clear", "blur_5"),
    batch_size=128,
    validation_ratio=0.2,
    random_state=42,
    num_workers=2,
    pin_memory=PIN_MEMORY,
):
    """
    Create train, validation, and test DataLoaders.

    Images are resized dynamically by model_preprocessing.
    """
    if not dataset_names:
        raise ValueError(
            "Provide at least one dataset name."
        )

    loaded_datasets = {}

    # Load full training and test datasets.
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_PATHS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: "
                f"{tuple(DATASET_PATHS.keys())}"
            )

        dataset_directory = Path(
            DATASET_PATHS[dataset_name]
        )

        loaded_datasets[dataset_name] = {
            "train_full": datasets.ImageFolder(
                root=dataset_directory / "train",
                transform=model_preprocessing,
            ),
            "test": datasets.ImageFolder(
                root=dataset_directory / "test",
                transform=model_preprocessing,
            ),
        }

    # Use the first dataset, normally clear, to define the split.
    reference_name = dataset_names[0]

    reference_train_dataset = (
        loaded_datasets[reference_name][
            "train_full"
        ]
    )

    train_indices, validation_indices = (
        create_split_indices(
            targets=reference_train_dataset.targets,
            validation_ratio=validation_ratio,
            random_state=random_state,
        )
    )

    output = {
        "datasets": {},
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }

    for dataset_name in dataset_names:
        train_dataset = (
            loaded_datasets[dataset_name][
                "train_full"
            ]
        )

        test_dataset = (
            loaded_datasets[dataset_name]["test"]
        )

        train_subset = Subset(
            train_dataset,
            train_indices,
        )

        validation_subset = Subset(
            train_dataset,
            validation_indices,
        )

        output["datasets"][dataset_name] = {
            "train": train_subset,
            "validation": validation_subset,
            "test": test_dataset,

            "train_loader": DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(
                    num_workers > 0
                ),
            ),

            "validation_loader": DataLoader(
                validation_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(
                    num_workers > 0
                ),
            ),

            "test_loader": DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(
                    num_workers > 0
                ),
            ),
        }

    print_dataloader_summary(output)

    return output


# ------------------------------------------------
# Print summary
# ------------------------------------------------

def print_dataloader_summary(data):
    """
    Print dataset sizes.
    """
    print("\nDataset summary")
    print("----------------")

    for dataset_name, dataset_data in (
        data["datasets"].items()
    ):
        print(f"\n{dataset_name}")
        print(
            f"  Training:   "
            f"{len(dataset_data['train'])}"
        )
        print(
            f"  Validation: "
            f"{len(dataset_data['validation'])}"
        )
        print(
            f"  Test:       "
            f"{len(dataset_data['test'])}"
        )


if __name__ == "__main__":
    create_dataloaders()