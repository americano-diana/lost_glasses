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
from preprocessing import PREPROCESSED_DATA_ROOT

# ------------------------------------------------
# Default configuration
# ------------------------------------------------

DEFAULT_BATCH_SIZE = 128
DEFAULT_VALIDATION_RATIO = 0.2
DEFAULT_RANDOM_STATE = 42


# ------------------------------------------------
# Model preprocessing
# ------------------------------------------------

model_preprocessing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])


# ------------------------------------------------
# Path utilities
# ------------------------------------------------

def get_latest_preprocessing_run(
    preprocessing_root: Path = PREPROCESSED_DATA_ROOT,
) -> Path:
    """
    Read the preprocessing path stored in latest_run.txt.
    """
    preprocessing_root = Path(preprocessing_root)
    latest_run_file = preprocessing_root / "latest_run.txt"

    if not latest_run_file.exists():
        raise FileNotFoundError(
            "Could not find the latest preprocessing run file:\n"
            f"{latest_run_file}\n\n"
            "Run preprocess_data.py first or pass a specific "
            "preprocessed_run_dir."
        )

    run_directory = Path(
        latest_run_file.read_text(encoding="utf-8").strip()
    )

    if not run_directory.exists():
        raise FileNotFoundError(
            "The preprocessing directory listed in latest_run.txt "
            f"does not exist:\n{run_directory}"
        )

    return run_directory


def validate_dataset_directory(
    dataset_directory: Path,
) -> None:
    """
    Confirm that a preprocessed dataset has train and test folders.
    """
    required_directories = [
        dataset_directory / "train",
        dataset_directory / "test",
    ]

    missing_directories = [
        directory
        for directory in required_directories
        if not directory.exists()
    ]

    if missing_directories:
        missing_text = "\n".join(
            str(directory)
            for directory in missing_directories
        )

        raise FileNotFoundError(
            "The following dataset directories are missing:\n"
            f"{missing_text}"
        )


# ------------------------------------------------
# Dataset splitting
# ------------------------------------------------

def create_stratified_split_indices(
    targets: list[int],
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[list[int], list[int]]:
    """
    Create reproducible, stratified train and validation indices.
    """
    all_indices = list(range(len(targets)))

    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=validation_ratio,
        stratify=targets,
        random_state=random_state,
    )

    return train_indices, validation_indices


def validate_matching_datasets(
    reference_dataset: datasets.ImageFolder,
    comparison_dataset: datasets.ImageFolder,
    reference_name: str,
    comparison_name: str,
) -> None:
    """
    Check whether two dataset versions have compatible samples.

    This matters when the same split indices are used for clear and
    blurred versions.
    """
    if reference_dataset.classes != comparison_dataset.classes:
        raise ValueError(
            f"{reference_name} and {comparison_name} have different "
            "class definitions."
        )

    if len(reference_dataset) != len(comparison_dataset):
        raise ValueError(
            f"{reference_name} contains {len(reference_dataset)} "
            f"training images, but {comparison_name} contains "
            f"{len(comparison_dataset)}."
        )

    if reference_dataset.targets != comparison_dataset.targets:
        raise ValueError(
            f"{reference_name} and {comparison_name} have different "
            "target ordering. The same train-validation indices cannot "
            "safely be used."
        )


# ------------------------------------------------
# Dataloader construction
# ------------------------------------------------

def create_dataloaders(
    preprocessed_run_dir: Optional[Path] = None,
    dataset_names: tuple[str, ...] = (
        "clear",
        "blur_2",
        "blur_5",
    ),
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> dict:
    """
    Create train, validation, and test dataloaders.

    Parameters
    ----------
    preprocessed_run_dir:
        Specific timestamped preprocessing directory. When omitted,
        the directory listed in latest_run.txt is used.

    dataset_names:
        Dataset versions to load, such as clear, blur_2, and blur_5.

    batch_size:
        Number of samples in each batch.

    validation_ratio:
        Fraction of training data used for validation.

    random_state:
        Seed used for the stratified split.

    num_workers:
        Number of worker processes used by each DataLoader.

    pin_memory:
        Whether DataLoader should pin CPU memory. When omitted, it is
        enabled automatically if CUDA is available.

    Returns
    -------
    dict
        A dictionary containing datasets, subsets, and dataloaders.
    """
    if preprocessed_run_dir is None:
        preprocessed_run_dir = get_latest_preprocessing_run()
    else:
        preprocessed_run_dir = Path(
            preprocessed_run_dir
        ).resolve()

    if not preprocessed_run_dir.exists():
        raise FileNotFoundError(
            f"Preprocessing run does not exist: "
            f"{preprocessed_run_dir}"
        )

    if not dataset_names:
        raise ValueError(
            "At least one dataset name must be provided."
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaded_datasets = {}

    for dataset_name in dataset_names:
        dataset_directory = (
            preprocessed_run_dir / dataset_name
        )

        validate_dataset_directory(dataset_directory)

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

    # Use the first requested dataset as the reference.
    reference_name = dataset_names[0]
    reference_train_dataset = (
        loaded_datasets[reference_name]["train_full"]
    )

    train_indices, validation_indices = (
        create_stratified_split_indices(
            targets=reference_train_dataset.targets,
            validation_ratio=validation_ratio,
            random_state=random_state,
        )
    )

    output = {
        "preprocessed_run_dir": preprocessed_run_dir,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "datasets": {},
    }

    for dataset_name in dataset_names:
        train_dataset = (
            loaded_datasets[dataset_name]["train_full"]
        )
        test_dataset = loaded_datasets[dataset_name]["test"]

        if dataset_name != reference_name:
            validate_matching_datasets(
                reference_dataset=reference_train_dataset,
                comparison_dataset=train_dataset,
                reference_name=reference_name,
                comparison_name=dataset_name,
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
            "train_full": train_dataset,
            "train": train_subset,
            "validation": validation_subset,
            "test": test_dataset,
            "train_loader": DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            "validation_loader": DataLoader(
                validation_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            "test_loader": DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
        }

    print_dataloader_summary(output)

    return output


def print_dataloader_summary(dataloader_data: dict) -> None:
    """
    Print the number of samples in each dataset split.
    """
    print("\nDataset summary")
    print("----------------")
    print(
        "Preprocessing version: "
        f"{dataloader_data['preprocessed_run_dir'].name}"
    )

    for dataset_name, dataset_data in (
        dataloader_data["datasets"].items()
    ):
        print(f"\n{dataset_name}")
        print(f"  Training:   {len(dataset_data['train'])}")
        print(
            f"  Validation: "
            f"{len(dataset_data['validation'])}"
        )
        print(f"  Test:       {len(dataset_data['test'])}")


if __name__ == "__main__":
    create_dataloaders()