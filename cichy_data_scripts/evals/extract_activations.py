"""Extract AlexNet activations and create model RDMs for the Cichy images.

This script does not load or analyze the fMRI data.

Run from the project root:

    python -m cichy_data_scripts.extract_activations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import alexnet
from torchvision.transforms import v2
from tqdm import tqdm

from .config import CHECKPOINT_DIR, DEVICE, SEED, CICHY_DATA
from .utils import set_seed


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CICHY_IMAGE_DIR = CICHY_DATA / "92images"

OUTPUT_DIR = (
    CHECKPOINT_DIR.parent
    / "rdms"
    / "cichy_data_models"
)


# ---------------------------------------------------------------------
# Select models and layers here
# ---------------------------------------------------------------------

MODELS_TO_EXTRACT = {
    "standard": (
        CHECKPOINT_DIR
        / "standard_imagenet_alexnet.pt"
    ),
    "expert": (
        CHECKPOINT_DIR
        / "imagenet_alexnet_mixed_last2_seed42"
        / "best.pt"
    ),
}

LAYERS_TO_EXTRACT = {
    # Second trained Linear layer, before ReLU.
    "fc7_pre_relu": "classifier.4",

    # Same layer after ReLU.
    "fc7_post_relu": "classifier.5",

    # Final 1,000-dimensional classifier output.
    "fc8_logits": "classifier.6",
}


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

EXPECTED_STIMULI = 92
BATCH_SIZE = 32
NUM_WORKERS = 4
SAVE_ACTIVATIONS = True


IMAGE_TRANSFORM = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(
            torch.float32,
            scale=True,
        ),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ---------------------------------------------------------------------
# Cichy image dataset
# ---------------------------------------------------------------------

def find_cichy_images(
    image_dir: Path,
) -> list[Path]:
    """Find the Cichy images in alphabetical order."""

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Cichy image directory not found:\n{image_dir}"
        )

    image_paths = sorted(
        image_dir.glob("*.jpg"),
        key=lambda path: path.name,
    )

    if len(image_paths) != EXPECTED_STIMULI:
        raise ValueError(
            f"Expected {EXPECTED_STIMULI} JPG images in "
            f"{image_dir}, but found {len(image_paths)}."
        )

    return image_paths


class CichyImageDataset(Dataset):
    """Dataset preserving the Cichy stimulus order."""

    def __init__(
        self,
        image_paths: list[Path],
    ) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int, str]:
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = IMAGE_TRANSFORM(image)

        return image_tensor, index, image_path.name


def create_cichy_dataloader(
    image_paths: list[Path],
) -> DataLoader:
    dataset = CichyImageDataset(
        image_paths=image_paths,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """Extract a state dictionary from common checkpoint formats."""

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected the checkpoint to be a dictionary."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    else:
        if all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise KeyError(
                "Could not find a model state dictionary. "
                f"Available keys: {list(checkpoint.keys())}"
            )

    if not isinstance(state_dict, dict):
        raise TypeError(
            "The extracted state dictionary is invalid."
        )

    return {
        name.removeprefix("module."): value
        for name, value in state_dict.items()
    }


def load_alexnet_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load an AlexNet checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    model = alexnet(weights=None)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------

def get_module_by_name(
    model: nn.Module,
    module_name: str,
) -> nn.Module:
    """Retrieve a nested module such as classifier.4."""

    module = model

    for component in module_name.split("."):
        if component.isdigit():
            module = module[int(component)]
        else:
            module = getattr(module, component)

    return module


@torch.inference_mode()
def extract_activations(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    selected_layers: dict[str, str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Extract selected layer activations in stimulus order."""

    activation_batches: dict[str, list[torch.Tensor]] = {
        layer_alias: []
        for layer_alias in selected_layers
    }

    extracted_indices: list[torch.Tensor] = []
    extracted_filenames: list[str] = []
    hook_handles = []

    def make_hook(layer_alias: str):
        def hook(
            _module: nn.Module,
            _inputs,
            output: torch.Tensor,
        ) -> None:
            if isinstance(output, tuple):
                output = output[0]

            output = (
                output
                .detach()
                .flatten(start_dim=1)
                .float()
                .cpu()
            )

            activation_batches[layer_alias].append(output)

        return hook

    for layer_alias, module_name in selected_layers.items():
        module = get_module_by_name(
            model=model,
            module_name=module_name,
        )

        handle = module.register_forward_hook(
            make_hook(layer_alias)
        )

        hook_handles.append(handle)

    try:
        for images, indices, filenames in tqdm(
            dataloader,
            desc="Extracting activations",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            extracted_indices.append(indices.cpu())
            extracted_filenames.extend(list(filenames))

            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                _ = model(images)

    finally:
        for handle in hook_handles:
            handle.remove()

    indices = torch.cat(
        extracted_indices,
        dim=0,
    ).numpy()

    expected_indices = np.arange(
        len(dataloader.dataset)
    )

    if not np.array_equal(
        indices,
        expected_indices,
    ):
        raise RuntimeError(
            "Stimulus order changed during activation extraction."
        )

    activations = {
        layer_alias: torch.cat(
            batches,
            dim=0,
        ).numpy()
        for layer_alias, batches
        in activation_batches.items()
    }

    for layer_alias, layer_activations in activations.items():
        if layer_activations.shape[0] != len(dataloader.dataset):
            raise RuntimeError(
                f"{layer_alias}: expected "
                f"{len(dataloader.dataset)} stimuli, but extracted "
                f"{layer_activations.shape[0]}."
            )

        if not np.isfinite(layer_activations).all():
            raise ValueError(
                f"{layer_alias} contains NaN or infinite values."
            )

    return activations, extracted_filenames


# ---------------------------------------------------------------------
# RDM creation
# ---------------------------------------------------------------------

def correlation_rdm(
    activations: np.ndarray,
) -> np.ndarray:
    """Calculate correlation distance: 1 - Pearson correlation."""

    if activations.ndim != 2:
        raise ValueError(
            "Activations must have shape [stimuli, features]. "
            f"Received {activations.shape}."
        )

    rdm = 1.0 - np.corrcoef(
        activations.astype(
            np.float64,
            copy=False,
        )
    )

    if not np.isfinite(rdm).all():
        raise ValueError(
            "The RDM contains NaN or infinite values. "
            "This may indicate a zero-variance activation vector."
        )

    rdm = (rdm + rdm.T) / 2.0
    np.fill_diagonal(rdm, 0.0)

    return rdm.astype(np.float32)


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_stimulus_order(
    filenames: list[str],
) -> None:
    """Save the exact row and column order used by every RDM."""

    output_path = OUTPUT_DIR / "stimulus_order.txt"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for index, filename in enumerate(filenames):
            file.write(
                f"{index:02d}\t{filename}\n"
            )

    print(f"Saved stimulus order: {output_path}")


def save_model_results(
    model_name: str,
    checkpoint_path: Path,
    activations: dict[str, np.ndarray],
    filenames: list[str],
) -> None:
    """Save activations and RDMs for one model."""

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    layer_metadata = {}

    for layer_alias, layer_activations in activations.items():
        print("\n" + "-" * 72)
        print(f"Model: {model_name}")
        print(f"Layer: {layer_alias}")
        print(f"Activation shape: {layer_activations.shape}")

        if SAVE_ACTIVATIONS:
            activation_path = (
                model_dir
                / f"activations_{layer_alias}.npy"
            )

            np.save(
                activation_path,
                layer_activations.astype(np.float32),
            )

            print(f"Saved activations: {activation_path}")

        model_rdm = correlation_rdm(
            layer_activations
        )

        expected_shape = (
            len(filenames),
            len(filenames),
        )

        if model_rdm.shape != expected_shape:
            raise RuntimeError(
                f"Expected RDM shape {expected_shape}, "
                f"but received {model_rdm.shape}."
            )

        rdm_path = (
            model_dir
            / f"rdm_{layer_alias}_correlation.npy"
        )

        np.save(
            rdm_path,
            model_rdm,
        )

        print(f"RDM shape: {model_rdm.shape}")
        print(
            f"RDM range: "
            f"{model_rdm.min():.6f} to "
            f"{model_rdm.max():.6f}"
        )
        print(f"Saved RDM: {rdm_path}")

        layer_metadata[layer_alias] = {
            "module": LAYERS_TO_EXTRACT[layer_alias],
            "activation_shape": list(
                layer_activations.shape
            ),
            "rdm_shape": list(model_rdm.shape),
            "activation_file": (
                f"activations_{layer_alias}.npy"
                if SAVE_ACTIVATIONS
                else None
            ),
            "rdm_file": (
                f"rdm_{layer_alias}_correlation.npy"
            ),
        }

    metadata = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "number_of_stimuli": len(filenames),
        "stimulus_order": filenames,
        "layers": layer_metadata,
        "rdm_metric": "correlation distance",
        "rdm_definition": "1 - Pearson correlation",
        "preprocessing": {
            "resize": [224, 224],
            "normalization_mean": [
                0.485,
                0.456,
                0.406,
            ],
            "normalization_std": [
                0.229,
                0.224,
                0.225,
            ],
        },
    }

    with (
        model_dir / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Image activations extraction and model RDM creation")
    print("=" * 72)

    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device(DEVICE)
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = find_cichy_images(
        CICHY_IMAGE_DIR
    )

    expected_filenames = [
        path.name
        for path in image_paths
    ]

    print(f"Number of images: {len(image_paths)}")
    print(f"First image: {expected_filenames[0]}")
    print(f"Last image: {expected_filenames[-1]}")

    save_stimulus_order(
        expected_filenames
    )

    dataloader = create_cichy_dataloader(
        image_paths=image_paths,
    )

    for model_name, checkpoint_path in MODELS_TO_EXTRACT.items():
        print("\n" + "=" * 72)
        print(f"Model: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 72)

        model = load_alexnet_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        activations, extracted_filenames = extract_activations(
            model=model,
            dataloader=dataloader,
            device=device,
            selected_layers=LAYERS_TO_EXTRACT,
        )

        if extracted_filenames != expected_filenames:
            raise RuntimeError(
                f"{model_name}: extracted stimulus order does not "
                "match the expected order."
            )

        save_model_results(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            activations=activations,
            filenames=expected_filenames,
        )

        del model
        del activations

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("Extraction complete")
    print(f"Outputs saved under:\n{OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()