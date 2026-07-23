"""Script to evaluate the standard ImageNet AlexNet on clear and blurred validation sets.
python -m cichy_data_scripts/evals/test_standard_alexnet.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import alexnet

from ..config import DEVICE, SEED, CHECKPOINT_DIR
from ..dataloader import create_dataloaders
from ..utils import set_seed

# Load checkpoint for standard imagenet pre-trained AlexNet
STANDARD_CHECKPOINT = CHECKPOINT_DIR / "standard_imagenet_alexnet.pt"

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    description: str,
) -> tuple[float, float]:
    """Evaluate a model and return mean loss and top-1 accuracy."""

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=description,
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item()
        total_correct += predictions.eq(labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(f"{description} dataloader is empty.")

    mean_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return mean_loss, accuracy


def load_standard_model(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load the saved standard ImageNet AlexNet."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Standard AlexNet checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    model = alexnet(weights=None)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Support several common checkpoint formats.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # The checkpoint itself may be a raw state_dict.
            state_dict = checkpoint
    else:
        raise TypeError(
            "The checkpoint must contain a PyTorch state dictionary."
        )

    # Handle checkpoints saved from DataParallel.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def main() -> None:
    print("=" * 70)
    print("Standard ImageNet AlexNet evaluation")
    print("=" * 70)

    set_seed(SEED)

    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nCreating validation dataloaders...")

    # Use the same dataloader function as the mixed-model training run.
    #
    # The training loader is created but is not used here. The important
    # loaders are clear_val_loader and blur_val_loader.
    (
        _,
        clear_val_loader,
        blur_val_loader,
    ) = create_dataloaders(
        condition="mixed",
    )

    print(f"Clear validation samples: {len(clear_val_loader.dataset):,}")
    print(f"Blur validation samples: {len(blur_val_loader.dataset):,}")

    if len(clear_val_loader.dataset) != 2_000:
        raise ValueError(
            "Expected 2,000 clear validation images, but found "
            f"{len(clear_val_loader.dataset):,}."
        )

    if len(blur_val_loader.dataset) != 2_000:
        raise ValueError(
            "Expected 2,000 blurred validation images, but found "
            f"{len(blur_val_loader.dataset):,}."
        )

    print(f"\nLoading standard model:\n  {STANDARD_CHECKPOINT}")

    model = load_standard_model(
        checkpoint_path=STANDARD_CHECKPOINT,
        device=device,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {total_parameters:,}")

    print("\nEvaluating clear validation images...")
    clear_loss, clear_accuracy = evaluate(
        model=model,
        dataloader=clear_val_loader,
        device=device,
        description="Clear validation",
    )

    print("Evaluating blurred validation images...")
    blur_loss, blur_accuracy = evaluate(
        model=model,
        dataloader=blur_val_loader,
        device=device,
        description="Blur validation",
    )

    selection_loss = (clear_loss + blur_loss) / 2.0

    print("\n" + "=" * 70)
    print("Standard AlexNet results")
    print("=" * 70)
    print(
        f"Clear validation | "
        f"loss: {clear_loss:.4f} | "
        f"accuracy: {clear_accuracy:.2f}%"
    )
    print(
        f"Blur validation  | "
        f"loss: {blur_loss:.4f} | "
        f"accuracy: {blur_accuracy:.2f}%"
    )
    print(f"Mean clear/blur validation loss: {selection_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()