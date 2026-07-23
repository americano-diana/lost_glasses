"""
main.py

Fine-tune an ImageNet-pretrained AlexNet using mixed clear and
Gaussian-blurred ImageNet images.

The untouched ImageNet-pretrained weights form the standard model.
The fine-tuned weights form the blur-expert model.
"""

# ------------------------------------------------
# Library imports
# ------------------------------------------------

from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torchvision.models import (
    AlexNet_Weights,
    alexnet,
)

from .config import (
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_VAL,
    BLUR_KERNEL_SIZE,
    BLUR_PROBABILITY,
    BLUR_SIGMA_MIN,
    BLUR_SIGMA_MAX,
    CHECKPOINT_DIR,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENT_NAME,
    IMAGENET_ROOT,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_WORKERS,
    RESUME_CHECKPOINT,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    SEED,
    TRAINABLE_CLASSIFIER_LAYERS,
    TRAINING_CONDITION,
    TRAIN_INDICES_PATH,
    USE_AMP,
    USE_WANDB,
    VAL_INDICES_PATH,
    WANDB_ENTITY,
    WANDB_NOTES,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
    WANDB_TAGS,
    WEIGHT_DECAY,
)

from .dataloader import create_dataloaders
from .train import (
    plot_training_history,
    train_model,
)
from .utils import (
    initialize_wandb,
    set_seed,
)


# ============================================================
# Model setup
# ============================================================

def create_finetuning_model(
    trainable_classifier_layers: int,
):
    """
    Load ImageNet-pretrained AlexNet and freeze most layers.

    Parameters
    ----------
    trainable_classifier_layers:
        Number of final Linear classifier layers to unfreeze.

        1:
            classifier[6]

        2:
            classifier[4] and classifier[6]

        3:
            classifier[1], classifier[4], classifier[6]
    """
    if trainable_classifier_layers not in {
        1,
        2,
        3,
    }:
        raise ValueError(
            "TRAINABLE_CLASSIFIER_LAYERS must "
            "be 1, 2, or 3."
        )

    weights = AlexNet_Weights.DEFAULT

    model = alexnet(
        weights=weights,
    )

    # Freeze all convolutional and classifier parameters.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Linear layers in torchvision AlexNet's classifier.
    linear_layer_indices = [
        1,
        4,
        6,
    ]

    layers_to_unfreeze = (
        linear_layer_indices[
            -trainable_classifier_layers:
        ]
    )

    for layer_index in layers_to_unfreeze:
        for parameter in (
            model.classifier[
                layer_index
            ].parameters()
        ):
            parameter.requires_grad = True

    return model, weights, layers_to_unfreeze


def save_standard_model(
    model: nn.Module,
    save_path,
):
    """
    Save the untouched ImageNet-pretrained baseline model.

    Does nothing if the file already exists.
    """
    save_path = Path(
        save_path
    )

    if save_path.exists():
        print(
            "Standard model already exists:"
            f"\n  {save_path}"
        )
        return

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "architecture": (
                "torchvision_alexnet"
            ),
            "source": (
                "AlexNet_Weights.DEFAULT"
            ),
            "condition": (
                "standard_imagenet"
            ),
            "num_classes": 1000,
            "model_state_dict": (
                model.state_dict()
            ),
        },
        save_path,
    )

    print(
        "Saved untouched standard model:"
        f"\n  {save_path}"
    )


# ============================================================
# Main
# ============================================================

def main():
    """
    Run ImageNet blur-expert fine-tuning.
    """
    condition = TRAINING_CONDITION

    if condition not in {
        "clear",
        "mixed",
        "blur",
    }:
        raise ValueError(
            "TRAINING_CONDITION must be "
            "'clear', 'mixed', or 'blur'."
        )

    set_seed(
        SEED
    )

    print("=" * 70)
    print("ImageNet-pretrained AlexNet fine-tuning")
    print("=" * 70)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Condition: {condition}")
    print(f"Device: {DEVICE}")
    print(
        "Trainable final classifier layers: "
        f"{TRAINABLE_CLASSIFIER_LAYERS}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
        print(
            "CUDA version:",
            torch.version.cuda,
        )
    else:
        print(
            "CUDA is not available."
        )

    # ------------------------------------------------
    # Output paths
    # ------------------------------------------------

    run_name = (
        f"imagenet_alexnet_{condition}"
        f"_last{TRAINABLE_CLASSIFIER_LAYERS}"
        f"_seed{SEED}"
    )

    condition_checkpoint_dir = (
        Path(CHECKPOINT_DIR)
        / run_name
    )

    condition_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    standard_model_path = (
        Path(CHECKPOINT_DIR)
        / "standard_imagenet_alexnet.pt"
    )

    # ------------------------------------------------
    # Data
    # ------------------------------------------------

    print(
        "\nCreating ImageNet dataloaders..."
    )

    (
        train_loader,
        clear_validation_loader,
        blur_validation_loader,
    ) = create_dataloaders(
        condition=condition,
    )

    train_samples = len(
        train_loader.dataset
    )

    clear_validation_samples = len(
        clear_validation_loader.dataset
    )

    blur_validation_samples = len(
        blur_validation_loader.dataset
    )

    print(
        f"Training samples: "
        f"{train_samples:,}"
    )

    print(
        f"Clear validation samples: "
        f"{clear_validation_samples:,}"
    )

    print(
        f"Blur validation samples: "
        f"{blur_validation_samples:,}"
    )

    sample_images, sample_labels = next(
        iter(train_loader)
    )

    print(
        "Sample image batch shape:",
        tuple(sample_images.shape),
    )

    print(
        "Sample label batch shape:",
        tuple(sample_labels.shape),
    )

    print(
        "Sample label range:",
        int(sample_labels.min()),
        "to",
        int(sample_labels.max()),
    )

    if sample_labels.min() < 0:
        raise RuntimeError(
            "Negative ImageNet class label found."
        )

    if sample_labels.max() >= 1000:
        raise RuntimeError(
            "ImageNet class label exceeds 999."
        )

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    (
        model,
        weights,
        unfrozen_layer_indices,
    ) = create_finetuning_model(
        trainable_classifier_layers=(
            TRAINABLE_CLASSIFIER_LAYERS
        )
    )

    # Save untouched pretrained weights before training.
    if RESUME_CHECKPOINT is None:
        save_standard_model(
            model=model,
            save_path=standard_model_path,
        )

    model = model.to(
        DEVICE
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    trainable_parameter_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    print()
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        "Trainable percentage: "
        f"{(100.0 * trainable_parameters / total_parameters):.2f}%"
    )

    print(
        "Unfrozen classifier indices:",
        unfrozen_layer_indices,
    )

    print(
        "Trainable parameter names:"
    )

    for name in trainable_parameter_names:
        print(
            f"  {name}"
        )

    # ------------------------------------------------
    # Optimization
    # ------------------------------------------------

    loss_function = (
        nn.CrossEntropyLoss()
    )

    parameters_to_optimize = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if len(parameters_to_optimize) == 0:
        raise RuntimeError(
            "No trainable parameters were found."
        )

    optimizer = torch.optim.AdamW(
        parameters_to_optimize,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )
    )

    # ------------------------------------------------
    # Metadata
    # ------------------------------------------------

    model_settings = {
        "architecture": (
            "torchvision_alexnet"
        ),
        "pretrained_weights": (
            str(weights)
        ),
        "num_classes": 1000,
        "total_parameters": (
            total_parameters
        ),
        "trainable_parameters": (
            trainable_parameters
        ),
        "trainable_percentage": (
            100.0
            * trainable_parameters
            / total_parameters
        ),
        "trainable_classifier_layers": (
            TRAINABLE_CLASSIFIER_LAYERS
        ),
        "unfrozen_classifier_indices": (
            unfrozen_layer_indices
        ),
        "trainable_parameter_names": (
            trainable_parameter_names
        ),
    }

    training_settings = {
        "dataset": "ILSVRC2012",
        "data_path": str(
            IMAGENET_ROOT
        ),
        "condition": condition,
        "train_samples": train_samples,
        "clear_validation_samples": (
            clear_validation_samples
        ),
        "blur_validation_samples": (
            blur_validation_samples
        ),
        "training_batches": (
            len(train_loader)
        ),
        "clear_validation_batches": (
            len(clear_validation_loader)
        ),
        "blur_validation_batches": (
            len(blur_validation_loader)
        ),
        "train_indices_path": (
            str(TRAIN_INDICES_PATH)
            if TRAIN_INDICES_PATH is not None
            else None
        ),
        "val_indices_path": (
            str(VAL_INDICES_PATH)
            if VAL_INDICES_PATH is not None
            else None
        ),
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size_train": (
            BATCH_SIZE_TRAIN
        ),
        "batch_size_val": (
            BATCH_SIZE_VAL
        ),
        "num_workers": NUM_WORKERS,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "scheduler": (
            "ReduceLROnPlateau"
        ),
        "scheduler_factor": (
            SCHEDULER_FACTOR
        ),
        "scheduler_patience": (
            SCHEDULER_PATIENCE
        ),
        "scheduler_min_lr": (
            SCHEDULER_MIN_LR
        ),
        "selection_metric": (
            "mean_clear_blur_validation_loss"
        ),
        "use_amp": USE_AMP,
        "seed": SEED,
        "blur_probability": (
            BLUR_PROBABILITY
        ),
        "blur_kernel_size": (
            BLUR_KERNEL_SIZE
        ),
        "blur_sigma_min": (
            BLUR_SIGMA_MIN
        ),
        "blur_sigma_max": (
            BLUR_SIGMA_MAX
        ),
    }

    # ------------------------------------------------
    # W&B
    # ------------------------------------------------

    if USE_WANDB:
        initialize_wandb(
            condition=condition,
            seed=SEED,
            run_name_base=(
                WANDB_RUN_NAME
            ),
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            tags=(
                list(WANDB_TAGS)
                + [
                    "imagenet",
                    "pretrained",
                    "finetuning",
                    "blur-expert",
                ]
            ),
            notes=WANDB_NOTES,
            model_settings=model_settings,
            training_settings=(
                training_settings
            ),
            data_path=IMAGENET_ROOT,
        )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    try:
        (
            history,
            best_selection_loss,
            final_epoch,
        ) = train_model(
            model=model,
            train_loader=train_loader,
            clear_validation_loader=(
                clear_validation_loader
            ),
            blur_validation_loader=(
                blur_validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            max_epochs=MAX_EPOCHS,
            condition=condition,
            checkpoint_directory=(
                condition_checkpoint_dir
            ),
            model_settings=model_settings,
            training_settings=(
                training_settings
            ),
            early_stopping_patience=(
                EARLY_STOPPING_PATIENCE
            ),
            use_amp=USE_AMP,
            resume_checkpoint=(
                RESUME_CHECKPOINT
            ),
            use_wandb=USE_WANDB,
        )

        plot_training_history(
            history=history,
            save_directory=(
                condition_checkpoint_dir
            ),
            title=(
                "ImageNet AlexNet "
                f"{condition} fine-tuning"
            ),
            show=False,
        )

        print()
        print("=" * 70)
        print("Fine-tuning complete")
        print("=" * 70)
        print(
            f"Condition: {condition}"
        )
        print(
            f"Final completed epoch: "
            f"{final_epoch}"
        )
        print(
            "Best mean clear/blur "
            "validation loss: "
            f"{best_selection_loss:.4f}"
        )
        print(
            "Standard model:"
            f"\n  {standard_model_path}"
        )
        print(
            "Expert checkpoints:"
            f"\n  {condition_checkpoint_dir}"
        )

    except KeyboardInterrupt:
        print(
            "\nFine-tuning interrupted by user."
        )
        print(
            "The latest completed epoch should "
            "remain available in last.pt."
        )
        raise

    except Exception:
        print(
            "\nFine-tuning stopped because "
            "of an error."
        )
        print(
            "The latest completed epoch should "
            "remain available in last.pt."
        )
        raise

    finally:
        if USE_WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()