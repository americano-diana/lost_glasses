"""
main.py

Train one independent AlexNet model on Ecoset using the condition
selected in config.py.
"""

# ------------------------------------------------
# Library imports
# ------------------------------------------------

import torch
import torch.nn as nn
import wandb

from config import (
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_VAL,
    BLUR_KERNEL_SIZE,
    BLUR_PROBABILITY,
    BLUR_SIGMA_MIN,
    BLUR_SIGMA_MAX,
    CHECKPOINT_DIR,
    CREATE_NEW_INITIALIZATION,
    TRAIN_DATA,
    DEVICE,
    DOWNSCALE_FACTOR,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENT_NAME,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_CLASSES,
    NUM_WORKERS,
    RESUME_CHECKPOINT,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    SEED,
    TRAIN_INDICES_PATH,
    TRAINING_CONDITION,
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

from dataloader import create_dataloaders
from model import AlexNet
from train import (
    plot_training_history,
    train_model,
)
from utils import (
    create_or_load_shared_initialization,
    initialize_wandb,
    set_seed,
)


def main():
    """
    Run one Ecoset pretraining experiment.
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

    set_seed(SEED)

    print("=" * 70)
    print("Ecoset AlexNet pretraining")
    print("=" * 70)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Condition: {condition}")
    print(f"Device: {DEVICE}")

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
        print("CUDA is not available.")

    # ------------------------------------------------
    # Output paths
    # ------------------------------------------------

    condition_checkpoint_dir = (
        CHECKPOINT_DIR
        / f"alexnet_ecoset_{condition}"
    )

    condition_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shared_initialization_path = (
        CHECKPOINT_DIR
        / "shared_initialization.pt"
    )

    # ------------------------------------------------
    # Data
    # ------------------------------------------------

    print("\nCreating dataloaders...")

    train_loader, validation_loader = (
        create_dataloaders(
            condition=condition,
        )
    )

    print(
        f"Training batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader):,}"
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

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    model_kwargs = {
        "num_classes": NUM_CLASSES,
        "downscale": DOWNSCALE_FACTOR,
    }

    model = AlexNet(
        **model_kwargs,
    )

    # A resumed checkpoint already contains trained weights.
    if RESUME_CHECKPOINT is None:
        initial_state = (
            create_or_load_shared_initialization(
                initialization_path=(
                    shared_initialization_path
                ),
                model_class=AlexNet,
                model_kwargs=model_kwargs,
                seed=SEED,
                create_new=(
                    CREATE_NEW_INITIALIZATION
                ),
            )
        )

        model.load_state_dict(
            initial_state
        )

    model = model.to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ------------------------------------------------
    # Optimization
    # ------------------------------------------------

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    # Experiment metadata
    # ------------------------------------------------

    model_settings = {
        "architecture": "AlexNet",
        "num_classes": NUM_CLASSES,
        "downscale_factor": DOWNSCALE_FACTOR,
        "total_parameters": total_parameters,
        "trainable_parameters": (
            trainable_parameters
        ),
    }

    training_settings = {
        "condition": condition,
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
            tags=WANDB_TAGS,
            notes=WANDB_NOTES,
            model_settings=model_settings,
            training_settings=(
                training_settings
            ),
            data_path=TRAIN_DATA,
            train_indices_path=(
                TRAIN_INDICES_PATH
            ),
            val_indices_path=(
                VAL_INDICES_PATH
            ),
        )

        # Optional. Remove this if it noticeably slows training.
        wandb.watch(
            model,
            log="gradients",
            log_freq=500,
        )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    try:
        (
            history,
            best_validation_loss,
            final_epoch,
        ) = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
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
                f"AlexNet Ecoset {condition}"
            ),
            show=False,
        )

        print()
        print("=" * 70)
        print("Training complete")
        print("=" * 70)
        print(f"Condition: {condition}")
        print(
            f"Final completed epoch: "
            f"{final_epoch}"
        )
        print(
            f"Best validation loss: "
            f"{best_validation_loss:.4f}"
        )
        print(
            "Checkpoints saved to:"
            f"\n  {condition_checkpoint_dir}"
        )

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted by user."
        )
        print(
            "The latest completed epoch should "
            "remain available in last.pt."
        )
        raise

    except Exception:
        print(
            "\nTraining stopped because of an error."
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