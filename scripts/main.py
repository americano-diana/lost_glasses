"""
Main file to train ExpertNet using with clear-image pretraining and blurry images traning
"""
# ------------------------------------------------
# Library imports
# ------------------------------------------------
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim

# ------------------------------------------------
# Local imports
# ------------------------------------------------
import config
from utils import set_seed, print_evaluation
from data import create_dataloaders
from model import AlexNet
from training import (
    plot_losses,
    save_checkpoint,
    train_phase,
)

# ------------------------------------------------
# Main experiment
# ------------------------------------------------

def main():
    set_seed(config.SEED)
    print("Starting main training function")
    # --------------------------------------------
    # Create output directory
    # --------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_name = (
        f"{timestamp}_{config.EXPERIMENT_NAME}"
    )

    output_directory = (
        config.OUTPUT / run_name
    )

    checkpoint_directory = (
        output_directory / "checkpoints"
    )

    figure_directory = (
        output_directory / "figures"
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Device: {config.DEVICE}")
    print(f"Results directory: {output_directory}")

    # --------------------------------------------
    # Load datasets
    # --------------------------------------------

    print("Creating dataloaders")
    blur_data = "blur_2"

    data = create_dataloaders(
        dataset_names=(
            "clear",
            blur_data,
        ),
        batch_size=config.BATCH_SIZE,
        validation_ratio=config.VALIDATION_RATIO,
        random_state=config.SEED,
        num_workers=config.NUM_WORKERS,
    )

    clear_data = data["datasets"]["clear"]
    blurry_data = data["datasets"][blur_data]

    clear_train_loader = (
        clear_data["train_loader"]
    )

    clear_validation_loader = (
        clear_data["validation_loader"]
    )

    clear_test_loader = (
        clear_data["test_loader"]
    )

    blurry_train_loader = (
        blurry_data["train_loader"]
    )

    blurry_validation_loader = (
        blurry_data["validation_loader"]
    )

    blurry_test_loader = (
        blurry_data["test_loader"]
    )

    # --------------------------------------------
    # Create ExpertNet
    # --------------------------------------------
    "Creating model instance..."
    model = AlexNet(
        num_classes=config.NUM_CLASSES,
        downscale=config.DOWNSCALE_FACTOR,
    )

    model = model.to(config.DEVICE)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.SCHEDULER_MIN_LR,
        )
    )

    model_settings = {
        "model_name": "ExpertNet",
        "num_classes": config.NUM_CLASSES,
        "downscale_factor": (
            config.DOWNSCALE_FACTOR
        ),
    }

    # --------------------------------------------
    # Store results here
    # --------------------------------------------

    history = {
        "epoch": [],
        "phase": [],
        "training_loss": [],
        "validation_loss": [],
        "training_accuracy": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }

    evaluation_results = {}

    # --------------------------------------------
    # Evaluate before any training
    # --------------------------------------------

    print("\nBefore training")
    print("---------------")

    evaluation_results["before_clear"] = (
        print_evaluation(
            model=model,
            dataloader=clear_test_loader,
            loss_function=loss_function,
            device=config.DEVICE,
            title="Clear test set",
        )
    )

    evaluation_results["before_blurry"] = (
        print_evaluation(
            model=model,
            dataloader=blurry_test_loader,
            loss_function=loss_function,
            device=config.DEVICE,
            title="Blurry test set",
        )
    )

    # Save the randomly initialized model.
    save_checkpoint(
        path=(
            checkpoint_directory
            / "expert_before_training.pkl"
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        validation_loss=None,
        model_settings=model_settings,
    )

    # --------------------------------------------
    # Phase 1: clear-image pretraining
    # --------------------------------------------

    history, last_epoch, _ = train_phase(
        model=model,
        train_loader=clear_train_loader,
        validation_loader=clear_validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        num_epochs=(
            config.NUM_PRETRAINING_EPOCHS
        ),
        phase_name="Clear pretraining",
        history=history,
        checkpoint_directory=checkpoint_directory,
        model_settings=model_settings,
        starting_epoch=0,
        save_best_checkpoint=False,
    )

    print("\nAfter clear pretraining")
    print("-----------------------")

    evaluation_results["pretrained_clear"] = (
        print_evaluation(
            model=model,
            dataloader=clear_test_loader,
            loss_function=loss_function,
            device=config.DEVICE,
            title="Clear test set",
        )
    )

    evaluation_results["pretrained_blurry"] = (
        print_evaluation(
            model=model,
            dataloader=blurry_test_loader,
            loss_function=loss_function,
            device=config.DEVICE,
            title="Blurry test set",
        )
    )

    save_checkpoint(
        path=(
            checkpoint_directory
            / "expert_after_pretraining.pkl"
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=last_epoch,
        validation_loss=history[
            "validation_loss"
        ][-1],
        model_settings=model_settings,
    )

    # --------------------------------------------
    # Phase 2: blurry-image training
    # --------------------------------------------

    history, last_epoch, _ = train_phase(
        model=model,
        train_loader=blurry_train_loader,
        validation_loader=(
            blurry_validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        num_epochs=(
            config.NUM_BLURRY_TRAINING_EPOCHS
        ),
        phase_name="Blurry training",
        history=history,
        checkpoint_directory=checkpoint_directory,
        model_settings=model_settings,
        starting_epoch=last_epoch,
        save_best_checkpoint=True,
    )

    # --------------------------------------------
    # Evaluate final model
    # --------------------------------------------

    print("\nAfter blurry training")
    print("---------------------")

    evaluation_results["final_clear"] = (
        print_evaluation(
            model=model,
            dataloader=clear_test_loader,
            loss_function=loss_function,
            device=config.DEVICE,
            title="Clear test set",
        )
    )

    evaluation_results["final_blurry"] = (
        print_evaluation(
            model=model,
            dataloader=blurry_test_loader,
            loss_function=loss_function,
            device=config.DEVICE,
            title="Blurry test set",
        )
    )

    # Save the model from the final epoch.
    save_checkpoint(
        path=(
            checkpoint_directory
            / "expert_final_model.pkl"
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=last_epoch,
        validation_loss=history[
            "validation_loss"
        ][-1],
        model_settings=model_settings,
    )

    # --------------------------------------------
    # Save results
    # --------------------------------------------

    results = {
        "history": history,
        "evaluation": evaluation_results,
        "model_settings": model_settings,
    }

    results_path = (
        output_directory / "results.json"
    )

    with results_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    # --------------------------------------------
    # Plot loss
    # --------------------------------------------

    plot_losses(
        history=history,
        pretraining_epochs=(
            config.NUM_PRETRAINING_EPOCHS
        ),
        save_path=(
            figure_directory
            / "expertnet_loss.png"
        ),
    )

    print("\nTraining complete.")
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    main()