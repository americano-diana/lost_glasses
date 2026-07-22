"""
training.py

Reusable training utilities for pretraining the custom AlexNet
architecture on Ecoset.

Each experiment trains one independent model under one condition:

    clear:
        All training images remain clear.

    mixed:
        Training images are dynamically sampled as clear or blurred.

    blur:
        Every training image is dynamically blurred.

The condition is controlled by the dataloader and passed here only for
logging and checkpoint metadata.
"""

# ------------------------------------------------
# Library imports
# ------------------------------------------------

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ============================================================
# Train for one epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    use_amp: bool,
):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model:
        Neural network to train.

    dataloader:
        Training DataLoader.

    loss_function:
        Classification loss, normally CrossEntropyLoss.

    optimizer:
        PyTorch optimizer.

    device:
        CPU or CUDA device.

    scaler:
        GradScaler used for mixed-precision training.

    use_amp:
        Whether automatic mixed precision is enabled.

    Returns
    -------
    metrics:
        Dictionary containing mean loss, accuracy percentage,
        and sample count.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
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

        optimizer.zero_grad(
            set_to_none=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            predictions = model(images)

            loss = loss_function(
                predictions,
                labels,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)

        total_loss += (
            loss.detach().item()
            * batch_size
        )

        predicted_classes = predictions.argmax(
            dim=1
        )

        total_correct += (
            predicted_classes == labels
        ).sum().item()

        total_samples += batch_size

        running_loss = (
            total_loss
            / total_samples
        )

        running_accuracy = (
            100.0
            * total_correct
            / total_samples
        )

        progress_bar.set_postfix(
            loss=f"{running_loss:.4f}",
            accuracy=f"{running_accuracy:.2f}%",
        )

    if total_samples == 0:
        raise RuntimeError(
            "The training DataLoader returned no samples."
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": (
            100.0
            * total_correct
            / total_samples
        ),
        "samples": total_samples,
    }


# ============================================================
# Evaluate without training
# ============================================================

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader,
    loss_function: nn.Module,
    device: torch.device,
    use_amp: bool,
):
    """
    Evaluate loss and accuracy without updating model parameters.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
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
            enabled=use_amp,
        ):
            predictions = model(images)

            loss = loss_function(
                predictions,
                labels,
            )

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predicted_classes = predictions.argmax(
            dim=1
        )

        total_correct += (
            predicted_classes == labels
        ).sum().item()

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "The validation DataLoader returned no samples."
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": (
            100.0
            * total_correct
            / total_samples
        ),
        "samples": total_samples,
    }


# ============================================================
# Checkpoint utilities
# ============================================================

def save_checkpoint(
    path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    history: dict,
    model_settings: dict,
    training_settings: dict,
    condition: str,
):
    """
    Save everything needed to resume training or reload the model.

    The recommended extension is .pt because the checkpoint is
    produced with torch.save.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "condition": condition,
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": (
            epochs_without_improvement
        ),
        "model_settings": model_settings,
        "training_settings": training_settings,
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device="cpu",
):
    """
    Load a saved checkpoint.

    Returns
    -------
    checkpoint:
        Full checkpoint dictionary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and checkpoint.get(
            "optimizer_state_dict"
        ) is not None
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint.get(
            "scheduler_state_dict"
        ) is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if (
        scaler is not None
        and checkpoint.get(
            "scaler_state_dict"
        ) is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    return checkpoint


# ============================================================
# History utilities
# ============================================================

def create_empty_history():
    """
    Create a new training-history dictionary.
    """
    return {
        "epoch": [],
        "training_loss": [],
        "validation_loss": [],
        "training_accuracy": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }


def save_history_json(
    history: dict,
    save_path,
):
    """
    Save training history as JSON.
    """
    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )


def save_history_csv(
    history: dict,
    save_path,
):
    """
    Save training history as CSV.
    """
    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        history.keys()
    )

    number_of_epochs = len(
        history["epoch"]
    )

    with open(
        save_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index in range(number_of_epochs):
            row = {
                key: history[key][index]
                for key in fieldnames
            }

            writer.writerow(row)


# ============================================================
# Complete training run
# ============================================================

def train_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    max_epochs: int,
    condition: str,
    checkpoint_directory,
    model_settings: dict,
    training_settings: dict,
    early_stopping_patience: int,
    use_amp: bool = True,
    resume_checkpoint=None,
    use_wandb: bool = False,
):
    """
    Train one model under one Ecoset condition.

    Parameters
    ----------
    condition:
        One of "clear", "mixed", or "blur".

    resume_checkpoint:
        Optional path to a saved last.pt checkpoint.

    Returns
    -------
    history:
        Complete training history.

    best_validation_loss:
        Lowest validation loss observed.

    final_epoch:
        Final completed epoch.
    """
    if condition not in {
        "clear",
        "mixed",
        "blur",
    }:
        raise ValueError(
            "condition must be 'clear', "
            "'mixed', or 'blur'."
        )

    checkpoint_directory = Path(
        checkpoint_directory
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    history = create_empty_history()

    starting_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(
            path=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        saved_condition = checkpoint.get(
            "condition"
        )

        if (
            saved_condition is not None
            and saved_condition != condition
        ):
            raise ValueError(
                "Checkpoint condition does not match "
                f"the requested condition: "
                f"{saved_condition!r} versus "
                f"{condition!r}."
            )

        starting_epoch = checkpoint["epoch"]

        best_validation_loss = checkpoint.get(
            "best_validation_loss",
            float("inf"),
        )

        epochs_without_improvement = (
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )

        history = checkpoint.get(
            "history",
            create_empty_history(),
        )

        print(
            f"Resuming from epoch "
            f"{starting_epoch}."
        )

    if starting_epoch >= max_epochs:
        print(
            "The checkpoint has already reached "
            f"MAX_EPOCHS={max_epochs}."
        )

        return (
            history,
            best_validation_loss,
            starting_epoch,
        )

    print()
    print("=" * 70)
    print(
        f"Training condition: {condition}"
    )
    print(
        f"Starting epoch: {starting_epoch + 1}"
    )
    print(
        f"Maximum epoch: {max_epochs}"
    )
    print(
        f"Automatic mixed precision: {amp_enabled}"
    )
    print("=" * 70)

    final_epoch = starting_epoch

    for epoch in range(
        starting_epoch + 1,
        max_epochs + 1,
    ):
        final_epoch = epoch

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=amp_enabled,
        )

        validation_metrics = evaluate(
            model=model,
            dataloader=validation_loader,
            loss_function=loss_function,
            device=device,
            use_amp=amp_enabled,
        )

        validation_loss = (
            validation_metrics["loss"]
        )

        # ReduceLROnPlateau uses validation loss.
        if scheduler is not None:
            scheduler.step(
                validation_loss
            )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history["epoch"].append(epoch)

        history["training_loss"].append(
            train_metrics["loss"]
        )

        history["validation_loss"].append(
            validation_metrics["loss"]
        )

        history["training_accuracy"].append(
            train_metrics["accuracy"]
        )

        history["validation_accuracy"].append(
            validation_metrics["accuracy"]
        )

        history["learning_rate"].append(
            current_learning_rate
        )

        improved = (
            validation_loss
            < best_validation_loss
        )

        if improved:
            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=(
                    checkpoint_directory
                    / "best.pt"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_validation_loss=(
                    best_validation_loss
                ),
                epochs_without_improvement=(
                    epochs_without_improvement
                ),
                history=history,
                model_settings=model_settings,
                training_settings=(
                    training_settings
                ),
                condition=condition,
            )

            checkpoint_message = (
                " | saved best.pt"
            )

        else:
            epochs_without_improvement += 1
            checkpoint_message = ""

        # Save a resumable checkpoint after every epoch.
        save_checkpoint(
            path=(
                checkpoint_directory
                / "last.pt"
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_validation_loss=(
                best_validation_loss
            ),
            epochs_without_improvement=(
                epochs_without_improvement
            ),
            history=history,
            model_settings=model_settings,
            training_settings=(
                training_settings
            ),
            condition=condition,
        )

        save_history_json(
            history=history,
            save_path=(
                checkpoint_directory
                / "history.json"
            ),
        )

        save_history_csv(
            history=history,
            save_path=(
                checkpoint_directory
                / "history.csv"
            ),
        )

        print(
            f"Epoch {epoch:03d}/{max_epochs:03d} | "
            f"train loss: "
            f"{train_metrics['loss']:.4f} | "
            f"val loss: "
            f"{validation_metrics['loss']:.4f} | "
            f"train acc: "
            f"{train_metrics['accuracy']:.2f}% | "
            f"val acc: "
            f"{validation_metrics['accuracy']:.2f}% | "
            f"lr: {current_learning_rate:.2e}"
            f"{checkpoint_message}"
        )

        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": (
                        train_metrics["loss"]
                    ),
                    "train/accuracy": (
                        train_metrics["accuracy"]
                    ),
                    "validation/loss": (
                        validation_metrics["loss"]
                    ),
                    "validation/accuracy": (
                        validation_metrics[
                            "accuracy"
                        ]
                    ),
                    "learning_rate": (
                        current_learning_rate
                    ),
                },
                step=epoch,
            )

        if (
            epochs_without_improvement
            >= early_stopping_patience
        ):
            print(
                "Early stopping triggered after "
                f"{epochs_without_improvement} "
                "epochs without validation-loss "
                "improvement."
            )
            break

    return (
        history,
        best_validation_loss,
        final_epoch,
    )


# ============================================================
# Plot training results
# ============================================================

def plot_training_history(
    history: dict,
    save_directory,
    title: str,
    show: bool = False,
):
    """
    Save separate loss and accuracy figures.

    Separate figures are used so that the scales remain readable.
    """
    save_directory = Path(
        save_directory
    )

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = history["epoch"]

    if len(epochs) == 0:
        raise ValueError(
            "Training history is empty."
        )

    # ------------------------------------------------
    # Loss plot
    # ------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["training_loss"],
        marker="o",
        label="Training loss",
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        marker="o",
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(
        f"{title}: loss"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_directory
        / "loss_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close()

    # ------------------------------------------------
    # Accuracy plot
    # ------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["training_accuracy"],
        marker="o",
        label="Training accuracy",
    )

    plt.plot(
        epochs,
        history["validation_accuracy"],
        marker="o",
        label="Validation accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(
        f"{title}: accuracy"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_directory
        / "accuracy_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close()