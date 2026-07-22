"""
train.py

Reusable training utilities for training  AlexNet

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
    model,
    dataloader,
    loss_function,
    optimizer,
    device,
    scaler,
    use_amp=True,
):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for batch_index, (images, labels) in enumerate(
        progress_bar
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).long()

        if labels.min().item() < 0:
            raise ValueError(
                f"Negative label in batch {batch_index}: "
                f"{labels.min().item()}"
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)

            if labels.max().item() >= logits.shape[1]:
                raise ValueError(
                    f"Label {labels.max().item()} exceeds "
                    f"model output dimension "
                    f"{logits.shape[1]}."
                )

            loss = loss_function(
                logits,
                labels,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at batch "
                f"{batch_index}: {loss.item()}"
            )

        batch_loss = loss.detach().item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)

        predictions = logits.argmax(
            dim=1
        )

        running_loss += (
            batch_loss * batch_size
        )

        running_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        average_loss = (
            running_loss / total_samples
        )

        average_accuracy = (
            100.0
            * running_correct
            / total_samples
        )

        progress_bar.set_postfix(
            batch_loss=f"{batch_loss:.4f}",
            mean_loss=f"{average_loss:.4f}",
            accuracy=f"{average_accuracy:.2f}%",
        )

    if total_samples == 0:
        raise RuntimeError(
            "The training DataLoader returned no samples."
        )

    return {
        "loss": running_loss / total_samples,
        "accuracy": (
            100.0
            * running_correct
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
    use_amp: bool = True,
):
    """
    Evaluate model loss and accuracy without updating parameters.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    )

    for batch_index, (images, labels) in enumerate(
        progress_bar
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).long()

        if labels.min().item() < 0:
            raise ValueError(
                f"Negative label in validation batch "
                f"{batch_index}: {labels.min().item()}"
            )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)

            if labels.max().item() >= logits.shape[1]:
                raise ValueError(
                    f"Label {labels.max().item()} exceeds "
                    f"model output dimension "
                    f"{logits.shape[1]}."
                )

            loss = loss_function(
                logits,
                labels,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss at batch "
                f"{batch_index}: {loss.item()}"
            )

        batch_size = images.size(0)

        predicted_classes = logits.argmax(
            dim=1
        )

        total_loss += (
            loss.detach().item()
            * batch_size
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
    Save everything needed to resume training.

    The checkpoint contains model weights, optimizer state,
    scheduler state, AMP scaler state, history, and metadata.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "condition": condition,
        "best_validation_loss": (
            best_validation_loss
        ),
        "epochs_without_improvement": (
            epochs_without_improvement
        ),
        "model_settings": model_settings,
        "training_settings": training_settings,
        "history": history,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
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
    Load a checkpoint and restore available training state.

    Returns
    -------
    checkpoint:
        The complete checkpoint dictionary.
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

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
    )

    if (
        optimizer is not None
        and optimizer_state is not None
    ):
        optimizer.load_state_dict(
            optimizer_state
        )

    scheduler_state = checkpoint.get(
        "scheduler_state_dict"
    )

    if (
        scheduler is not None
        and scheduler_state is not None
    ):
        scheduler.load_state_dict(
            scheduler_state
        )

    scaler_state = checkpoint.get(
        "scaler_state_dict"
    )

    if (
        scaler is not None
        and scaler_state is not None
    ):
        scaler.load_state_dict(
            scaler_state
        )

    return checkpoint

# ============================================================
# History utilities
# ============================================================

# ============================================================
# History utilities
# ============================================================

def create_empty_history():
    """
    Create the training-history dictionary.

    Clear and blurred validation metrics are stored separately.
    The selection loss is the mean of clear and blurred
    validation loss and is used to select best.pt.
    """
    return {
        "epoch": [],
        "training_loss": [],
        "training_accuracy": [],
        "clear_validation_loss": [],
        "clear_validation_accuracy": [],
        "blur_validation_loss": [],
        "blur_validation_accuracy": [],
        "selection_loss": [],
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
    clear_validation_loader,
    blur_validation_loader,
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
    Fine-tune an ImageNet-pretrained AlexNet.

    The model is trained using the requested training condition
    and evaluated after every epoch on:

    1. clear ImageNet validation images;
    2. blurred versions of the same validation images.

    best.pt is selected using the mean of clear and blurred
    validation loss.

    Parameters
    ----------
    condition:
        One of "clear", "mixed", or "blur".

    resume_checkpoint:
        Optional path to a previous last.pt checkpoint.

    Returns
    -------
    history:
        Complete training history.

    best_selection_loss:
        Lowest mean clear/blur validation loss.

    final_epoch:
        Last completed epoch.
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
    best_selection_loss = float("inf")
    epochs_without_improvement = 0

    # ------------------------------------------------
    # Resume
    # ------------------------------------------------

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
                "the requested condition: "
                f"{saved_condition!r} versus "
                f"{condition!r}."
            )

        starting_epoch = checkpoint["epoch"]

        # Old checkpoints may use this name.
        best_selection_loss = checkpoint.get(
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
            f"Resuming from completed epoch "
            f"{starting_epoch}."
        )

    if starting_epoch >= max_epochs:
        print(
            "The checkpoint has already reached "
            f"MAX_EPOCHS={max_epochs}."
        )

        return (
            history,
            best_selection_loss,
            starting_epoch,
        )

    print()
    print("=" * 70)
    print("ImageNet AlexNet fine-tuning")
    print("=" * 70)
    print(f"Training condition: {condition}")
    print(f"Starting epoch: {starting_epoch + 1}")
    print(f"Maximum epoch: {max_epochs}")
    print(f"Automatic mixed precision: {amp_enabled}")
    print("=" * 70)

    final_epoch = starting_epoch

    # ------------------------------------------------
    # Epoch loop
    # ------------------------------------------------

    for epoch in range(
        starting_epoch + 1,
        max_epochs + 1,
    ):
        final_epoch = epoch

        print()
        print(
            f"Epoch {epoch}/{max_epochs}"
        )

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=amp_enabled,
        )

        print("Evaluating clear validation images...")

        clear_validation_metrics = evaluate(
            model=model,
            dataloader=clear_validation_loader,
            loss_function=loss_function,
            device=device,
            use_amp=amp_enabled,
        )

        print("Evaluating blurred validation images...")

        blur_validation_metrics = evaluate(
            model=model,
            dataloader=blur_validation_loader,
            loss_function=loss_function,
            device=device,
            use_amp=amp_enabled,
        )

        clear_validation_loss = (
            clear_validation_metrics["loss"]
        )

        blur_validation_loss = (
            blur_validation_metrics["loss"]
        )

        # Equal weighting prevents optimization from focusing
        # exclusively on clear or blurred validation images.
        selection_loss = (
            clear_validation_loss
            + blur_validation_loss
        ) / 2.0

        # ReduceLROnPlateau follows the same metric used to
        # select the best checkpoint.
        if scheduler is not None:
            scheduler.step(
                selection_loss
            )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        # ------------------------------------------------
        # History
        # ------------------------------------------------

        history["epoch"].append(
            epoch
        )

        history["training_loss"].append(
            train_metrics["loss"]
        )

        history["training_accuracy"].append(
            train_metrics["accuracy"]
        )

        history[
            "clear_validation_loss"
        ].append(
            clear_validation_metrics["loss"]
        )

        history[
            "clear_validation_accuracy"
        ].append(
            clear_validation_metrics["accuracy"]
        )

        history[
            "blur_validation_loss"
        ].append(
            blur_validation_metrics["loss"]
        )

        history[
            "blur_validation_accuracy"
        ].append(
            blur_validation_metrics["accuracy"]
        )

        history["selection_loss"].append(
            selection_loss
        )

        history["learning_rate"].append(
            current_learning_rate
        )

        # ------------------------------------------------
        # Best checkpoint
        # ------------------------------------------------

        improved = (
            selection_loss
            < best_selection_loss
        )

        if improved:
            best_selection_loss = (
                selection_loss
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
                    best_selection_loss
                ),
                epochs_without_improvement=(
                    epochs_without_improvement
                ),
                history=history,
                model_settings=model_settings,
                training_settings=training_settings,
                condition=condition,
            )

            checkpoint_message = (
                " | saved best.pt"
            )

        else:
            epochs_without_improvement += 1
            checkpoint_message = ""

        # ------------------------------------------------
        # Latest resumable checkpoint
        # ------------------------------------------------

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
                best_selection_loss
            ),
            epochs_without_improvement=(
                epochs_without_improvement
            ),
            history=history,
            model_settings=model_settings,
            training_settings=training_settings,
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

        # ------------------------------------------------
        # Console summary
        # ------------------------------------------------

        print(
            f"Epoch {epoch:03d}/{max_epochs:03d} | "
            f"train loss: "
            f"{train_metrics['loss']:.4f} | "
            f"train acc: "
            f"{train_metrics['accuracy']:.2f}% | "
            f"clear val loss: "
            f"{clear_validation_metrics['loss']:.4f} | "
            f"clear val acc: "
            f"{clear_validation_metrics['accuracy']:.2f}% | "
            f"blur val loss: "
            f"{blur_validation_metrics['loss']:.4f} | "
            f"blur val acc: "
            f"{blur_validation_metrics['accuracy']:.2f}% | "
            f"selection loss: "
            f"{selection_loss:.4f} | "
            f"lr: {current_learning_rate:.2e}"
            f"{checkpoint_message}"
        )

        # ------------------------------------------------
        # W&B
        # ------------------------------------------------

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
                    "validation_clear/loss": (
                        clear_validation_metrics[
                            "loss"
                        ]
                    ),
                    "validation_clear/accuracy": (
                        clear_validation_metrics[
                            "accuracy"
                        ]
                    ),
                    "validation_blur/loss": (
                        blur_validation_metrics[
                            "loss"
                        ]
                    ),
                    "validation_blur/accuracy": (
                        blur_validation_metrics[
                            "accuracy"
                        ]
                    ),
                    "validation/selection_loss": (
                        selection_loss
                    ),
                    "learning_rate": (
                        current_learning_rate
                    ),
                },
                step=epoch,
            )

        # ------------------------------------------------
        # Early stopping
        # ------------------------------------------------

        if (
            epochs_without_improvement
            >= early_stopping_patience
        ):
            print(
                "Early stopping triggered after "
                f"{epochs_without_improvement} "
                "epochs without improvement in "
                "mean clear/blur validation loss."
            )
            break

    return (
        history,
        best_selection_loss,
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
    Save fine-tuning loss and accuracy plots.
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
        figsize=(9, 5)
    )

    plt.plot(
        epochs,
        history["training_loss"],
        marker="o",
        label="Training loss",
    )

    plt.plot(
        epochs,
        history["clear_validation_loss"],
        marker="o",
        label="Clear validation loss",
    )

    plt.plot(
        epochs,
        history["blur_validation_loss"],
        marker="o",
        label="Blur validation loss",
    )

    plt.plot(
        epochs,
        history["selection_loss"],
        marker="o",
        linestyle="--",
        label="Mean validation loss",
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
        figsize=(9, 5)
    )

    plt.plot(
        epochs,
        history["training_accuracy"],
        marker="o",
        label="Training accuracy",
    )

    plt.plot(
        epochs,
        history[
            "clear_validation_accuracy"
        ],
        marker="o",
        label="Clear validation accuracy",
    )

    plt.plot(
        epochs,
        history[
            "blur_validation_accuracy"
        ],
        marker="o",
        label="Blur validation accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Top-1 accuracy (%)")
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