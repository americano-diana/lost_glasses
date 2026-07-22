""""
This training idea is to pre-train two models, 
one naive and one expert using clear and blurry datasets
"""
# ------------------------------------------------
# Library imports
# ------------------------------------------------
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# ------------------------------------------------
# Training loop for EcoNet
# ------------------------------------------------

def train_one_epoch(
    model,
    dataloader,
    loss_function,
    optimizer,
    device,
):
    """
    Train the model for one epoch.

    Returns
    -------
    average_loss:
        Mean loss across all training samples.

    accuracy:
        Training accuracy as a percentage.
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in tqdm(
        dataloader,
        desc="Training",
        leave=False,
    ):
        images = images.to(device)
        labels = labels.to(device)

        # Remove gradients from the previous batch.
        optimizer.zero_grad()

        # Forward pass.
        predictions = model(images)

        # Calculate the loss.
        loss = loss_function(predictions, labels)

        # Calculate gradients.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predicted_classes = predictions.argmax(dim=1)

        correct += (
            predicted_classes == labels
        ).sum().item()

    average_loss = total_loss / total_samples
    accuracy = 100 * correct / total_samples

    return average_loss, accuracy


# ------------------------------------------------
# Evaluate without training
# ------------------------------------------------

def evaluate(
    model,
    dataloader,
    loss_function,
    device,
):
    """
    Calculate loss and accuracy without updating the model.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    # No gradients are needed during evaluation.
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = loss_function(predictions, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predicted_classes = predictions.argmax(dim=1)

            correct += (
                predicted_classes == labels
            ).sum().item()

    average_loss = total_loss / total_samples
    accuracy = 100 * correct / total_samples

    return average_loss, accuracy


# ------------------------------------------------
# Save checkpoint
# ------------------------------------------------

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    validation_loss,
    model_settings,
):
    """
    Save everything needed to reload the model.

    The file uses a .pkl extension, but it is saved using torch.save.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_settings": model_settings,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    torch.save(checkpoint, path)


# ------------------------------------------------
# Train one complete phase
# ------------------------------------------------

def train_phase(
    model,
    train_loader,
    validation_loader,
    loss_function,
    optimizer,
    scheduler,
    device,
    num_epochs,
    phase_name,
    history,
    checkpoint_directory,
    model_settings,
    starting_epoch=0,
    save_best_checkpoint=False,
):
    """
    Train either the clear-image phase or blurry-image phase.

    Returns
    -------
    history:
        Updated training history.

    last_epoch:
        Number of the final completed epoch.

    best_validation_loss:
        Lowest validation loss reached in this phase.
    """
    best_validation_loss = float("inf")
    last_epoch = starting_epoch

    print(f"\n{phase_name}")
    print("-" * len(phase_name))

    for phase_epoch in range(1, num_epochs + 1):
        last_epoch += 1

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            dataloader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        # ReduceLROnPlateau watches validation loss.
        scheduler.step(validation_loss)

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history["epoch"].append(last_epoch)
        history["phase"].append(phase_name)

        history["training_loss"].append(train_loss)
        history["validation_loss"].append(
            validation_loss
        )

        history["training_accuracy"].append(
            train_accuracy
        )
        history["validation_accuracy"].append(
            validation_accuracy
        )

        history["learning_rate"].append(
            current_learning_rate
        )

        print(
            f"Epoch {phase_epoch}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"train accuracy: {train_accuracy:.2f}% | "
            f"validation accuracy: "
            f"{validation_accuracy:.2f}% | "
            f"lr: {current_learning_rate:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            if save_best_checkpoint:
                save_checkpoint(
                    path=(
                        checkpoint_directory
                        / "expert_best_validation.pkl"
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=last_epoch,
                    validation_loss=validation_loss,
                    model_settings=model_settings,
                )

                print("Saved new best checkpoint.")

    return history, last_epoch, best_validation_loss


# ------------------------------------------------
# Plot losses
# ------------------------------------------------

def plot_losses(
    history,
    pretraining_epochs,
    save_path,
):
    """
    Plot training and validation loss.

    The dashed line separates clear-image pretraining from blurry-image
    training.
    """
    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["training_loss"],
        "o-",
        label="Training loss",
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        "o-",
        label="Validation loss",
    )

    # Draw the phase boundary between the two stages.
    plt.axvline(
        x=pretraining_epochs + 0.5,
        linestyle="--",
        color="black",
        label="Start blurry training",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Eco_ExpertNet training")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.show()
    plt.close()