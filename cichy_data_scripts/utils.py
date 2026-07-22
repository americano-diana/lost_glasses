"""
utils.py

General utilities for reproducibility, shared initialization,
and Weights & Biases setup.
"""

# ------------------------------------------------
# Library imports
# ------------------------------------------------

import random
from pathlib import Path

import numpy as np
import torch
import wandb


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    """
    Set Python, NumPy, and PyTorch random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Favor reproducibility over maximum speed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# Shared model initialization
# ============================================================

def create_or_load_shared_initialization(
    initialization_path: str | Path,
    model_class,
    model_kwargs: dict,
    seed: int,
    create_new: bool = False,
):
    """
    Create or load an identical initial state for paired models.

    The clear, mixed, and blur models can all begin from this exact
    state dictionary, reducing variability caused by initialization.

    Parameters
    ----------
    initialization_path:
        File in which the initial state dictionary is stored.

    model_class:
        Model class to instantiate, such as AlexNet.

    model_kwargs:
        Keyword arguments passed to the model constructor.

    seed:
        Random seed used before creating the initial model.

    create_new:
        If True, overwrite an existing shared initialization.

    Returns
    -------
    state_dict:
        Initial model state dictionary stored on CPU.
    """
    initialization_path = Path(initialization_path)

    initialization_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if initialization_path.exists() and not create_new:
        print(
            "Loading shared initialization from:"
            f"\n  {initialization_path}"
        )

        return torch.load(
            initialization_path,
            map_location="cpu",
            weights_only=True,
        )

    print("Creating a new shared model initialization.")

    # Reset immediately before constructing the model.
    torch.manual_seed(seed)

    initial_model = model_class(
        **model_kwargs,
    )

    state_dict = {
        name: value.detach().cpu().clone()
        for name, value
        in initial_model.state_dict().items()
    }

    torch.save(
        state_dict,
        initialization_path,
    )

    print(
        "Saved shared initialization to:"
        f"\n  {initialization_path}"
    )

    return state_dict


# ============================================================
# Weights & Biases setup
# ============================================================

def initialize_wandb(
    condition: str,
    seed: int,
    run_name_base: str,
    entity: str | None,
    project: str,
    tags: list[str],
    notes: str,
    model_settings: dict,
    training_settings: dict,
    data_path,
):
    """
    Initialize a Weights & Biases run.

    The final run name includes the training condition and seed.
    Model, training, and dataset settings are stored in the
    W&B run configuration.
    """
    run_name = (
        f"{run_name_base}_{condition}_seed{seed}"
    )

    run_tags = list(tags) + [
        condition,
        f"seed-{seed}",
    ]

    run_config = {
        **model_settings,
        **training_settings,
        "data_path": str(data_path),
    }

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        tags=run_tags,
        notes=notes,
        config=run_config,
    )

    return run