"""
utils.py

General utilities for reproducibility and Weights & Biases setup.
"""

# ------------------------------------------------
# Library imports
# ------------------------------------------------

import random

import numpy as np
import torch
import wandb


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    """
    Set Python, NumPy, and PyTorch random seeds.

    cuDNN benchmark mode is enabled for faster training when
    image and batch shapes remain constant.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Faster for fixed-size inputs such as 224 × 224 images.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
    W&B configuration.
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

    return wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        tags=run_tags,
        notes=notes,
        config=run_config,
    )