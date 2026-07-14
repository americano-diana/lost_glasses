""""
Config script for paths, model settings, training settings, checkpoints, and reproducibility.
"""
# ------------------------------------------------
# Library imports
# ------------------------------------------------
from pathlib import Path
import torch

# ------------------------------------------------
# Local imports
# ------------------------------------------------
from pathlib import Path

# ============================================================
# Path configuration
# Note: All paths written with upper case
# ============================================================

# Set project root to the parent directory of current file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define folder with the raw dataset
CAT_DOG_DATA = PROJECT_ROOT / "cat_dog_data"

# List path to raw datasets
RAW_DATASET = CAT_DOG_DATA / "dataset"
RAW_BLUR_2 = CAT_DOG_DATA / "dataset_blur_2"
RAW_BLUR_5 = CAT_DOG_DATA / "dataset_blur_5"

OUTPUT = PROJECT_ROOT / "outputs"

# ============================================================
# Experiment config
# ============================================================

# Change name of experiment
EXPERIMENT_NAME = "First_expernet_test"

# Set seed number for reproducibility
SEED = 42

# Set device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# Model config
# ============================================================

# Model architecture params
NUM_CLASSES = 2
DOWNSCALE_FACTOR = 2 # None or 1 = full AlexNet, 2 = half, 4 = quarter

# ------------------------------------------------
# DataLoader
# ------------------------------------------------

IMAGE_SIZE = (256, 256) # Standard for AlexNet
BATCH_SIZE = 128
VALIDATION_RATIO = 0.20

# 0 on a notebook, 2 or 4 if on a cluster
NUM_WORKERS = 2

# Faster CPU-to-GPU transfer when CUDA is available. 
PIN_MEMORY = DEVICE.type == "cuda"

# ------------------------------------------------
# Training
# ------------------------------------------------

NUM_PRETRAINING_EPOCHS = 5
NUM_BLURRY_TRAINING_EPOCHS = 5

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4


# ------------------------------------------------
# Learning-rate scheduler
# ------------------------------------------------

# Reduce the learning rate when validation loss stops improving.
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 1e-7