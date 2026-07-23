"""
Config script for paths, model settings, training settings, checkpoints, and reproducibility.
"""
# ------------------------------------------------
# Library imports
# ------------------------------------------------
from pathlib import Path
import torch
import os
from dotenv import load_dotenv # This is to load local environment variables like paths

# ============================================================
# Path configuration
# Note: All paths written with upper case
# ============================================================

load_dotenv() 

# Loading scratch paths
USERNAME = os.environ.get("USER", "unknown_user")
SCRATCH_ROOT = Path(os.environ.get("SCRATCH", f"/scratch/{USERNAME}"))
HOME_ROOT = Path.home()  

# Setting project root
PROJECT_NAME = "lost_glasses"
PROJECT_ROOT = SCRATCH_ROOT / PROJECT_NAME

# Imagenet dataset path
imagenet_path = os.getenv("TRAIN_DATA_PATH") 
if imagenet_path is None:
    raise RuntimeError( "TRAIN_DATA_PATH was not found in the environment. " "Add it to your .env file." ) 
IMAGENET_ROOT = Path(imagenet_path)

if not (IMAGENET_ROOT / "train").is_dir():
    raise FileNotFoundError(
        f"ImageNet train directory not found: "
        f"{IMAGENET_ROOT / 'train'}"
    )

if not (IMAGENET_ROOT / "val").is_dir():
    raise FileNotFoundError(
        f"ImageNet validation directory not found: "
        f"{IMAGENET_ROOT / 'val'}"
    )

# Sub-indices splits testing
SPLIT_DIR = PROJECT_ROOT / "splits" 
TRAIN_INDICES_PATH = SPLIT_DIR / "imagenet_train_50000_seed42.npy"
VAL_INDICES_PATH =  SPLIT_DIR / "imagenet_val_2000_seed42.npy"

# Subsets for fine-tuning
TRAIN_SUBSET_SIZE = 50_000
VAL_SUBSET_SIZE = 2_000

# Output and checkpoints in /home
# Checkpoints and logs are kept under /home because scratch # storage may be temporary or periodically cleaned. 
OUTPUT_DIR = ( HOME_ROOT / "projects" / PROJECT_NAME / "outputs" )
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# Define folder with the raw fmri data
CICHY_DATA = PROJECT_ROOT / "cichy_data"
CICHY_IMAGE_DIR = ( CICHY_DATA / "92_Image_Set" / "92images")

# ============================================================
# Experiment config
# ============================================================

# Change name of experiment
EXPERIMENT_NAME = "finetuning_alexnet_blurry"

# Set seed number for reproducibility
SEED = 42

# Set device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load pre-trained alexnet, freeze layers, train only last network
TRAINABLE_CLASSIFIER_LAYERS = 2

# --------------------------------------------------
# Dynamic blur params
# --------------------------------------------------

# Proportion of training presentations that will be blurred
# in the mixed condition.
BLUR_PROBABILITY = 0.5

# Gaussian blur requires a positive odd integer.
BLUR_KERNEL_SIZE = 11

# A new sigma is sampled for every blurred image.
BLUR_SIGMA_MIN = 0.1
BLUR_SIGMA_MAX = 5.0

# ------------------------------------------------
# DataLoader
# ------------------------------------------------

# Ecoset images are stored at 256 × 256. During training, # use RandomCrop(224); during validation, use CenterCrop(224).
IMAGE_SIZE = 224
RESIZE_SIZE = 256

MEAN = [ 0.485, 0.456, 0.406, ] 
STD = [ 0.229, 0.224, 0.225, ]

BATCH_SIZE_TRAIN = 256
BATCH_SIZE_VAL = 256

# 0 on a notebook, 2 or 4 if on a cluster
NUM_WORKERS = 2
PIN_MEMORY = True

# persistent_workers is invalid when NUM_WORKERS == 0. 
PERSISTENT_WORKERS = NUM_WORKERS > 0 
PREFETCH_FACTOR = 2

USE_AMP = True

# ------------------------------------------------
# Training
# ------------------------------------------------
# Each condition is a separate training run
# choose clear, mixed or blur
TRAINING_CONDITION = "mixed"

MAX_EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 2

# ------------------------------------------------
# Learning-rate scheduler
# ------------------------------------------------

# Reduce the learning rate when validation loss stops improving, intended for ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 1e-6

# ============================================================ 
# # Checkpointing 
# # ============================================================ 
#AVE_LAST_CHECKPOINT = True 
#AVE_BEST_CHECKPOINT = True 

# Resume from this path when provided. 
# # Example: # RESUME_CHECKPOINT = CHECKPOINT_DIR / "alexnet_ecoset_clear" / "last.pt" 
RESUME_CHECKPOINT = None
# ------------------------------------------------
# Optional: Weights & Biases set-up to see training and val losses
# ------------------------------------------------
USE_WANDB = True
WANDB_ENTITY = os.getenv("WANDB_PATH")
WANDB_PROJECT = "lost_glasses"      
WANDB_RUN_NAME = "imagenet_alexnet_blur_finetuning"
WANDB_TAGS = ["alexnet", "pre-train", "imagenet"]
WANDB_NOTES = (
    "Fine-tuning the final two classifier layers of an "
    "ImageNet-pretrained AlexNet using a dynamic mixture "
    "of clear and Gaussian-blurred ImageNet images."
)