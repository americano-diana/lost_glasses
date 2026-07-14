""""
Util functions
"""

import random
import torch
import numpy as np

from training import evaluate

############
# Helper functions
############

def set_seed(seed=None, seed_torch=True):
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')

# In case that `DataLoader` is used
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: For this notebook to perform best, "
        "if possible, in the menu under `Runtime` -> "
        "`Change runtime type.`  select `GPU` ")
  else:
    print("GPU is enabled in this notebook.")

  return device


def print_evaluation(
    model,
    dataloader,
    loss_function,
    device,
    title,
):
    """
    Evaluate a model and print the result.
    """
    loss, accuracy = evaluate(
        model=model,
        dataloader=dataloader,
        loss_function=loss_function,
        device=device,
    )

    print(
        f"{title}: "
        f"loss = {loss:.4f}, "
        f"accuracy = {accuracy:.2f}%"
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
    }
