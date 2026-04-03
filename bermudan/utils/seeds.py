import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    """Resolve device string to torch.device.

    Parameters
    ----------
    device : str or None
        - None or "auto" : CUDA if available, else CPU
        - "cpu"          : force CPU
        - "cuda"         : force CUDA (raises if unavailable)
        - "cuda:0", etc. : specific GPU index
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
