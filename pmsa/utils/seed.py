"""Reproducibility helpers. Every experiment fixes a seed and logs it."""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 0, deterministic: bool = True) -> int:
    """Seed python, numpy, and torch (if present). Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
