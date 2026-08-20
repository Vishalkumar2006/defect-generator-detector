"""Best-effort deterministic controls for PyTorch experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def configure_reproducibility(seed: int = 42, deterministic: bool = True, warn_only: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False if deterministic else torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = deterministic

