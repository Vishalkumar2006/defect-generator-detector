"""Deterministic sampling for imbalanced development training."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def deterministic_weighted_sampler(labels: list[bool], seed: int = 42, num_samples: int | None = None):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Balanced sampling requires both defective and normal samples")
    weights = [0.5 / positives if label else 0.5 / negatives for label in labels]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples or len(labels),
        replacement=True,
        generator=generator,
    )


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

