"""Valid-region-aware binary segmentation losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _validate(logits: torch.Tensor, targets: torch.Tensor, valid_region: torch.Tensor) -> None:
    if logits.shape != targets.shape or logits.shape != valid_region.shape:
        raise ValueError(
            f"logits, targets, and valid_region must have identical shapes: {logits.shape}, {targets.shape}, {valid_region.shape}"
        )
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Targets must be binary")
    if not torch.all((valid_region == 0) | (valid_region == 1)):
        raise ValueError("valid_region must be binary")
    if not torch.any(valid_region):
        raise ValueError("valid_region cannot be empty")


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_region: torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    _validate(logits, targets, valid_region)
    positive_weight = None
    if pos_weight is not None:
        if pos_weight <= 0:
            raise ValueError("pos_weight must be positive")
        positive_weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    elementwise = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=positive_weight
    )
    return (elementwise * valid_region).sum() / valid_region.sum().clamp_min(1)


def masked_soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_region: torch.Tensor,
    epsilon: float = 1.0,
) -> torch.Tensor:
    _validate(logits, targets, valid_region)
    probabilities = torch.sigmoid(logits) * valid_region
    masked_targets = targets * valid_region
    dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * masked_targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + masked_targets.sum(dim=dimensions)
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    return 1 - dice.mean()


class CombinedBCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight == 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight

    def components(
        self, logits: torch.Tensor, targets: torch.Tensor, valid_region: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        bce = masked_bce_with_logits(logits, targets, valid_region, self.pos_weight)
        dice = masked_soft_dice_loss(logits, targets, valid_region)
        return {"bce": bce, "dice": dice, "total": self.bce_weight * bce + self.dice_weight * dice}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_region: torch.Tensor) -> torch.Tensor:
        return self.components(logits, targets, valid_region)["total"]

