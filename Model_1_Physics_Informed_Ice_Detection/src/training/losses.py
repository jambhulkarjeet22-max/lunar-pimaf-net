"""Production training loss composition for LUNAR-PIMAF-Net."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses import EvidentialLoss, PhysicsLoss, PhysicsLossWeights
from src.models.lunar_pimaf_net import CHANNEL_PHYSICS_PRIORS, LunarPIMAFOutput
from src.models.prediction_heads import NUM_SEGMENTATION_CLASSES


@dataclass
class TrainingLossOutput:
    """Detailed breakdown of the composite training objective."""

    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor
    physics: torch.Tensor
    uncertainty: torch.Tensor


class DiceLoss(nn.Module):
    """Soft Dice loss for binary or multi-class segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            targets = targets.float()
        else:
            probs = F.softmax(logits, dim=1)
            if targets.shape[1] != logits.shape[1]:
                raise ValueError("DiceLoss channel mismatch between logits and targets.")

        dims = tuple(range(2, probs.dim()))
        intersection = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class ProductionTrainingLoss(nn.Module):
    """Combine BCE, Dice, physics consistency, and evidential uncertainty losses."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        physics_weight: float = 0.3,
        uncertainty_weight: float = 0.5,
        positive_class: int = 2,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.physics_weight = physics_weight
        self.uncertainty_weight = uncertainty_weight
        self.positive_class = positive_class

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.physics = PhysicsLoss(PhysicsLossWeights())
        self.uncertainty = EvidentialLoss(num_classes=NUM_SEGMENTATION_CLASSES)

    def forward(
        self,
        outputs: LunarPIMAFOutput,
        batch: dict[str, torch.Tensor],
        kl_annealing: float = 1.0,
    ) -> TrainingLossOutput:
        logits = outputs["segmentation_logits"]
        y_soft = batch["y_soft"]
        target_class = batch["target_class"]

        subsurface_logits = logits[:, self.positive_class : self.positive_class + 1]
        subsurface_target = y_soft[:, self.positive_class : self.positive_class + 1]

        bce_loss = self.bce(subsurface_logits, subsurface_target)
        dice_loss = self.dice(subsurface_logits, subsurface_target)

        physics_priors = batch["inputs"][:, CHANNEL_PHYSICS_PRIORS]
        physics_loss = self.physics(
            outputs["physics_residuals"],
            outputs["latent_physics"],
            physics_priors,
        )

        uncertainty_loss = self.uncertainty(
            outputs["dirichlet_alpha"],
            target_class,
            kl_annealing=kl_annealing,
        )

        total = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.physics_weight * physics_loss
            + self.uncertainty_weight * uncertainty_loss
        )

        return TrainingLossOutput(
            total=total,
            bce=bce_loss,
            dice=dice_loss,
            physics=physics_loss,
            uncertainty=uncertainty_loss,
        )


__all__ = [
    "DiceLoss",
    "ProductionTrainingLoss",
    "TrainingLossOutput",
]
