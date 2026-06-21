"""Segmentation metrics for ice detection evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MetricAccumulator:
    """Running sums for batch-wise metric aggregation."""

    intersection: float = 0.0
    union: float = 0.0
    pred_sum: float = 0.0
    target_sum: float = 0.0
    true_positive: float = 0.0
    false_positive: float = 0.0
    false_negative: float = 0.0
    count: int = 0

    def update(self, pred_binary: torch.Tensor, target_binary: torch.Tensor) -> None:
        pred = pred_binary.detach().float()
        target = target_binary.detach().float()
        self.intersection += float((pred * target).sum())
        self.union += float(((pred + target) > 0).float().sum())
        self.pred_sum += float(pred.sum())
        self.target_sum += float(target.sum())
        self.true_positive += float((pred * target).sum())
        self.false_positive += float((pred * (1.0 - target)).sum())
        self.false_negative += float(((1.0 - pred) * target).sum())
        self.count += int(pred.shape[0])

    def compute(self) -> dict[str, float]:
        eps = 1e-8
        dice_den = self.pred_sum + self.target_sum
        dice = (2.0 * self.intersection + eps) / (dice_den + eps)
        iou = (self.intersection + eps) / (self.union + eps)
        precision = (self.true_positive + eps) / (self.true_positive + self.false_positive + eps)
        recall = (self.true_positive + eps) / (self.true_positive + self.false_negative + eps)
        f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def subsurface_binary_targets(
    y_soft: torch.Tensor,
    positive_class: int = 2,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Extract binary subsurface-ice target mask from soft labels."""
    return (y_soft[:, positive_class : positive_class + 1] >= threshold).float()


def subsurface_binary_predictions(
    logits: torch.Tensor,
    positive_class: int = 2,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Binarize subsurface logits for metric computation."""
    return (logits[:, positive_class : positive_class + 1] > threshold).float()


__all__ = [
    "MetricAccumulator",
    "subsurface_binary_predictions",
    "subsurface_binary_targets",
]
