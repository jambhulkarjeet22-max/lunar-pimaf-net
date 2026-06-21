"""Evaluation metrics for rover hazard and navigation prediction."""

from __future__ import annotations

from typing import Any

import torch


class MetricsCalculator:
    """Accumulates traversability regression and hazard classification metrics."""

    def __init__(self, hazard_threshold: float = 0.5) -> None:
        self.hazard_threshold = hazard_threshold
        self.reset()

    def reset(self) -> None:
        self._trav_sse = 0.0
        self._trav_count = 0
        self._crater_correct = 0
        self._crater_total = 0
        self._boulder_correct = 0
        self._boulder_total = 0
        self._slope_correct = 0
        self._slope_total = 0

    @torch.no_grad()
    def update(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> None:
        trav_pred = predictions["traversability_score"]
        trav_tgt = targets["traversability_score"]
        self._trav_sse += torch.sum((trav_pred - trav_tgt) ** 2).item()
        self._trav_count += trav_pred.numel()

        # Binary accuracies for hazards
        crater_pred = (predictions["crater_hazard_probability"] >= self.hazard_threshold).float()
        crater_tgt = (targets["crater_hazard_probability"] >= self.hazard_threshold).float()
        self._crater_correct += (crater_pred == crater_tgt).sum().item()
        self._crater_total += crater_pred.numel()

        boulder_pred = (predictions["boulder_hazard_probability"] >= self.hazard_threshold).float()
        boulder_tgt = (targets["boulder_hazard_probability"] >= self.hazard_threshold).float()
        self._boulder_correct += (boulder_pred == boulder_tgt).sum().item()
        self._boulder_total += boulder_pred.numel()

        slope_pred = (predictions["slope_hazard_score"] >= self.hazard_threshold).float()
        slope_tgt = (targets["slope_hazard_score"] >= self.hazard_threshold).float()
        self._slope_correct += (slope_pred == slope_tgt).sum().item()
        self._slope_total += slope_pred.numel()

    def compute(self) -> dict[str, float]:
        traversability_mse = self._trav_sse / max(self._trav_count, 1)
        crater_acc = self._crater_correct / max(self._crater_total, 1)
        boulder_acc = self._boulder_correct / max(self._boulder_total, 1)
        slope_acc = self._slope_correct / max(self._slope_total, 1)
        
        # Hazard accuracy represents the average accuracy across all three individual hazards
        hazard_accuracy = (crater_acc + boulder_acc + slope_acc) / 3.0
        return {
            "traversability_mse": traversability_mse,
            "hazard_accuracy": hazard_accuracy,
            "crater_accuracy": crater_acc,
            "boulder_accuracy": boulder_acc,
            "slope_accuracy": slope_acc,
        }


def format_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"traversability_mse={metrics['traversability_mse']:.4f} "
        f"hazard_acc={metrics['hazard_accuracy']:.4f} "
        f"(crater={metrics['crater_accuracy']:.4f}, boulder={metrics['boulder_accuracy']:.4f}, slope={metrics['slope_accuracy']:.4f})"
    )


__all__ = ["MetricsCalculator", "format_metrics"]
