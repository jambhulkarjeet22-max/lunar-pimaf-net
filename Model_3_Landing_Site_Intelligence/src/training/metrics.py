"""Evaluation metrics for landing site intelligence."""

from __future__ import annotations

from typing import Any

import torch


class MetricsCalculator:
    """Accumulates safety regression and hazard classification metrics."""

    def __init__(self, hazard_threshold: float = 0.5) -> None:
        self.hazard_threshold = hazard_threshold
        self.reset()

    def reset(self) -> None:
        self._safety_sse = 0.0
        self._safety_count = 0
        self._hazard_correct = 0
        self._hazard_total = 0
        self._suitability_sse = 0.0
        self._suitability_count = 0

    @torch.no_grad()
    def update(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> None:
        safety_pred = predictions["landing_safety_score"]
        safety_tgt = targets["landing_safety_score"]
        self._safety_sse += torch.sum((safety_pred - safety_tgt) ** 2).item()
        self._safety_count += safety_pred.numel()

        hazard_pred = (predictions["hazard_probability"] >= self.hazard_threshold).float()
        hazard_tgt = (targets["hazard_probability"] >= self.hazard_threshold).float()
        self._hazard_correct += (hazard_pred == hazard_tgt).sum().item()
        self._hazard_total += hazard_pred.numel()

        suit_pred = predictions["final_suitability_score"]
        suit_tgt = targets["final_suitability_score"]
        self._suitability_sse += torch.sum((suit_pred - suit_tgt) ** 2).item()
        self._suitability_count += suit_pred.numel()

    def compute(self) -> dict[str, float]:
        safety_mse = self._safety_sse / max(self._safety_count, 1)
        hazard_accuracy = self._hazard_correct / max(self._hazard_total, 1)
        suitability_mse = self._suitability_sse / max(self._suitability_count, 1)
        return {
            "safety_mse": safety_mse,
            "hazard_accuracy": hazard_accuracy,
            "suitability_mse": suitability_mse,
        }


def format_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"safety_mse={metrics['safety_mse']:.4f} "
        f"hazard_acc={metrics['hazard_accuracy']:.4f} "
        f"suitability_mse={metrics['suitability_mse']:.4f}"
    )


__all__ = ["MetricsCalculator", "format_metrics"]
