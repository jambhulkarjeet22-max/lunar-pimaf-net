"""Evaluation metrics for radiation risk prediction."""

from __future__ import annotations

from typing import Any

import torch


class MetricsCalculator:
    """Accumulates radiation dose rate regression and hazard classification metrics."""

    def __init__(self, hazard_threshold: float = 0.5) -> None:
        self.hazard_threshold = hazard_threshold
        self.reset()

    def reset(self) -> None:
        self._radiation_sse = 0.0
        self._radiation_count = 0
        self._habitat_safety_sse = 0.0
        self._habitat_safety_count = 0
        self._hazard_correct = 0
        self._hazard_total = 0

    @torch.no_grad()
    def update(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> None:
        rad_pred = predictions["radiation_dose_rate"]
        rad_tgt = targets["radiation_dose_rate"]
        self._radiation_sse += torch.sum((rad_pred - rad_tgt) ** 2).item()
        self._radiation_count += rad_pred.numel()

        safety_pred = predictions["habitat_safety_score"]
        safety_tgt = targets["habitat_safety_score"]
        self._habitat_safety_sse += torch.sum((safety_pred - safety_tgt) ** 2).item()
        self._habitat_safety_count += safety_pred.numel()

        # Threshold hazard maps to calculate binary accuracy
        hazard_pred = (predictions["final_radiation_hazard_map"] >= self.hazard_threshold).float()
        hazard_tgt = (targets["final_radiation_hazard_map"] >= self.hazard_threshold).float()
        self._hazard_correct += (hazard_pred == hazard_tgt).sum().item()
        self._hazard_total += hazard_pred.numel()

    def compute(self) -> dict[str, float]:
        radiation_mse = self._radiation_sse / max(self._radiation_count, 1)
        habitat_safety_mse = self._habitat_safety_sse / max(self._habitat_safety_count, 1)
        hazard_accuracy = self._hazard_correct / max(self._hazard_total, 1)
        return {
            "radiation_mse": radiation_mse,
            "habitat_safety_mse": habitat_safety_mse,
            "hazard_accuracy": hazard_accuracy,
        }


def format_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"radiation_mse={metrics['radiation_mse']:.4f} "
        f"habitat_safety_mse={metrics['habitat_safety_mse']:.4f} "
        f"hazard_acc={metrics['hazard_accuracy']:.4f}"
    )


__all__ = ["MetricsCalculator", "format_metrics"]
