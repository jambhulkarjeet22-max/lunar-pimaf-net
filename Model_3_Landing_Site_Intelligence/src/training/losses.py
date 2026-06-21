"""Physics-aware and multi-task losses for landing site intelligence."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

SCORE_KEYS: Final[tuple[str, ...]] = (
    "landing_safety_score",
    "hazard_probability",
    "illumination_score",
    "resource_accessibility_score",
    "final_suitability_score",
)


class LandingSiteLoss(nn.Module):
    """Multi-task loss with slope-aware safety penalty."""

    def __init__(
        self,
        slope_penalty_weight: float = 0.25,
        slope_threshold: float = 0.15,
    ) -> None:
        super().__init__()
        self.slope_penalty_weight = slope_penalty_weight
        self.slope_threshold = slope_threshold
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def _physics_slope_penalty(
        self,
        safety_pred: torch.Tensor,
        slope_magnitude: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize high predicted safety on steep terrain."""
        steep_mask = (slope_magnitude > self.slope_threshold).float()
        unsafe_confidence = safety_pred * steep_mask
        return unsafe_confidence.mean()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        safety_loss = self.mse(predictions["landing_safety_score"], targets["landing_safety_score"])
        hazard_loss = self.bce(predictions["hazard_probability"], targets["hazard_probability"])
        illumination_loss = self.mse(predictions["illumination_score"], targets["illumination_score"])
        resource_loss = self.mse(
            predictions["resource_accessibility_score"],
            targets["resource_accessibility_score"],
        )
        suitability_loss = self.mse(
            predictions["final_suitability_score"],
            targets["final_suitability_score"],
        )

        slope_penalty = torch.tensor(0.0, device=safety_loss.device)
        if "slope_magnitude" in targets:
            slope_penalty = self._physics_slope_penalty(
                predictions["landing_safety_score"],
                targets["slope_magnitude"],
            )

        regression_total = safety_loss + illumination_loss + resource_loss + suitability_loss
        total_loss = (
            regression_total
            + hazard_loss
            + self.slope_penalty_weight * slope_penalty
        )

        return {
            "total_loss": total_loss,
            "safety_loss": safety_loss,
            "hazard_loss": hazard_loss,
            "illumination_loss": illumination_loss,
            "resource_loss": resource_loss,
            "suitability_loss": suitability_loss,
            "slope_penalty": slope_penalty,
        }


__all__ = ["LandingSiteLoss", "SCORE_KEYS"]
