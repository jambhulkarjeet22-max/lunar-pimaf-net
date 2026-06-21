"""Physics-aware and multi-task loss functions for rover hazard and navigation prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoverNavigationLoss(nn.Module):
    """Multi-task loss function with physics-aware constraints for safe rover traversability."""

    def __init__(self, physics_weight: float = 0.1) -> None:
        super().__init__()
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Supervised multi-task losses
        loss_trav = self.mse(predictions["traversability_score"], targets["traversability_score"])
        loss_crater = self.bce(
            predictions["crater_hazard_probability"], targets["crater_hazard_probability"]
        )
        loss_boulder = self.bce(
            predictions["boulder_hazard_probability"], targets["boulder_hazard_probability"]
        )
        loss_slope = self.mse(predictions["slope_hazard_score"], targets["slope_hazard_score"])
        loss_cost = self.mse(predictions["navigation_cost_map"], targets["navigation_cost_map"])
        loss_safety = self.mse(predictions["rover_safety_score"], targets["rover_safety_score"])

        # Physics-aware loss calculations
        physics_loss = torch.tensor(0.0, device=loss_trav.device)
        physics_trav_limit = torch.tensor(0.0, device=loss_trav.device)
        physics_slope_align = torch.tensor(0.0, device=loss_trav.device)
        physics_safety_coupling = torch.tensor(0.0, device=loss_trav.device)
        physics_cost_align = torch.tensor(0.0, device=loss_trav.device)
        physics_illum_trav = torch.tensor(0.0, device=loss_trav.device)

        if "inputs" in targets:
            inputs = targets["inputs"]
            slope = inputs["slope"]
            boulder = inputs["boulder"]
            illumination = inputs["illumination"]
            crater = inputs["crater"]

            # 1. Traversability safety upper limit:
            # Penalize traversability that exceeds 1.0 - slope - boulder.
            bound_trav = (1.0 - slope - boulder).clamp(0.0, 1.0)
            physics_trav_limit = torch.mean(torch.relu(predictions["traversability_score"] - bound_trav))

            # 2. Slope hazard direct alignment:
            # Predicted slope hazard score should align closely with the slope input.
            physics_slope_align = self.mse(predictions["slope_hazard_score"], slope)

            # 3. Multiplicative safety score consistency:
            # Safety score should align with (1.0 - crater_prob) * (1.0 - boulder_prob) * (1.0 - slope_hazard)
            expected_safety = (
                (1.0 - predictions["crater_hazard_probability"])
                * (1.0 - predictions["boulder_hazard_probability"])
                * (1.0 - predictions["slope_hazard_score"])
            )
            physics_safety_coupling = self.mse(predictions["rover_safety_score"], expected_safety)

            # 4. Navigation cost map alignment:
            # Navigation cost should reflect 1.0 - safety.
            physics_cost_align = self.mse(predictions["navigation_cost_map"], 1.0 - predictions["rover_safety_score"])

            # 5. Illumination-traversability lower limit:
            # Traversability should be at least illumination * (1.0 - average_hazard)
            avg_hazard = (crater + boulder + slope) / 3.0
            expected_trav_lower = (illumination * (1.0 - avg_hazard)).clamp(0.0, 1.0)
            physics_illum_trav = torch.mean(torch.relu(expected_trav_lower - predictions["traversability_score"]))

            physics_loss = (
                physics_trav_limit
                + physics_slope_align
                + physics_safety_coupling
                + physics_cost_align
                + physics_illum_trav
            )

        total_loss = (
            loss_trav
            + loss_crater
            + loss_boulder
            + loss_slope
            + loss_cost
            + loss_safety
            + self.physics_weight * physics_loss
        )

        return {
            "total_loss": total_loss,
            "loss_trav": loss_trav,
            "loss_crater": loss_crater,
            "loss_boulder": loss_boulder,
            "loss_slope": loss_slope,
            "loss_cost": loss_cost,
            "loss_safety": loss_safety,
            "physics_loss": physics_loss,
            "physics_trav_limit": physics_trav_limit,
            "physics_slope_align": physics_slope_align,
            "physics_safety_coupling": physics_safety_coupling,
            "physics_cost_align": physics_cost_align,
            "physics_illum_trav": physics_illum_trav,
        }


__all__ = ["RoverNavigationLoss"]
