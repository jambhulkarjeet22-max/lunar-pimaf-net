"""Physics-aware and multi-task loss functions for radiation risk prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadiationLoss(nn.Module):
    """Multi-task loss function with physics-aware regularization constraints."""

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
        loss_dose = self.mse(predictions["radiation_dose_rate"], targets["radiation_dose_rate"])
        loss_risk = self.mse(predictions["radiation_risk_score"], targets["radiation_risk_score"])
        loss_shielding = self.mse(
            predictions["shielding_effectiveness_score"], targets["shielding_effectiveness_score"]
        )
        loss_safety = self.mse(predictions["habitat_safety_score"], targets["habitat_safety_score"])
        loss_hazard = self.bce(predictions["final_radiation_hazard_map"], targets["final_radiation_hazard_map"])

        # Inputs required for physics constraints are tucked inside the targets dictionary under "inputs"
        # If running in a setup where "inputs" is not in targets, we fall back to zero physics loss.
        physics_loss = torch.tensor(0.0, device=loss_dose.device)
        physics_shielding = torch.tensor(0.0, device=loss_dose.device)
        physics_exposure = torch.tensor(0.0, device=loss_dose.device)
        physics_shield_eff = torch.tensor(0.0, device=loss_dose.device)
        physics_habitat_safety = torch.tensor(0.0, device=loss_dose.device)

        if "inputs" in targets:
            inputs = targets["inputs"]
            regolith = inputs["regolith"]
            psr = inputs["psr"]
            elevation = inputs["elevation"]

            # 1. Regolith & PSR shielding constraint:
            # High regolith thickness and PSR should bound the radiation risk from above.
            bound_upper = (1.0 - 0.5 * regolith - 0.3 * psr).clamp(0.0, 1.0)
            physics_shielding = torch.mean(torch.relu(predictions["radiation_risk_score"] - bound_upper))

            # 2. Elevation exposure constraint:
            # Exposed high elevation regions should have a minimum radiation risk.
            bound_lower = (0.6 * elevation).clamp(0.0, 1.0)
            physics_exposure = torch.mean(torch.relu(bound_lower - predictions["radiation_risk_score"]))

            # 3. Shielding effectiveness coupling with regolith thickness:
            # Shielding effectiveness should closely follow regolith thickness.
            physics_shield_eff = self.mse(predictions["shielding_effectiveness_score"], regolith)

            # 4. Habitat safety coupling:
            # Safety score should align with shielding_effectiveness * (1.0 - radiation_risk_score)
            expected_safety = predictions["shielding_effectiveness_score"] * (
                1.0 - predictions["radiation_risk_score"]
            )
            physics_habitat_safety = self.mse(predictions["habitat_safety_score"], expected_safety)

            physics_loss = (
                physics_shielding + physics_exposure + physics_shield_eff + physics_habitat_safety
            )

        total_loss = (
            loss_dose + loss_risk + loss_shielding + loss_safety + loss_hazard
            + self.physics_weight * physics_loss
        )

        return {
            "total_loss": total_loss,
            "loss_dose": loss_dose,
            "loss_risk": loss_risk,
            "loss_shielding": loss_shielding,
            "loss_safety": loss_safety,
            "loss_hazard": loss_hazard,
            "physics_loss": physics_loss,
            "physics_shielding": physics_shielding,
            "physics_exposure": physics_exposure,
            "physics_shield_eff": physics_shield_eff,
            "physics_habitat_safety": physics_habitat_safety,
        }


__all__ = ["RadiationLoss"]
