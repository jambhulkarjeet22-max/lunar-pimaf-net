from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IceCharacterizationLoss(nn.Module):
    """Multi-task loss with physics constraints for ice characterization."""
    def __init__(self, weights: dict[str, float] | None = None):
        super().__init__()
        self.weights = weights or {
            "purity": 1.0,
            "depth": 1.0,
            "type": 1.0,
            "stability": 1.0,
            "physics": 0.5,
        }
        
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Ensure all predictions match target spatial dimensions
        target_size = targets["purity_percentage"].shape[-2:]
        upsampled_preds = {}
        for k, v in predictions.items():
            if v.shape[-2:] != target_size:
                upsampled_preds[k] = F.interpolate(v, size=target_size, mode="bilinear", align_corners=False)
            else:
                upsampled_preds[k] = v
        
        # 1. Purity Loss
        loss_purity = self.mse(upsampled_preds["purity_percentage"], targets["purity_percentage"])
        
        # 2. Depth Loss
        loss_depth = self.mse(upsampled_preds["ice_depth"], targets["ice_depth"])
        
        # 3. Type Classification Loss
        # Type is (B, H, W) of long
        loss_type = self.ce(upsampled_preds["ice_type"], targets["ice_type"])
        
        # 4. Stability Loss
        loss_stability = self.mse(upsampled_preds["stability_score"], targets["stability_score"])
        
        # 5. Physics-aware constraint:
        # If type is Surface Ice (0), depth should be close to 0.
        # upsampled_preds["ice_type"]: (B, 3, H, W)
        probs = F.softmax(upsampled_preds["ice_type"], dim=1)
        prob_surface = probs[:, 0:1, :, :] # (B, 1, H, W)
        
        # Penalize large depth predictions when the model is confident it's surface ice
        physics_penalty = torch.mean(prob_surface * upsampled_preds["ice_depth"])
        
        total_loss = (
            self.weights["purity"] * loss_purity +
            self.weights["depth"] * loss_depth +
            self.weights["type"] * loss_type +
            self.weights["stability"] * loss_stability +
            self.weights["physics"] * physics_penalty
        )
        
        return {
            "total_loss": total_loss,
            "loss_purity": loss_purity,
            "loss_depth": loss_depth,
            "loss_type": loss_type,
            "loss_stability": loss_stability,
            "loss_physics": physics_penalty
        }
