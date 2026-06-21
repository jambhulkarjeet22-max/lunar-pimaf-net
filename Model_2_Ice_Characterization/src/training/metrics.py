from __future__ import annotations

import torch
import torch.nn.functional as F

class MetricsCalculator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total_purity_mse = 0.0
        self.total_depth_mse = 0.0
        self.total_type_accuracy = 0.0
        self.batches = 0
        
    def update(self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        with torch.no_grad():
            self.total_purity_mse += F.mse_loss(predictions["purity_percentage"], targets["purity_percentage"]).item()
            self.total_depth_mse += F.mse_loss(predictions["ice_depth"], targets["ice_depth"]).item()
            
            preds = torch.argmax(predictions["ice_type"], dim=1)
            self.total_type_accuracy += (preds == targets["ice_type"]).float().mean().item()
            self.batches += 1
            
    def compute(self) -> dict[str, float]:
        if self.batches == 0:
            return {}
        return {
            "purity_mse": self.total_purity_mse / self.batches,
            "depth_mse": self.total_depth_mse / self.batches,
            "type_accuracy": self.total_type_accuracy / self.batches,
        }
