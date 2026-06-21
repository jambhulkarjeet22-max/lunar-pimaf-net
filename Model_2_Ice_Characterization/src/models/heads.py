from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHeads(nn.Module):
    """Multi-task prediction heads for ice characterization."""
    def __init__(self, in_channels: int = 64):
        super().__init__()
        
        # 1. Ice Purity (%) - Regression (0-100 mapped to 0-1 internally)
        self.purity_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 2. Ice Depth (meters) - Regression (positive values)
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus() # Ensures positive depth
        )
        
        # 3. Ice Type Classification (Surface, Subsurface, Mixed) - Classification (3 classes)
        self.type_classification_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        
        # 4. Ice Stability Score (0-1) - Regression
        self.stability_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 5. Confidence Score - Regression (0-1, Uncertainty Estimation)
        self.confidence_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "purity_percentage": self.purity_head(x),
            "ice_depth": self.depth_head(x),
            "ice_type": self.type_classification_head(x),
            "stability_score": self.stability_head(x),
            "confidence": self.confidence_head(x),
        }
