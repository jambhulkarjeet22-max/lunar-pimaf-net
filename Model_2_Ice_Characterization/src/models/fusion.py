from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """Attention-based feature fusion network for multi-modal lunar data."""
    def __init__(self, feature_dim: int = 32, num_modalities: int = 6):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim * num_modalities, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, num_modalities, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * num_modalities, feature_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        # Stack all modalities (B, C, H, W) -> (B, num_modalities * C, H, W)
        modality_keys = sorted(list(features.keys()))
        stacked = torch.cat([features[k] for k in modality_keys], dim=1)
        
        # Calculate attention weights
        attn_weights = self.attention(stacked)  # (B, num_modalities, H, W)
        
        # Apply attention to each modality
        weighted_features = []
        for i, k in enumerate(modality_keys):
            weight = attn_weights[:, i:i+1, :, :]
            weighted_features.append(features[k] * weight)
            
        attended_stacked = torch.cat(weighted_features, dim=1)
        
        # Fuse into a single dense representation
        fused = self.fusion_conv(attended_stacked)
        return fused
