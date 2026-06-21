"""Attention-based multi-modal fusion for radiation risk prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """Learn modality weights and fuse encoded radiation features."""

    def __init__(self, feature_dim: int = 32, num_modalities: int = 7) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim * num_modalities, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, num_modalities, kernel_size=1),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * num_modalities, feature_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        modality_keys = sorted(features.keys())
        if len(modality_keys) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modality features, got {len(modality_keys)}."
            )

        stacked = torch.cat([features[key] for key in modality_keys], dim=1)
        attn_logits = self.attention(stacked)
        attn_weights = torch.softmax(attn_logits, dim=1)

        weighted = []
        for index, key in enumerate(modality_keys):
            weight = attn_weights[:, index : index + 1]
            weighted.append(features[key] * weight)

        attended = torch.cat(weighted, dim=1)
        return self.fusion_conv(attended)


__all__ = ["AttentionFusion"]
