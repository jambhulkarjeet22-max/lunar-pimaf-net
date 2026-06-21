from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import MultiModalEncoder
from .fusion import AttentionFusion
from .heads import MultiTaskHeads


class IceCharacterizationNet(nn.Module):
    """Deep learning architecture for lunar ice characterization."""
    def __init__(self, channels_per_modality: dict[str, int] | None = None):
        super().__init__()
        
        if channels_per_modality is None:
            # Default input channels for each modality instrument
            channels_per_modality = {
                "mini_rf": 3,
                "diviner": 1,
                "lola": 1,
                "lend": 1,
                "lamp": 1,
                "m3": 2,
            }
        
        feature_dim = 32
        num_modalities = len(channels_per_modality)
        
        self.encoder = MultiModalEncoder(channels_per_modality, feature_dim=feature_dim)
        self.fusion = AttentionFusion(feature_dim=feature_dim, num_modalities=num_modalities)
        self.heads = MultiTaskHeads(in_channels=feature_dim * 2)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 1. Encode each modality independently
        encoded_features = self.encoder(inputs)
        
        # 2. Fuse features using attention
        fused_representation = self.fusion(encoded_features)
        
        # 3. Predict ice characteristics through multi-task heads
        predictions = self.heads(fused_representation)
        return predictions
