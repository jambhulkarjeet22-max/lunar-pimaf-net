"""Attention-gated U-Net decoder for LUNAR-PIMAF-Net.

Upsamples fused pyramid features (F5→F1) with skip connections and spatial
attention gating to produce a high-resolution decoder embedding for prediction
heads.

Reference: docs/ARCHITECTURE_SPECIFICATION.md §8
"""

from __future__ import annotations

from typing import Final, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_pyramid import (
    DEFAULT_FPN_OUT_CHANNELS,
    FUSED_PYRAMID_LEVELS,
    FusedPyramidFeatures,
    _num_groups,
)
from src.models.modality_encoders import PYRAMID_SPATIAL

DECODER_OUTPUT_CHANNELS: Final[int] = 128
DECODER_STAGE_CHANNELS: Final[tuple[int, ...]] = (256, 128, 128, 128, 128)


def _validate_fused_pyramid(features: FusedPyramidFeatures) -> int:
    """Validate fused pyramid tensors and return batch size."""
    batch_size: int | None = None
    for level_idx, level_name in enumerate(FUSED_PYRAMID_LEVELS):
        if level_name not in features:
            raise ValueError(f"Fused pyramid missing level '{level_name}'.")
        tensor = features[level_name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Level '{level_name}' must be a torch.Tensor, got {type(tensor).__name__}."
            )
        if tensor.dim() != 4:
            raise ValueError(
                f"Level '{level_name}' must be 4-D (B, C, H, W), got {tuple(tensor.shape)}."
            )
        expected_spatial = PYRAMID_SPATIAL[level_idx]
        if tensor.shape[1] != DEFAULT_FPN_OUT_CHANNELS:
            raise ValueError(
                f"Level '{level_name}' expected {DEFAULT_FPN_OUT_CHANNELS} channels, "
                f"got {tensor.shape[1]}."
            )
        if tuple(tensor.shape[2:]) != (expected_spatial, expected_spatial):
            raise ValueError(
                f"Level '{level_name}' expected spatial size "
                f"({expected_spatial}, {expected_spatial}), got "
                f"({tensor.shape[2]}, {tensor.shape[3]})."
            )
        if batch_size is None:
            batch_size = tensor.shape[0]
        elif tensor.shape[0] != batch_size:
            raise ValueError(
                f"Inconsistent batch size at '{level_name}': expected {batch_size}, "
                f"got {tensor.shape[0]}."
            )
    if batch_size is None:
        raise ValueError("Could not infer batch size from fused pyramid.")
    return batch_size


class ConvGNAct(nn.Module):
    """Convolution → GroupNorm → SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            ),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Spatial attention gate for skip connections (Oktay et al., 2018).

    Computes ``α = σ(ψ(ReLU(W_g·g + W_x·x)))`` and returns ``α ⊙ x``.
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: int | None = None,
    ) -> None:
        super().__init__()
        inter = inter_channels or max(gate_channels // 2, 1)
        self.W_g = nn.Conv2d(gate_channels, inter, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(skip_channels, inter, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter, 1, kernel_size=1, bias=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(
                gate,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        attention = torch.sigmoid(self.psi(F.relu(self.W_g(gate) + self.W_x(skip))))
        return skip * attention


class DecoderBlock(nn.Module):
    """Upsample, attend skip features, fuse, and refine."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.attention = AttentionGate(
            gate_channels=in_channels,
            skip_channels=skip_channels,
        )
        fused_channels = in_channels + skip_channels
        self.fuse = nn.Sequential(
            ConvGNAct(fused_channels, out_channels, kernel_size=3, padding=1),
            ConvGNAct(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gated_skip = self.attention(x, skip)
        fused = torch.cat([x, gated_skip], dim=1)
        return self.fuse(fused)


class AttentionUNetDecoder(nn.Module):
    """Attention-gated U-Net decoder over fused pyramid levels F5→F1.

    Args:
        fpn_channels: Channel width of fused pyramid inputs (default 256).
        stage_channels: Output channels after each decoder stage D5→D1.
    """

    def __init__(
        self,
        fpn_channels: int = DEFAULT_FPN_OUT_CHANNELS,
        stage_channels: Sequence[int] = DECODER_STAGE_CHANNELS,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(FUSED_PYRAMID_LEVELS):
            raise ValueError(
                f"stage_channels must have {len(FUSED_PYRAMID_LEVELS)} entries."
            )

        self.fpn_channels = fpn_channels
        self.stage_channels = tuple(stage_channels)

        self.entry = ConvGNAct(fpn_channels, stage_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        for idx in range(len(FUSED_PYRAMID_LEVELS) - 1):
            in_ch = stage_channels[idx]
            out_ch = stage_channels[idx + 1]
            self.blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=fpn_channels,
                    out_channels=out_ch,
                )
            )

    def forward(self, pyramid: FusedPyramidFeatures) -> torch.Tensor:
        """Decode fused pyramid features to a full-resolution embedding.

        Args:
            pyramid: Fused features ``F1``–``F5``, each
                ``(B, fpn_channels, H_l, W_l)``.

        Returns:
            Decoder embedding of shape ``(B, 128, 128, 128)``.
        """
        _validate_fused_pyramid(pyramid)

        skips = [pyramid[level] for level in reversed(FUSED_PYRAMID_LEVELS)]
        x = self.entry(skips[0])  # F5

        for block, skip in zip(self.blocks, skips[1:]):
            x = block(x, skip)

        if x.shape[1] != DECODER_OUTPUT_CHANNELS:
            raise RuntimeError(
                f"Decoder output channels {x.shape[1]} != "
                f"expected {DECODER_OUTPUT_CHANNELS}."
            )
        if tuple(x.shape[2:]) != (128, 128):
            raise RuntimeError(
                f"Decoder output spatial size {tuple(x.shape[2:])} != (128, 128)."
            )
        return x


__all__ = [
    "DECODER_OUTPUT_CHANNELS",
    "DECODER_STAGE_CHANNELS",
    "AttentionGate",
    "AttentionUNetDecoder",
    "ConvGNAct",
    "DecoderBlock",
]
