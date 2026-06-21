"""Feature Pyramid Network for LUNAR-PIMAF-Net multi-modal fusion.

Fuses per-modality encoder pyramids (P1–P5) into a unified top-down feature pyramid
(F1–F5) via lateral 1×1 projections, nearest-neighbor top-down propagation, and
3×3 refinement convolutions.

Reference: docs/ARCHITECTURE_SPECIFICATION.md §5.3
"""

from __future__ import annotations

from typing import Final, Sequence, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modality_encoders import (
    MODALITY_INPUT_CHANNELS,
    PYRAMID_CHANNELS,
    PYRAMID_LEVELS,
    PYRAMID_SPATIAL,
    PyramidFeatures,
)

# ---------------------------------------------------------------------------
# Fused pyramid specification
# ---------------------------------------------------------------------------

FUSED_PYRAMID_LEVELS: Final[tuple[str, ...]] = ("F1", "F2", "F3", "F4", "F5")

DEFAULT_MODALITIES: Final[tuple[str, ...]] = tuple(MODALITY_INPUT_CHANNELS.keys())

DEFAULT_FPN_OUT_CHANNELS: Final[int] = 256


class FusedPyramidFeatures(TypedDict):
    """Unified multi-modal FPN output keyed by fused pyramid level."""

    F1: torch.Tensor
    F2: torch.Tensor
    F3: torch.Tensor
    F4: torch.Tensor
    F5: torch.Tensor


def _num_groups(channels: int) -> int:
    """Select a GroupNorm group count that divides ``channels``."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _validate_modality_pyramid(
    modality_features: dict[str, PyramidFeatures],
    expected_modalities: Sequence[str],
) -> int:
    """Validate encoder pyramids and return the batch size.

    Args:
        modality_features: Mapping from modality name to encoder pyramid outputs.
        expected_modalities: Ordered modality names required at every level.

    Returns:
        Batch size ``B`` shared across all inputs.

    Raises:
        ValueError: If modalities are missing, shapes disagree, or ranks are invalid.
        TypeError: If a pyramid level is not a tensor.
    """
    if not modality_features:
        raise ValueError("modality_features must contain at least one modality pyramid.")

    missing = [name for name in expected_modalities if name not in modality_features]
    if missing:
        raise ValueError(
            "Missing modality pyramids required for FPN fusion: "
            f"{', '.join(missing)}."
        )

    batch_size: int | None = None
    reference_modality = expected_modalities[0]

    for modality in expected_modalities:
        pyramid = modality_features[modality]
        if not isinstance(pyramid, dict):
            raise TypeError(
                f"Modality '{modality}' must provide a PyramidFeatures mapping, "
                f"got {type(pyramid).__name__}."
            )

        for level_idx, level_name in enumerate(PYRAMID_LEVELS):
            if level_name not in pyramid:
                raise ValueError(
                    f"Modality '{modality}' pyramid is missing level '{level_name}'."
                )

            tensor = pyramid[level_name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Modality '{modality}' level '{level_name}' must be a torch.Tensor, "
                    f"got {type(tensor).__name__}."
                )
            if tensor.dim() != 4:
                raise ValueError(
                    f"Modality '{modality}' level '{level_name}' must be 4-D "
                    f"(B, C, H, W), got shape {tuple(tensor.shape)}."
                )

            expected_channels = PYRAMID_CHANNELS[level_idx]
            expected_spatial = PYRAMID_SPATIAL[level_idx]
            if tensor.shape[1] != expected_channels:
                raise ValueError(
                    f"Modality '{modality}' level '{level_name}' expected "
                    f"{expected_channels} channels, got {tensor.shape[1]}."
                )
            if tuple(tensor.shape[2:]) != (expected_spatial, expected_spatial):
                raise ValueError(
                    f"Modality '{modality}' level '{level_name}' expected spatial size "
                    f"({expected_spatial}, {expected_spatial}), got "
                    f"({tensor.shape[2]}, {tensor.shape[3]})."
                )

            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch size for modality '{modality}' level "
                    f"'{level_name}': expected {batch_size}, got {tensor.shape[0]}."
                )

    if batch_size is None:
        raise ValueError(
            f"Could not infer batch size from modality '{reference_modality}' pyramid."
        )

    return batch_size


def _fused_pyramid_from_levels(level_tensors: Sequence[torch.Tensor]) -> FusedPyramidFeatures:
    """Pack ordered level tensors into a ``FusedPyramidFeatures`` mapping."""
    if len(level_tensors) != len(FUSED_PYRAMID_LEVELS):
        raise ValueError(
            f"Expected {len(FUSED_PYRAMID_LEVELS)} fused levels, got {len(level_tensors)}."
        )
    return FusedPyramidFeatures(
        F1=level_tensors[0],
        F2=level_tensors[1],
        F3=level_tensors[2],
        F4=level_tensors[3],
        F5=level_tensors[4],
    )


def fused_to_encoder_pyramid(fused: FusedPyramidFeatures) -> PyramidFeatures:
    """Convert fused ``F*`` keys to encoder-style ``P*`` keys for downstream modules."""
    return PyramidFeatures(
        P1=fused["F1"],
        P2=fused["F2"],
        P3=fused["F3"],
        P4=fused["F4"],
        P5=fused["F5"],
    )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvGNAct(nn.Module):
    """Convolution → GroupNorm → SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
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


class LateralConnection(nn.Module):
    """1×1 lateral projection from concatenated modality features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvGNAct(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeatureFusionBlock(nn.Module):
    """3×3 refinement convolution applied after top-down feature merging."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = ConvGNAct(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(x)


class FeaturePyramidNetwork(nn.Module):
    """Top-down Feature Pyramid Network over five lateral feature maps.

    Implements the standard FPN pathway (Lin et al., 2017):
        1. Start from the coarsest level.
        2. Upsample and add to the next finer lateral.
        3. Apply a 3×3 refinement convolution at each level.

    Args:
        out_channels: Channel width of every pyramid level (all inputs must match).
    """

    def __init__(self, out_channels: int = DEFAULT_FPN_OUT_CHANNELS) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.fusion_blocks = nn.ModuleList(
            FeatureFusionBlock(out_channels) for _ in range(len(PYRAMID_LEVELS))
        )

    def forward(self, laterals: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Run top-down fusion over lateral features.

        Args:
            laterals: Sequence of five tensors ordered P1→P5, each of shape
                ``(B, out_channels, H_l, W_l)``.

        Returns:
            List of refined feature maps ordered F1→F5.
        """
        if len(laterals) != len(PYRAMID_LEVELS):
            raise ValueError(
                f"FeaturePyramidNetwork expects {len(PYRAMID_LEVELS)} lateral maps, "
                f"got {len(laterals)}."
            )

        merged: list[torch.Tensor | None] = [None] * len(PYRAMID_LEVELS)
        merged[-1] = laterals[-1]

        for level_idx in range(len(PYRAMID_LEVELS) - 2, -1, -1):
            finer = laterals[level_idx]
            coarser = merged[level_idx + 1]
            assert coarser is not None

            upsampled = F.interpolate(
                coarser,
                size=finer.shape[-2:],
                mode="nearest",
            )
            merged[level_idx] = finer + upsampled

        outputs: list[torch.Tensor] = []
        for level_idx, feature in enumerate(merged):
            assert feature is not None
            outputs.append(self.fusion_blocks[level_idx](feature))

        return outputs


class MultiModalFeaturePyramid(nn.Module):
    """Fuse all modality encoder pyramids into a unified FPN.

    For each pyramid level ``P*``:
        1. Concatenate features from every modality along the channel axis.
        2. Project to ``out_channels`` with a lateral 1×1 convolution.
        3. Propagate coarse features downward with nearest-neighbor upsampling.
        4. Refine each level with a 3×3 convolution.

    Args:
        out_channels: Unified channel width for all fused levels ``F1``–``F5``.
        modalities: Ordered modality names used for concatenation.
        encoder_channels: Per-level encoder channel schedule (defaults to
            ``PYRAMID_CHANNELS``).
    """

    def __init__(
        self,
        out_channels: int = DEFAULT_FPN_OUT_CHANNELS,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        encoder_channels: Sequence[int] = PYRAMID_CHANNELS,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.modalities: tuple[str, ...] = tuple(modalities)
        self.encoder_channels: tuple[int, ...] = tuple(encoder_channels)

        if len(self.encoder_channels) != len(PYRAMID_LEVELS):
            raise ValueError(
                f"encoder_channels must have {len(PYRAMID_LEVELS)} entries, "
                f"got {len(self.encoder_channels)}."
            )

        num_modalities = len(self.modalities)
        self.lateral_connections = nn.ModuleList(
            LateralConnection(num_modalities * in_channels, out_channels)
            for in_channels in self.encoder_channels
        )
        self.fpn = FeaturePyramidNetwork(out_channels=out_channels)

    @property
    def num_modalities(self) -> int:
        """Number of modalities fused at each pyramid level."""
        return len(self.modalities)

    def _fuse_modalities_at_level(
        self,
        modality_features: dict[str, PyramidFeatures],
        level_name: str,
    ) -> torch.Tensor:
        """Concatenate all modality tensors at a single pyramid level."""
        return torch.cat(
            [modality_features[modality][level_name] for modality in self.modalities],
            dim=1,
        )

    def forward(
        self,
        modality_features: dict[str, PyramidFeatures],
    ) -> FusedPyramidFeatures:
        """Fuse multi-modal encoder pyramids into unified FPN features.

        Args:
            modality_features: Mapping ``modality_name -> PyramidFeatures`` produced
                by the modality encoders.  Every modality in ``self.modalities`` must
                be present with levels ``P1``–``P5``.

        Returns:
            Fused pyramid with levels ``F1``–``F5``, each of shape
            ``(B, out_channels, H_l, W_l)``.
        """
        _validate_modality_pyramid(modality_features, self.modalities)

        laterals: list[torch.Tensor] = []
        for level_idx, level_name in enumerate(PYRAMID_LEVELS):
            concatenated = self._fuse_modalities_at_level(modality_features, level_name)
            laterals.append(self.lateral_connections[level_idx](concatenated))

        fused_levels = self.fpn(laterals)
        return _fused_pyramid_from_levels(fused_levels)


__all__ = [
    "DEFAULT_FPN_OUT_CHANNELS",
    "DEFAULT_MODALITIES",
    "FUSED_PYRAMID_LEVELS",
    "ConvGNAct",
    "FeatureFusionBlock",
    "FeaturePyramidNetwork",
    "FusedPyramidFeatures",
    "LateralConnection",
    "MultiModalFeaturePyramid",
    "fused_to_encoder_pyramid",
]
