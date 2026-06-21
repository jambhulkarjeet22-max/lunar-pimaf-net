"""LUNAR-PIMAF-Net orchestration layer.

Wires modality encoders, multi-modal FPN, cross-modal attention, physics
constraints, attention U-Net decoder, and multi-task prediction heads into a
single end-to-end model for lunar subsurface ice detection.

Reference: docs/ARCHITECTURE_SPECIFICATION.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.confidence_head import ConfidenceHead
from src.models.cross_modal_attention import (
    CMMA_LEVELS,
    CMMA_SPATIAL,
    CrossModalMultiScaleAttention,
    MultiScaleModalityFeatures,
)
from src.models.decoder import AttentionUNetDecoder, DECODER_OUTPUT_CHANNELS
from src.models.feature_pyramid import (
    DEFAULT_FPN_OUT_CHANNELS,
    DEFAULT_MODALITIES,
    FUSED_PYRAMID_LEVELS,
    FusedPyramidFeatures,
    LateralConnection,
    MultiModalFeaturePyramid,
)
from src.models.modality_encoders import (
    EXPECTED_INPUT_SPATIAL,
    MODALITY_ENCODER_REGISTRY,
    MODALITY_INPUT_CHANNELS,
    PYRAMID_CHANNELS,
    PYRAMID_LEVELS,
    BaseModalityEncoder,
    PyramidFeatures,
)
from src.models.physics_constraint_module import (
    PhysicsConstraintModule,
    PhysicsInputs,
    PhysicsResiduals,
)
from src.models.prediction_heads import MultiTaskPredictionHead, PredictionOutputs
from src.models.uncertainty_head import EvidentialUncertaintyModule, UncertaintyOutputs

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

EXPECTED_INPUT_CHANNELS: Final[int] = sum(MODALITY_INPUT_CHANNELS.values())
NUM_MODALITIES: Final[int] = len(MODALITY_INPUT_CHANNELS)

# Raw observation channel indices (DATA_PREPROCESSING_PIPELINE.md §4.2).
CHANNEL_LOLA_SLOPE_DEG: Final[int] = 1
CHANNEL_DIV_TBOL_MAX: Final[int] = 18
CHANNEL_DIV_ICE_STABILITY: Final[int] = 30
CHANNEL_DIV_INSOLATION: Final[int] = 32
CHANNEL_LAMP_ALBEDO_RATIO: Final[int] = 35
CHANNEL_LEND_H_WT_PCT: Final[int] = 42
CHANNEL_PHYSICS_PRIORS: Final[slice] = slice(52, 59)

# Thermal group-local index for ``div_ice_stability`` (channel 30).
THERMAL_ICE_STABILITY_OFFSET: Final[int] = CHANNEL_DIV_ICE_STABILITY - 18

# Six sensor validity masks (topography is the reference grid and assumed valid).
SENSOR_MODALITIES: Final[tuple[str, ...]] = (
    "radar",
    "thermal",
    "uv",
    "neutron",
    "spectral",
    "physics",
)

CMMA_PYRAMID_LEVELS: Final[tuple[str, ...]] = ("P3", "P4", "P5")

ModalityTensorDict = dict[str, torch.Tensor]
ModalityPyramidDict = dict[str, PyramidFeatures]
ModalityMaskDict = dict[str, torch.Tensor]


def _build_modality_channel_slices() -> dict[str, slice]:
    """Map modality names to channel slices over the 59-channel input tensor."""
    slices: dict[str, slice] = {}
    start = 0
    for name, count in MODALITY_INPUT_CHANNELS.items():
        slices[name] = slice(start, start + count)
        start += count
    return slices


MODALITY_CHANNEL_SLICES: Final[dict[str, slice]] = _build_modality_channel_slices()


class LunarPIMAFOutput(TypedDict):
    """Bundled inference outputs from ``LunarPIMAFNet``."""

    segmentation_logits: torch.Tensor
    surface_ice_probability: torch.Tensor
    subsurface_ice_probability: torch.Tensor
    dirichlet_alpha: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    total_uncertainty: torch.Tensor
    confidence: torch.Tensor
    physics_residuals: PhysicsResiduals
    decoder_features: torch.Tensor
    fused_pyramid: FusedPyramidFeatures


@dataclass(frozen=True)
class LunarPIMAFFeatures:
    """Intermediate representations produced by ``forward_features``."""

    fused_pyramid: FusedPyramidFeatures
    decoder_features: torch.Tensor
    modality_pyramids: ModalityPyramidDict


@dataclass(frozen=True)
class LunarPIMAFPredictions:
    """Prediction-head outputs without physics residuals."""

    segmentation_logits: torch.Tensor
    surface_ice_probability: torch.Tensor
    subsurface_ice_probability: torch.Tensor
    dirichlet_alpha: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    total_uncertainty: torch.Tensor
    confidence: torch.Tensor


def _validate_input_tensor(x: torch.Tensor) -> int:
    """Validate the fused observation tensor and return batch size."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Input must be a torch.Tensor, got {type(x).__name__}."
        )
    if x.dim() != 4:
        raise ValueError(
            f"Input must be 4-D (B, C, H, W), got shape {tuple(x.shape)}."
        )
    if x.shape[1] != EXPECTED_INPUT_CHANNELS:
        raise ValueError(
            f"Input expected {EXPECTED_INPUT_CHANNELS} channels, got {x.shape[1]}."
        )
    if tuple(x.shape[2:]) != EXPECTED_INPUT_SPATIAL:
        raise ValueError(
            f"Input expected spatial size {EXPECTED_INPUT_SPATIAL}, "
            f"got ({x.shape[2]}, {x.shape[3]})."
        )
    return x.shape[0]


def _normalize_modality_masks(
    modality_masks: torch.Tensor | ModalityMaskDict | None,
    batch_size: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ModalityMaskDict:
    """Convert optional masks to per-modality ``(B, 1, 128, 128)`` tensors."""
    masks: ModalityMaskDict = {}
    resolved_device = device or torch.device("cpu")
    resolved_dtype = dtype or torch.float32

    if modality_masks is None:
        ones = torch.ones(
            batch_size,
            1,
            *EXPECTED_INPUT_SPATIAL,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        return {name: ones for name in MODALITY_INPUT_CHANNELS}

    if isinstance(modality_masks, dict):
        for name in MODALITY_INPUT_CHANNELS:
            if name not in modality_masks:
                raise ValueError(
                    f"modality_masks dict missing key '{name}'. "
                    f"Expected keys: {', '.join(MODALITY_INPUT_CHANNELS)}."
                )
            mask = modality_masks[name]
            if mask.dim() != 4 or mask.shape[1] != 1:
                raise ValueError(
                    f"Mask for '{name}' must have shape (B, 1, 128, 128), "
                    f"got {tuple(mask.shape)}."
                )
            if mask.shape[0] != batch_size:
                raise ValueError(
                    f"Mask for '{name}' batch size {mask.shape[0]} does not match "
                    f"input batch size {batch_size}."
                )
            if tuple(mask.shape[2:]) != EXPECTED_INPUT_SPATIAL:
                raise ValueError(
                    f"Mask for '{name}' spatial size must be {EXPECTED_INPUT_SPATIAL}, "
                    f"got ({mask.shape[2]}, {mask.shape[3]})."
                )
            masks[name] = mask.float().clamp(0.0, 1.0)
        return masks

    if not isinstance(modality_masks, torch.Tensor):
        raise TypeError(
            "modality_masks must be a torch.Tensor, dict, or None, "
            f"got {type(modality_masks).__name__}."
        )
    if modality_masks.dim() != 4:
        raise ValueError(
            "modality_masks tensor must be 4-D (B, M, H, W), "
            f"got shape {tuple(modality_masks.shape)}."
        )
    if modality_masks.shape[0] != batch_size:
        raise ValueError(
            f"modality_masks batch size {modality_masks.shape[0]} does not match "
            f"input batch size {batch_size}."
        )
    if tuple(modality_masks.shape[2:]) != EXPECTED_INPUT_SPATIAL:
        raise ValueError(
            f"modality_masks spatial size must be {EXPECTED_INPUT_SPATIAL}, "
            f"got ({modality_masks.shape[2]}, {modality_masks.shape[3]})."
        )

    num_mask_channels = modality_masks.shape[1]
    if num_mask_channels not in (len(SENSOR_MODALITIES), NUM_MODALITIES):
        raise ValueError(
            "modality_masks channel dimension must be "
            f"{len(SENSOR_MODALITIES)} (sensor groups) or "
            f"{NUM_MODALITIES} (all modalities), got {num_mask_channels}."
        )

    clamped = modality_masks.float().clamp(0.0, 1.0)
    if num_mask_channels == NUM_MODALITIES:
        for idx, name in enumerate(MODALITY_INPUT_CHANNELS):
            masks[name] = clamped[:, idx : idx + 1]
    else:
        masks["topography"] = torch.ones(
            batch_size,
            1,
            *EXPECTED_INPUT_SPATIAL,
            device=clamped.device,
            dtype=clamped.dtype,
        )
        for idx, name in enumerate(SENSOR_MODALITIES):
            masks[name] = clamped[:, idx : idx + 1]
    return masks


class ModalityInputSplitter(nn.Module):
    """Split the 59-channel fused observation tensor into modality groups.

    Channel layout follows ``DATA_PREPROCESSING_PIPELINE.md`` §4.2 and matches
    ``MODALITY_INPUT_CHANNELS`` in ``modality_encoders.py``.
    """

    def __init__(
        self,
        channel_slices: Mapping[str, slice] | None = None,
    ) -> None:
        super().__init__()
        self.channel_slices: dict[str, slice] = dict(
            channel_slices or MODALITY_CHANNEL_SLICES
        )
        expected_total = sum(
            sl.stop - sl.start for sl in self.channel_slices.values()
        )
        if expected_total != EXPECTED_INPUT_CHANNELS:
            raise ValueError(
                f"Channel slices sum to {expected_total}, expected "
                f"{EXPECTED_INPUT_CHANNELS}."
            )

    @property
    def modalities(self) -> tuple[str, ...]:
        """Ordered modality names."""
        return tuple(self.channel_slices.keys())

    def forward(self, x: torch.Tensor) -> ModalityTensorDict:
        """Split ``(B, 59, 128, 128)`` into seven modality tensors.

        Args:
            x: Fused observation tensor.

        Returns:
            Mapping from modality name to tensor
            ``(B, C_k, 128, 128)`` where ``C_k`` is the group channel count.
        """
        _validate_input_tensor(x)
        return {
            name: x[:, channel_slice]
            for name, channel_slice in self.channel_slices.items()
        }


class ModalityCMMAAdapter(nn.Module):
    """Project encoder pyramids to CMMA token space and re-fuse attended tokens."""

    def __init__(
        self,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        embed_dim: int = DEFAULT_FPN_OUT_CHANNELS,
        encoder_channels: Sequence[int] = PYRAMID_CHANNELS,
    ) -> None:
        super().__init__()
        self.modalities: tuple[str, ...] = tuple(modalities)
        self.embed_dim = embed_dim

        self.projections = nn.ModuleDict()
        self.refusion = nn.ModuleDict()
        for level_name, pyramid_level in zip(CMMA_LEVELS, CMMA_PYRAMID_LEVELS):
            level_idx = PYRAMID_LEVELS.index(pyramid_level)
            in_channels = encoder_channels[level_idx]
            self.projections[level_name] = nn.ModuleDict(
                {
                    modality: nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
                    for modality in self.modalities
                }
            )
            self.refusion[level_name] = LateralConnection(
                len(self.modalities) * embed_dim,
                embed_dim,
            )

        # Initialize gates at g = 0.5 per architecture §6.4.
        self.gate_logits = nn.Parameter(torch.zeros(len(CMMA_LEVELS)))

    def project_encoder_pyramids(
        self,
        modality_pyramids: ModalityPyramidDict,
    ) -> MultiScaleModalityFeatures:
        """Build CMMA inputs from per-modality encoder pyramids."""
        missing = [m for m in self.modalities if m not in modality_pyramids]
        if missing:
            raise ValueError(
                "Missing encoder pyramids for modalities: "
                f"{', '.join(missing)}."
            )

        multi_scale: MultiScaleModalityFeatures = {}
        for level_name, pyramid_level in zip(CMMA_LEVELS, CMMA_PYRAMID_LEVELS):
            multi_scale[level_name] = {
                modality: self.projections[level_name][modality](
                    modality_pyramids[modality][pyramid_level]
                )
                for modality in self.modalities
            }
        return multi_scale

    def refuse_level(
        self,
        attended_features: ModalityTensorDict,
        level_name: str,
    ) -> torch.Tensor:
        """Concatenate attended modality tokens and lateral-fuse to ``embed_dim``."""
        concatenated = torch.cat(
            [attended_features[modality] for modality in self.modalities],
            dim=1,
        )
        return self.refusion[level_name](concatenated)

    def blend_with_fpn(
        self,
        fused_pyramid: FusedPyramidFeatures,
        cmma_pyramid: dict[str, torch.Tensor],
    ) -> FusedPyramidFeatures:
        """Gated blend ``g · F_CMM + (1 − g) · F_FPN`` at F3–F5."""
        blended: dict[str, torch.Tensor] = dict(fused_pyramid)
        for gate_idx, level_name in enumerate(CMMA_LEVELS):
            gate = torch.sigmoid(self.gate_logits[gate_idx])
            blended[level_name] = (
                gate * cmma_pyramid[level_name]
                + (1.0 - gate) * fused_pyramid[level_name]
            )
        return FusedPyramidFeatures(
            F1=blended["F1"],
            F2=blended["F2"],
            F3=blended["F3"],
            F4=blended["F4"],
            F5=blended["F5"],
        )


class LunarPIMAFNet(nn.Module):
    """Physics-informed multi-modal attention fusion network for lunar ice detection.

    End-to-end stack:
        split → encode → FPN → CMMA (F3–F5) → decoder → heads → PCM.

    Args:
        dropout: Dropout probability forwarded to encoders and CMMA.
        fpn_channels: Unified FPN / CMMA embedding width.
    """

    def __init__(
        self,
        dropout: float = 0.1,
        fpn_channels: int = DEFAULT_FPN_OUT_CHANNELS,
    ) -> None:
        super().__init__()
        self.fpn_channels = fpn_channels
        self.modalities: tuple[str, ...] = DEFAULT_MODALITIES

        self.splitter = ModalityInputSplitter()
        self.encoders = nn.ModuleDict(
            {
                name: MODALITY_ENCODER_REGISTRY[name](dropout=dropout)
                for name in self.modalities
            }
        )
        self.fpn = MultiModalFeaturePyramid(out_channels=fpn_channels)
        self.cmma_adapter = ModalityCMMAAdapter(
            modalities=self.modalities,
            embed_dim=fpn_channels,
        )
        self.cmma = CrossModalMultiScaleAttention(
            embed_dim=fpn_channels,
            modalities=self.modalities,
            dropout=dropout,
        )
        self.decoder = AttentionUNetDecoder(fpn_channels=fpn_channels)
        self.prediction_heads = MultiTaskPredictionHead()
        self.uncertainty = EvidentialUncertaintyModule()
        self.confidence_head = ConfidenceHead()
        self.physics = PhysicsConstraintModule(feature_channels=fpn_channels)

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters.

        Args:
            trainable_only: If ``True``, count only parameters with
                ``requires_grad=True``.

        Returns:
            Parameter count.
        """
        if trainable_only:
            return sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in self.parameters())

    def freeze_encoders(self) -> None:
        """Freeze all modality encoder weights for transfer-learning stages."""
        for encoder in self.encoders.values():
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def unfreeze_encoders(self) -> None:
        """Re-enable gradients on modality encoders."""
        for encoder in self.encoders.values():
            for parameter in encoder.parameters():
                parameter.requires_grad = True

    def _encode_modalities(
        self,
        modality_tensors: ModalityTensorDict,
        modality_masks: ModalityMaskDict,
    ) -> ModalityPyramidDict:
        """Run all modality encoders."""
        pyramids: ModalityPyramidDict = {}
        for name in self.modalities:
            pyramids[name] = self.encoders[name](
                modality_tensors[name],
                mask=modality_masks[name],
            )
        return pyramids

    def _apply_cmma(
        self,
        modality_pyramids: ModalityPyramidDict,
        modality_masks: ModalityMaskDict,
        thermal_stability: torch.Tensor,
        fused_fpn: FusedPyramidFeatures,
    ) -> FusedPyramidFeatures:
        """Run CMMA at F3–F5 and gated-blend with the FPN path."""
        cmma_inputs = self.cmma_adapter.project_encoder_pyramids(modality_pyramids)

        cmma_masks: dict[str, torch.Tensor] = {}
        thermal_maps: dict[str, torch.Tensor] = {}
        for level_idx, level_name in enumerate(CMMA_LEVELS):
            spatial = CMMA_SPATIAL[level_idx]
            cmma_masks[level_name] = self._aggregate_modality_mask(
                modality_masks,
                spatial_size=spatial,
            )
            thermal_maps[level_name] = F.interpolate(
                thermal_stability,
                size=(spatial, spatial),
                mode="bilinear",
                align_corners=False,
            )

        attended = self.cmma(
            cmma_inputs,
            modality_masks=cmma_masks,
            thermal_stability_maps=thermal_maps,
        )

        cmma_fused: dict[str, torch.Tensor] = {}
        for level_name in CMMA_LEVELS:
            cmma_fused[level_name] = self.cmma_adapter.refuse_level(
                attended[level_name],
                level_name,
            )

        return self.cmma_adapter.blend_with_fpn(fused_fpn, cmma_fused)

    @staticmethod
    def _aggregate_modality_mask(
        modality_masks: ModalityMaskDict,
        spatial_size: int,
    ) -> torch.Tensor:
        """Reduce spatial masks to per-modality validity ``(B, M)`` for CMMA."""
        batch_size = next(iter(modality_masks.values())).shape[0]
        values: list[torch.Tensor] = []
        for modality in DEFAULT_MODALITIES:
            mask = modality_masks[modality]
            resized = F.interpolate(
                mask,
                size=(spatial_size, spatial_size),
                mode="nearest",
            )
            valid = resized.amax(dim=(2, 3))
            values.append(valid)
        stacked = torch.cat(values, dim=1)
        if stacked.shape != (batch_size, NUM_MODALITIES):
            raise RuntimeError(
                f"Aggregated CMMA mask shape {tuple(stacked.shape)} != "
                f"({batch_size}, {NUM_MODALITIES})."
            )
        return stacked

    def _build_physics_inputs(
        self,
        x: torch.Tensor,
        fused_pyramid: FusedPyramidFeatures,
        subsurface_ice_probability: torch.Tensor,
        modality_masks: ModalityMaskDict,
    ) -> PhysicsInputs:
        """Assemble PCM inputs from raw observations and model predictions."""
        return PhysicsInputs(
            fused_features=fused_pyramid["F4"],
            div_tbol_max=x[:, CHANNEL_DIV_TBOL_MAX : CHANNEL_DIV_TBOL_MAX + 1],
            div_insolation=x[:, CHANNEL_DIV_INSOLATION : CHANNEL_DIV_INSOLATION + 1],
            albedo_proxy=x[:, CHANNEL_LAMP_ALBEDO_RATIO : CHANNEL_LAMP_ALBEDO_RATIO + 1],
            lola_slope_deg=x[:, CHANNEL_LOLA_SLOPE_DEG : CHANNEL_LOLA_SLOPE_DEG + 1],
            lend_h_wt_pct=x[:, CHANNEL_LEND_H_WT_PCT : CHANNEL_LEND_H_WT_PCT + 1],
            lend_valid=modality_masks["neutron"],
            subsurface_ice_probability=subsurface_ice_probability,
            physics_priors=x[:, CHANNEL_PHYSICS_PRIORS],
        )

    def forward_features(
        self,
        x: torch.Tensor,
        modality_masks: torch.Tensor | ModalityMaskDict | None = None,
    ) -> LunarPIMAFFeatures:
        """Encode, fuse, attend, and decode without running prediction heads.

        Args:
            x: Fused observation tensor ``(B, 59, 128, 128)``.
            modality_masks: Optional validity masks as ``(B, 6|7, 128, 128)``
                tensor or per-modality ``(B, 1, 128, 128)`` dict.

        Returns:
            Fused pyramid, decoder embedding, and per-modality encoder pyramids.
        """
        batch_size = _validate_input_tensor(x)
        masks = _normalize_modality_masks(
            modality_masks,
            batch_size,
            device=x.device,
            dtype=x.dtype,
        )

        modality_tensors = self.splitter(x)
        modality_pyramids = self._encode_modalities(modality_tensors, masks)
        fused_fpn = self.fpn(modality_pyramids)

        thermal_stability = modality_tensors["thermal"][
            :, THERMAL_ICE_STABILITY_OFFSET : THERMAL_ICE_STABILITY_OFFSET + 1
        ]
        fused_pyramid = self._apply_cmma(
            modality_pyramids,
            masks,
            thermal_stability,
            fused_fpn,
        )

        decoder_features = self.decoder(fused_pyramid)
        return LunarPIMAFFeatures(
            fused_pyramid=fused_pyramid,
            decoder_features=decoder_features,
            modality_pyramids=modality_pyramids,
        )

    def forward_predictions(
        self,
        decoder_features: torch.Tensor,
    ) -> LunarPIMAFPredictions:
        """Run prediction, uncertainty, and confidence heads on decoder features.

        Args:
            decoder_features: Tensor of shape ``(B, 128, 128, 128)``.

        Returns:
            Segmentation, ice probabilities, uncertainty maps, and confidence.
        """
        if decoder_features.shape[1] != DECODER_OUTPUT_CHANNELS:
            raise ValueError(
                f"decoder_features expected {DECODER_OUTPUT_CHANNELS} channels, "
                f"got {decoder_features.shape[1]}."
            )

        predictions: PredictionOutputs = self.prediction_heads(decoder_features)
        uncertainty: UncertaintyOutputs = self.uncertainty(decoder_features)
        confidence = self.confidence_head(decoder_features)

        return LunarPIMAFPredictions(
            segmentation_logits=predictions["segmentation_logits"],
            surface_ice_probability=predictions["surface_ice_probability"],
            subsurface_ice_probability=predictions["subsurface_ice_probability"],
            dirichlet_alpha=uncertainty["dirichlet_alpha"],
            epistemic_uncertainty=uncertainty["epistemic_uncertainty"],
            aleatoric_uncertainty=uncertainty["aleatoric_uncertainty"],
            total_uncertainty=uncertainty["total_uncertainty"],
            confidence=confidence,
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_masks: torch.Tensor | ModalityMaskDict | None = None,
    ) -> LunarPIMAFOutput:
        """Full forward pass through the LUNAR-PIMAF-Net stack.

        Args:
            x: Fused observation tensor ``(B, 59, 128, 128)``.
            modality_masks: Optional per-modality or per-sensor-group validity masks.

        Returns:
            Complete inference dictionary including predictions, uncertainty,
            confidence, physics residuals, decoder features, and fused pyramid.
        """
        batch_size = _validate_input_tensor(x)
        masks = _normalize_modality_masks(
            modality_masks,
            batch_size,
            device=x.device,
            dtype=x.dtype,
        )

        features = self.forward_features(x, modality_masks=masks)
        predictions = self.forward_predictions(features.decoder_features)

        physics_out = self.physics(
            self._build_physics_inputs(
                x=x,
                fused_pyramid=features.fused_pyramid,
                subsurface_ice_probability=predictions.subsurface_ice_probability,
                modality_masks=masks,
            )
        )

        return LunarPIMAFOutput(
            segmentation_logits=predictions.segmentation_logits,
            surface_ice_probability=predictions.surface_ice_probability,
            subsurface_ice_probability=predictions.subsurface_ice_probability,
            dirichlet_alpha=predictions.dirichlet_alpha,
            epistemic_uncertainty=predictions.epistemic_uncertainty,
            aleatoric_uncertainty=predictions.aleatoric_uncertainty,
            total_uncertainty=predictions.total_uncertainty,
            confidence=predictions.confidence,
            physics_residuals=physics_out["residuals"],
            decoder_features=features.decoder_features,
            fused_pyramid=features.fused_pyramid,
        )

    def model_summary(self, input_shape: tuple[int, ...] = (1, 59, 128, 128)) -> str:
        """Return a human-readable module and parameter summary.

        Args:
            input_shape: Nominal input shape ``(B, C, H, W)`` used in the header.

        Returns:
            Multi-line summary string.
        """
        lines: list[str] = [
            "LUNAR-PIMAF-Net",
            f"  input shape: {tuple(input_shape)}",
            f"  modalities: {', '.join(self.modalities)}",
            f"  fpn channels: {self.fpn_channels}",
            f"  decoder channels: {DECODER_OUTPUT_CHANNELS}",
            "",
            "Submodules:",
        ]

        submodule_groups: tuple[tuple[str, nn.Module | nn.ModuleDict], ...] = (
            ("splitter", self.splitter),
            ("encoders", self.encoders),
            ("fpn", self.fpn),
            ("cmma_adapter", self.cmma_adapter),
            ("cmma", self.cmma),
            ("decoder", self.decoder),
            ("prediction_heads", self.prediction_heads),
            ("uncertainty", self.uncertainty),
            ("confidence_head", self.confidence_head),
            ("physics", self.physics),
        )

        for name, module in submodule_groups:
            if isinstance(module, nn.ModuleDict):
                for child_name, child in module.items():
                    params = sum(parameter.numel() for parameter in child.parameters())
                    lines.append(f"  {name}.{child_name:<12} {params:>12,} params")
            else:
                params = sum(parameter.numel() for parameter in module.parameters())
                lines.append(f"  {name:<22} {params:>12,} params")

        total = self.count_parameters()
        trainable = self.count_parameters(trainable_only=True)
        lines.extend(
            [
                "",
                f"Total parameters:     {total:>12,}",
                f"Trainable parameters: {trainable:>12,}",
            ]
        )
        return "\n".join(lines)


__all__ = [
    "CHANNEL_DIV_ICE_STABILITY",
    "CHANNEL_DIV_INSOLATION",
    "CHANNEL_DIV_TBOL_MAX",
    "CHANNEL_LAMP_ALBEDO_RATIO",
    "CHANNEL_LEND_H_WT_PCT",
    "CHANNEL_LOLA_SLOPE_DEG",
    "CHANNEL_PHYSICS_PRIORS",
    "EXPECTED_INPUT_CHANNELS",
    "LunarPIMAFFeatures",
    "LunarPIMAFNet",
    "LunarPIMAFOutput",
    "LunarPIMAFPredictions",
    "ModalityCMMAAdapter",
    "ModalityInputSplitter",
    "MODALITY_CHANNEL_SLICES",
    "SENSOR_MODALITIES",
]
