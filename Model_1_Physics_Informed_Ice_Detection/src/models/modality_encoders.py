"""Modality-specific encoders for LUNAR-PIMAF-Net.

Each lunar remote-sensing instrument produces observations with distinct physical
units, noise characteristics, and spatial fidelity.  Independent encoders preserve
these statistical boundaries before cross-modal fusion in the shared FPN / CMMA stack.

Reference: docs/ARCHITECTURE_SPECIFICATION.md §4–§5
"""

from __future__ import annotations

import math
from typing import Final, Optional, Sequence, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pyramid level specification (spatial size at 128×128 input)
# ---------------------------------------------------------------------------

PYRAMID_LEVELS: Final[tuple[str, ...]] = ("P1", "P2", "P3", "P4", "P5")

PYRAMID_CHANNELS: Final[tuple[int, ...]] = (64, 128, 256, 512, 768)

PYRAMID_SPATIAL: Final[tuple[int, ...]] = (128, 64, 32, 16, 8)

# Fixed encoder input geometry (240 m/pixel polar patch contract).
EXPECTED_INPUT_SPATIAL: Final[tuple[int, int]] = (128, 128)

# Input channel counts aligned with DATA_PREPROCESSING_PIPELINE.md §4.2
MODALITY_INPUT_CHANNELS: Final[dict[str, int]] = {
    "topography": 8,
    "radar": 10,
    "thermal": 15,
    "uv": 6,
    "neutron": 6,
    "spectral": 7,
    "physics": 7,
}


class PyramidFeatures(TypedDict):
    """Multi-scale encoder output keyed by pyramid level."""

    P1: torch.Tensor
    P2: torch.Tensor
    P3: torch.Tensor
    P4: torch.Tensor
    P5: torch.Tensor


def _num_groups(channels: int) -> int:
    """Select a GroupNorm group count that divides ``channels``."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _validate_encoder_input(
    x: torch.Tensor,
    in_channels: int,
    modality_name: str,
) -> None:
    """Validate modality tensor shape ``(B, C_in, 128, 128)``."""
    if x.dim() != 4:
        raise ValueError(
            f"{modality_name} encoder expected 4-D input (B, C, H, W), "
            f"got {x.dim()}-D tensor with shape {tuple(x.shape)}."
        )
    if x.shape[1] != in_channels:
        raise ValueError(
            f"{modality_name} encoder expected {in_channels} channels, got {x.shape[1]}."
        )
    if tuple(x.shape[2:]) != EXPECTED_INPUT_SPATIAL:
        raise ValueError(
            f"{modality_name} encoder expected input shape "
            f"(B, {in_channels}, 128, 128), got "
            f"(B, {x.shape[1]}, {x.shape[2]}, {x.shape[3]})."
        )


def _validate_and_prepare_mask(
    mask: torch.Tensor,
    batch_size: int,
    modality_name: str,
) -> torch.Tensor:
    """Validate mask shape ``(B, 1, 128, 128)`` and cast to float."""
    if mask.dim() != 4:
        raise ValueError(
            f"{modality_name} encoder mask must be 4-D with shape (B, 1, 128, 128), "
            f"got {mask.dim()}-D tensor with shape {tuple(mask.shape)}."
        )
    if mask.shape[0] != batch_size:
        raise ValueError(
            f"{modality_name} encoder mask batch size {mask.shape[0]} does not match "
            f"input batch size {batch_size}."
        )
    if mask.shape[1] != 1:
        raise ValueError(
            f"{modality_name} encoder mask must have a single channel (B, 1, 128, 128), "
            f"got shape {tuple(mask.shape)}."
        )
    if tuple(mask.shape[2:]) != EXPECTED_INPUT_SPATIAL:
        raise ValueError(
            f"{modality_name} encoder mask spatial size must be (128, 128), "
            f"got ({mask.shape[2]}, {mask.shape[3]})."
        )
    return mask.float().clamp(0.0, 1.0)


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Down/up-sample a validity mask to ``(height, width)``."""
    return F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvNormAct(nn.Module):
    """Convolution → GroupNorm → SiLU with optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3×3 convolution followed by pointwise 1×1 projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = 1
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=in_channels,
            dropout=0.0,
        )
        self.pointwise = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class ResidualBlock(nn.Module):
    """Depthwise-separable residual block with optional channel projection."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        expansion: float = 1.0,
    ) -> None:
        super().__init__()
        hidden = max(int(channels * expansion), channels)
        self.conv1 = DepthwiseSeparableConv(channels, hidden, stride=1, dropout=0.0)
        self.conv2 = DepthwiseSeparableConv(hidden, channels, stride=1, dropout=dropout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + residual)


class DownsampleBlock(nn.Module):
    """Anti-aliased spatial downsampling with channel expansion.

    Applies a depthwise box-filter blur (low-pass) before a strided convolution to
    reduce shift-sensitive aliasing in planetary tile boundaries.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blur = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        nn.init.constant_(blur.weight, 1.0 / 9.0)
        self.blur = blur
        self.down = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.blur(x))


class MaskedInstanceNorm2d(nn.Module):
    """Instance normalization that excludes invalid pixels from moment estimation.

    For planetary remote sensing with structured missingness (PSR gaps, coarse
    neutron coverage), statistics are computed only over valid pixels indicated by
    ``mask``. Invalid locations do not bias the per-channel mean or variance.

    When ``mask`` is ``None``, behavior matches ``nn.InstanceNorm2d``.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mean = x.mean(dim=(2, 3), keepdim=True)
            variance = x.var(dim=(2, 3), unbiased=False, keepdim=True)
            x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        else:
            # Validity weights: (B, 1, H, W), broadcast across channels.
            weights = mask
            valid_count = weights.sum(dim=(2, 3), keepdim=True)
            has_valid = (valid_count > 0).to(dtype=x.dtype)
            safe_count = valid_count.clamp(min=1.0)

            mean = (x * weights).sum(dim=(2, 3), keepdim=True) / safe_count
            centered = (x - mean) * weights
            variance = centered.pow(2).sum(dim=(2, 3), keepdim=True) / safe_count
            x_norm = (x - mean) / torch.sqrt(variance + self.eps)

            # Preserve raw values when no valid pixels exist in a batch element.
            x_norm = x_norm * has_valid + x * (1.0 - has_valid)

        if self.weight is not None and self.bias is not None:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x_norm = x_norm * weight + bias

        return x_norm


class EncoderStage(nn.Module):
    """Stack of residual blocks operating at a fixed spatial resolution."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, dropout=dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MissingnessModulation(nn.Module):
    """Inject learned embeddings at invalid pixels (M3-in-PSR, coarse LEND gaps).

    Implements the mask-aware input rule from the architecture specification:
        x' = x ⊙ mask + η ⊙ (1 − mask)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x
        return x * mask + self.embedding * (1.0 - mask)


class MaskFeatureGate(nn.Module):
    """Softly attenuate activations where the modality is unobserved."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Learnable floor prevents complete feature erasure at invalid pixels.
        self.invalid_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x
        mask = _resize_mask(mask, x.shape[-2], x.shape[-1])
        gate = mask + self.invalid_scale * (1.0 - mask)
        return x * gate


class SinusoidalPositionalEncoding2D(nn.Module):
    """Fixed 2-D sinusoidal positional encoding for topographic reference frames.

    The encoding tensor is built once per spatial resolution and cached via
    ``register_buffer`` for AMP-stable, allocation-free subsequent forwards.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(
                f"SinusoidalPositionalEncoding2D requires channels divisible by 4, "
                f"got {channels}"
            )
        self.channels = channels
        self.register_buffer("pe", torch.empty(0), persistent=False)
        self._cached_hw: tuple[int, int] = (0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` matching its spatial shape."""
        height, width = x.shape[-2], x.shape[-1]
        pe = self._get_positional_encoding(height, width, x.device)
        return x + pe.to(dtype=x.dtype)

    def _get_positional_encoding(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return cached PE, lazily building it on first use or resolution change."""
        if (
            self.pe.numel() == 0
            or self._cached_hw != (height, width)
            or self.pe.device != device
        ):
            pe = self._build_encoding(height, width, device)
            self.register_buffer("pe", pe, persistent=False)
            self._cached_hw = (height, width)
        return self.pe

    def _build_encoding(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build sinusoidal PE in float32 for numerical stability under AMP."""
        dtype = torch.float32
        half = self.channels // 2
        quarter = self.channels // 4

        y_pos = torch.arange(height, device=device, dtype=dtype).unsqueeze(1)
        x_pos = torch.arange(width, device=device, dtype=dtype).unsqueeze(0)

        div_term = torch.exp(
            torch.arange(0, quarter, device=device, dtype=dtype)
            * (-math.log(10_000.0) / quarter)
        )

        pe = torch.zeros(1, self.channels, height, width, device=device, dtype=dtype)

        for i in range(quarter):
            pe[:, i, :, :] = torch.sin(y_pos * div_term[i]).expand(height, width)
            pe[:, quarter + i, :, :] = torch.cos(y_pos * div_term[i]).expand(height, width)
            pe[:, half + i, :, :] = torch.sin(x_pos * div_term[i]).expand(height, width)
            pe[:, half + quarter + i, :, :] = torch.cos(x_pos * div_term[i]).expand(
                height, width
            )

        return pe


# ---------------------------------------------------------------------------
# Base encoder
# ---------------------------------------------------------------------------


class BaseModalityEncoder(nn.Module):
    """Shared multi-scale encoder backbone for a single remote-sensing modality.

    Produces a five-level feature pyramid:
        P1: (B,  64, 128, 128)
        P2: (B, 128,  64,  64)
        P3: (B, 256,  32,  32)
        P4: (B, 512,  16,  16)
        P5: (B, 768,   8,   8)

    Args:
        in_channels: Number of input feature channels for this modality.
        stage_depths: Residual block counts for pyramid levels P1–P5.
        dropout: Dropout probability inside residual blocks.
        modality_name: Identifier used for logging / diagnostics.
        use_positional_encoding: Whether to add 2-D sinusoidal PE at P1 (topography).
    """

    def __init__(
        self,
        in_channels: int,
        stage_depths: Sequence[int] = (2, 2, 2, 1, 1),
        dropout: float = 0.1,
        modality_name: str = "modality",
        use_positional_encoding: bool = False,
    ) -> None:
        super().__init__()

        if len(stage_depths) != len(PYRAMID_LEVELS):
            raise ValueError(
                f"stage_depths must have {len(PYRAMID_LEVELS)} entries, "
                f"got {len(stage_depths)}"
            )

        self.in_channels = in_channels
        self.stage_depths = tuple(stage_depths)
        self.modality_name = modality_name
        self.dropout = dropout

        # Input boundary: masked instance norm handles N/S pole shift without
        # invalid pixels contaminating spatial moment estimates.
        self.input_norm = MaskedInstanceNorm2d(in_channels, affine=True)

        # Project raw sensor channels to P1 width.
        self.stem = ConvNormAct(
            in_channels,
            PYRAMID_CHANNELS[0],
            kernel_size=3,
            stride=1,
            padding=1,
            dropout=dropout,
        )
        self.missingness = MissingnessModulation(in_channels)
        self.post_stem_gate = MaskFeatureGate(PYRAMID_CHANNELS[0])

        self.positional_encoding: Optional[SinusoidalPositionalEncoding2D]
        if use_positional_encoding:
            self.positional_encoding = SinusoidalPositionalEncoding2D(PYRAMID_CHANNELS[0])
        else:
            self.positional_encoding = None

        # Encoder stages and inter-level downsamplers.
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.stage_gates = nn.ModuleList()

        for level_idx, (channels, num_blocks) in enumerate(
            zip(PYRAMID_CHANNELS, self.stage_depths)
        ):
            self.stages.append(EncoderStage(channels, num_blocks, dropout=dropout))
            self.stage_gates.append(MaskFeatureGate(channels))

            if level_idx < len(PYRAMID_LEVELS) - 1:
                next_channels = PYRAMID_CHANNELS[level_idx + 1]
                self.downsamplers.append(
                    DownsampleBlock(channels, next_channels, dropout=dropout)
                )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> PyramidFeatures:
        """Encode a modality tensor into a five-level feature pyramid.

        Args:
            x: Modality input of shape ``(B, C_in, 128, 128)``.
            mask: Optional validity mask of shape ``(B, 1, 128, 128)`` where ``1``
                denotes a valid observation and ``0`` missing data.

        Returns:
            Dictionary with keys ``P1`` … ``P5`` mapping to feature tensors.
        """
        _validate_encoder_input(x, self.in_channels, self.modality_name)

        norm_mask: Optional[torch.Tensor] = None
        if mask is not None:
            norm_mask = _validate_and_prepare_mask(mask, x.shape[0], self.modality_name)

        x = self.input_norm(x, norm_mask)
        x = self.missingness(x, norm_mask)
        x = self.stem(x)
        x = self.post_stem_gate(x, norm_mask)

        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        outputs: dict[str, torch.Tensor] = {}

        for level_idx, (level_name, spatial_size) in enumerate(
            zip(PYRAMID_LEVELS, PYRAMID_SPATIAL)
        ):
            level_mask = (
                _resize_mask(norm_mask, spatial_size, spatial_size)
                if norm_mask is not None
                else None
            )

            x = self.stages[level_idx](x)
            x = self.stage_gates[level_idx](x, level_mask)
            outputs[level_name] = x

            if level_idx < len(self.downsamplers):
                x = self.downsamplers[level_idx](x)

        return PyramidFeatures(
            P1=outputs["P1"],
            P2=outputs["P2"],
            P3=outputs["P3"],
            P4=outputs["P4"],
            P5=outputs["P5"],
        )


# ---------------------------------------------------------------------------
# Modality-specific encoders
# ---------------------------------------------------------------------------


class TopographyEncoder(BaseModalityEncoder):
    """LOLA-derived topographic and geometric features (channels 0–7).

  Encodes elevation, slope, aspect, roughness, TPI, curvature, and PSR fraction.
  Includes 2-D sinusoidal positional encoding because topography defines the
  spatial reference frame for all fused modalities.
    """

    IN_CHANNELS: Final[int] = MODALITY_INPUT_CHANNELS["topography"]

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(
            in_channels=self.IN_CHANNELS,
            stage_depths=(2, 2, 2, 1, 1),
            dropout=dropout,
            modality_name="topography",
            use_positional_encoding=True,
        )


class RadarEncoder(BaseModalityEncoder):
    """Mini-RF S-band radar features (channels 8–17).

    Encodes CPR, roughness-corrected CPR, Stokes parameters, and m-chi
    decomposition terms sensitive to dielectric contrast from subsurface ice.
    """

    IN_CHANNELS: Final[int] = MODALITY_INPUT_CHANNELS["radar"]

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(
            in_channels=self.IN_CHANNELS,
            stage_depths=(2, 2, 2, 1, 1),
            dropout=dropout,
            modality_name="radar",
        )


class ThermalEncoder(BaseModalityEncoder):
    """Diviner thermal infrared features (channels 18–32).

    Deepest P1 stack among encoders because thermal stability is the primary
    thermodynamic gate for ice retention in permanently shadowed regions.
    """

    IN_CHANNELS: Final[int] = MODALITY_INPUT_CHANNELS["thermal"]

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(
            in_channels=self.IN_CHANNELS,
            stage_depths=(3, 2, 2, 1, 1),
            dropout=dropout,
            modality_name="thermal",
        )


class UVEncoder(BaseModalityEncoder):
    """LAMP far-UV features (channels 33–38).

    Encodes on/off-band albedo, H₂O absorption depth, and temporal variability
    indicative of surface water frost in permanently shadowed regions.
    """

    IN_CHANNELS: Final[int] = MODALITY_INPUT_CHANNELS["uv"]

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(
            in_channels=self.IN_CHANNELS,
            stage_depths=(2, 1, 1, 1, 1),
            dropout=dropout,
            modality_name="uv",
        )


class NeutronEncoder(BaseModalityEncoder):
    """LEND neutron spectrometer features (channels 39–44).

    Encodes epithermal / fast neutron counts and hydrogen-equivalent maps.
    Native LEND resolution is coarse (~5 km); validity masks are critical.
    """

    IN_CHANNELS: Final[int] = MODALITY_INPUT_CHANNELS["neutron"]

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(
            in_channels=self.IN_CHANNELS,
            stage_depths=(2, 1, 1, 1, 1),
            dropout=dropout,
            modality_name="neutron",
        )


class SpectralEncoder(BaseModalityEncoder):
    """Chandrayaan-1 M3 spectral features (channels 45–51).

    Encodes continuum reflectance, OH/H₂O band depths, and spectral slopes.
    Valid only on sunlit pixels; mask is expected to be zero inside PSR interiors.
    """

    IN_CHANNELS: Final[int] = MODALITY_INPUT_CHANNELS["spectral"]

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(
            in_channels=self.IN_CHANNELS,
            stage_depths=(2, 2, 1, 1, 1),
            dropout=dropout,
            modality_name="spectral",
        )


class PhysicsEncoder(BaseModalityEncoder):
    """Pre-computed physics-derived channels (channels 52–58).

    Encodes Stefan residuals, ice-retention probability, multi-sensor agreement,
    and subsurface accessibility indices used as anchors for the DCM physics layer.
    """

    IN_CHANNELS: Final[int] = MODALITY_INPUT_CHANNELS["physics"]

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(
            in_channels=self.IN_CHANNELS,
            stage_depths=(1, 1, 1, 1, 1),
            dropout=dropout,
            modality_name="physics",
        )


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


MODALITY_ENCODER_REGISTRY: Final[dict[str, type[BaseModalityEncoder]]] = {
    "topography": TopographyEncoder,
    "radar": RadarEncoder,
    "thermal": ThermalEncoder,
    "uv": UVEncoder,
    "neutron": NeutronEncoder,
    "spectral": SpectralEncoder,
    "physics": PhysicsEncoder,
}


def build_modality_encoder(
    modality: str,
    dropout: float = 0.1,
) -> BaseModalityEncoder:
    """Instantiate an encoder by modality name.

    Args:
        modality: One of ``topography``, ``radar``, ``thermal``, ``uv``,
            ``neutron``, ``spectral``, ``physics``.
        dropout: Dropout probability forwarded to the encoder constructor.

    Returns:
        Configured encoder instance.

    Raises:
        KeyError: If ``modality`` is not registered.
    """
    if modality not in MODALITY_ENCODER_REGISTRY:
        registered = ", ".join(sorted(MODALITY_ENCODER_REGISTRY))
        raise KeyError(f"Unknown modality '{modality}'. Registered: {registered}")
    return MODALITY_ENCODER_REGISTRY[modality](dropout=dropout)


__all__ = [
    "PYRAMID_CHANNELS",
    "PYRAMID_LEVELS",
    "PYRAMID_SPATIAL",
    "MODALITY_INPUT_CHANNELS",
    "MODALITY_ENCODER_REGISTRY",
    "PyramidFeatures",
    "ConvNormAct",
    "DepthwiseSeparableConv",
    "ResidualBlock",
    "DownsampleBlock",
    "EncoderStage",
    "EXPECTED_INPUT_SPATIAL",
    "MaskedInstanceNorm2d",
    "MissingnessModulation",
    "MaskFeatureGate",
    "SinusoidalPositionalEncoding2D",
    "BaseModalityEncoder",
    "TopographyEncoder",
    "RadarEncoder",
    "ThermalEncoder",
    "UVEncoder",
    "NeutronEncoder",
    "SpectralEncoder",
    "PhysicsEncoder",
    "build_modality_encoder",
]
