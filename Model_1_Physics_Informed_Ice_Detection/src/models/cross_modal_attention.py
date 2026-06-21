"""Cross-Modal Multi-Scale Attention (CMMA) for LUNAR-PIMAF-Net.

Enables conditional information exchange between lunar remote-sensing modalities at
coarse pyramid levels (F3–F5), where crater-scale context dominates and LEND
upsampling noise is partially averaged out.

Reference: docs/ARCHITECTURE_SPECIFICATION.md §6
"""

from __future__ import annotations

from typing import Final, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_pyramid import DEFAULT_FPN_OUT_CHANNELS, DEFAULT_MODALITIES

# ---------------------------------------------------------------------------
# CMMA level specification (fused FPN levels F3–F5)
# ---------------------------------------------------------------------------

CMMA_LEVELS: Final[tuple[str, ...]] = ("F3", "F4", "F5")

CMMA_SPATIAL: Final[tuple[int, ...]] = (32, 16, 8)

ModalityFeatures = dict[str, torch.Tensor]
MultiScaleModalityFeatures = dict[str, ModalityFeatures]

DEFAULT_NUM_HEADS: Final[int] = 8
DEFAULT_DROPOUT: Final[float] = 0.1
DEFAULT_MLP_RATIO: Final[float] = 4.0

# Modality indices (must match DEFAULT_MODALITIES order).
MODALITY_INDEX: Final[dict[str, int]] = {
    name: idx for idx, name in enumerate(DEFAULT_MODALITIES)
}

THERMAL_INDEX: Final[int] = MODALITY_INDEX["thermal"]
RADAR_INDEX: Final[int] = MODALITY_INDEX["radar"]
NEUTRON_INDEX: Final[int] = MODALITY_INDEX["neutron"]

# Literature-informed attention coupling priors (query → key), from architecture §6.3.
_PRIOR_ATTENTION_BIAS: Final[tuple[tuple[float, ...], ...]] = (
    # topo   radar  therm   uv   neutron spectral phys
    (0.00, 0.50, 0.30, 0.20, 0.20, 0.30, 0.40),  # topography
    (0.90, 0.00, 0.80, 0.20, 0.70, 0.10, 0.80),  # radar
    (0.30, 0.80, 0.00, 0.60, 0.90, 0.20, 1.00),  # thermal
    (0.20, 0.20, 0.50, 0.00, 0.20, 0.40, 0.30),  # uv
    (0.20, 0.70, 0.90, 0.30, 0.00, 0.10, 0.90),  # neutron
    (0.20, 0.10, 0.20, 0.30, 0.10, 0.00, 0.20),  # spectral
    (0.40, 0.80, 1.00, 0.30, 0.90, 0.20, 0.00),  # physics
)


def _validate_modality_features(
    modality_features: ModalityFeatures,
    expected_modalities: Sequence[str],
    embed_dim: int,
    spatial_size: int,
    context: str,
) -> int:
    """Validate a single-scale modality feature mapping.

    Args:
        modality_features: Mapping from modality name to feature tensor.
        expected_modalities: Ordered modality names required in the mapping.
        embed_dim: Expected channel dimension per modality tensor.
        spatial_size: Expected height and width.
        context: String used in error messages (e.g. level name).

    Returns:
        Batch size ``B``.

    Raises:
        ValueError: On missing modalities or inconsistent tensor shapes.
        TypeError: If values are not tensors.
    """
    if not modality_features:
        raise ValueError(f"{context}: modality_features must not be empty.")

    missing = [name for name in expected_modalities if name not in modality_features]
    if missing:
        raise ValueError(
            f"{context}: missing modality tensors: {', '.join(missing)}."
        )

    batch_size: int | None = None
    for modality in expected_modalities:
        tensor = modality_features[modality]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{context}: modality '{modality}' must be a torch.Tensor, "
                f"got {type(tensor).__name__}."
            )
        if tensor.dim() != 4:
            raise ValueError(
                f"{context}: modality '{modality}' must be 4-D (B, C, H, W), "
                f"got shape {tuple(tensor.shape)}."
            )
        if tensor.shape[1] != embed_dim:
            raise ValueError(
                f"{context}: modality '{modality}' expected {embed_dim} channels, "
                f"got {tensor.shape[1]}."
            )
        if tuple(tensor.shape[2:]) != (spatial_size, spatial_size):
            raise ValueError(
                f"{context}: modality '{modality}' expected spatial size "
                f"({spatial_size}, {spatial_size}), got "
                f"({tensor.shape[2]}, {tensor.shape[3]})."
            )
        if batch_size is None:
            batch_size = tensor.shape[0]
        elif tensor.shape[0] != batch_size:
            raise ValueError(
                f"{context}: inconsistent batch size for modality '{modality}': "
                f"expected {batch_size}, got {tensor.shape[0]}."
            )

    if batch_size is None:
        raise ValueError(f"{context}: could not infer batch size.")

    return batch_size


def _validate_modality_mask(
    modality_mask: torch.Tensor,
    batch_size: int,
    num_modalities: int,
    context: str,
) -> torch.Tensor:
    """Validate and cast a per-modality validity mask of shape ``(B, M)``."""
    if not isinstance(modality_mask, torch.Tensor):
        raise TypeError(
            f"{context}: modality_mask must be a torch.Tensor, "
            f"got {type(modality_mask).__name__}."
        )
    if modality_mask.dim() != 2:
        raise ValueError(
            f"{context}: modality_mask must be 2-D (B, M), "
            f"got shape {tuple(modality_mask.shape)}."
        )
    if modality_mask.shape[0] != batch_size:
        raise ValueError(
            f"{context}: modality_mask batch size {modality_mask.shape[0]} does not "
            f"match feature batch size {batch_size}."
        )
    if modality_mask.shape[1] != num_modalities:
        raise ValueError(
            f"{context}: modality_mask expected {num_modalities} modalities, "
            f"got {modality_mask.shape[1]}."
        )
    return modality_mask.float().clamp(0.0, 1.0)


def _validate_stability_map(
    thermal_stability: torch.Tensor,
    batch_size: int,
    spatial_size: int,
    context: str,
) -> torch.Tensor:
    """Validate a thermal stability map of shape ``(B, 1, H, W)``."""
    if not isinstance(thermal_stability, torch.Tensor):
        raise TypeError(
            f"{context}: thermal_stability must be a torch.Tensor, "
            f"got {type(thermal_stability).__name__}."
        )
    if thermal_stability.dim() != 4:
        raise ValueError(
            f"{context}: thermal_stability must be 4-D (B, 1, H, W), "
            f"got shape {tuple(thermal_stability.shape)}."
        )
    if thermal_stability.shape[0] != batch_size:
        raise ValueError(
            f"{context}: thermal_stability batch size {thermal_stability.shape[0]} "
            f"does not match feature batch size {batch_size}."
        )
    if thermal_stability.shape[1] != 1:
        raise ValueError(
            f"{context}: thermal_stability must have a single channel, "
            f"got shape {tuple(thermal_stability.shape)}."
        )
    if tuple(thermal_stability.shape[2:]) != (spatial_size, spatial_size):
        raise ValueError(
            f"{context}: thermal_stability spatial size must be "
            f"({spatial_size}, {spatial_size}), got "
            f"({thermal_stability.shape[2]}, {thermal_stability.shape[3]})."
        )
    return thermal_stability.float().clamp(0.0, 1.0)


def _stack_modalities(
    modality_features: ModalityFeatures,
    modalities: Sequence[str],
) -> torch.Tensor:
    """Stack modality maps into ``(B, H, W, M, C)``."""
    tensors = [modality_features[name] for name in modalities]
    stacked = torch.stack(tensors, dim=1)  # (B, M, C, H, W)
    return stacked.permute(0, 3, 4, 1, 2).contiguous()


def _unstack_modalities(
    tokens: torch.Tensor,
    modalities: Sequence[str],
) -> ModalityFeatures:
    """Convert ``(B, H, W, M, C)`` back to a modality feature mapping."""
    permuted = tokens.permute(0, 3, 4, 1, 2).contiguous()  # (B, M, C, H, W)
    return {name: permuted[:, idx] for idx, name in enumerate(modalities)}


class CrossModalSelfAttention(nn.Module):
    """Multi-head self-attention across modalities with additive coupling bias.

    Operates on sequences of shape ``(N, M, C)`` where ``M`` is the number of
    modalities at a single spatial location.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_modalities: int,
        dropout: float = DEFAULT_DROPOUT,
        bias_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_modalities = num_modalities
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        prior = torch.tensor(_PRIOR_ATTENTION_BIAS, dtype=torch.float32)
        self.attention_bias = nn.Parameter(prior * bias_strength)

        # Extra coupling strengths gated by thermal stability (cold-trap regions).
        self.radar_thermal_gate = nn.Parameter(torch.tensor(1.0))
        self.neutron_thermal_gate = nn.Parameter(torch.tensor(1.0))

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, M, C)`` → ``(N, heads, M, head_dim)``."""
        n_tokens, num_modalities, _ = x.shape
        x = x.view(n_tokens, num_modalities, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, heads, M, head_dim)`` → ``(N, M, C)``."""
        n_tokens, _, num_modalities, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(n_tokens, num_modalities, self.embed_dim)

    def _build_thermal_guided_bias(
        self,
        batch_size: int,
        spatial_size: int,
        thermal_stability: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build per-location bias boosting radar/neutron → thermal in cold traps."""
        num_positions = batch_size * spatial_size * spatial_size
        bias = self.attention_bias.to(device=device, dtype=dtype).unsqueeze(0)
        bias = bias.expand(num_positions, -1, -1).clone()

        if thermal_stability is None:
            return bias

        stability = thermal_stability.view(batch_size, 1, -1)  # (B, 1, HW)
        stability = stability.permute(0, 2, 1).reshape(num_positions, 1)  # (N, 1)

        radar_coupling = self.radar_thermal_gate.to(dtype=dtype) * stability
        neutron_coupling = self.neutron_thermal_gate.to(dtype=dtype) * stability

        bias[:, RADAR_INDEX, THERMAL_INDEX] = (
            bias[:, RADAR_INDEX, THERMAL_INDEX] + radar_coupling.squeeze(-1)
        )
        bias[:, NEUTRON_INDEX, THERMAL_INDEX] = (
            bias[:, NEUTRON_INDEX, THERMAL_INDEX] + neutron_coupling.squeeze(-1)
        )
        return bias

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_size: int,
        modality_mask: Optional[torch.Tensor] = None,
        thermal_stability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multi-head self-attention across the modality axis.

        Args:
            tokens: Modality tokens ``(N, M, C)`` where ``N = B·H·W``.
            spatial_size: Spatial edge length ``H = W``.
            modality_mask: Optional validity mask ``(B, M)``; ``0`` marks missing
                modalities excluded from keys/values.
            thermal_stability: Optional cold-trap stability map ``(B, 1, H, W)``
                used to strengthen radar/neutron attention to thermal features.

        Returns:
            Updated tokens with the same shape as ``tokens``.
        """
        if tokens.dim() != 3:
            raise ValueError(
                f"CrossModalSelfAttention expects tokens of shape (N, M, C), "
                f"got {tuple(tokens.shape)}."
            )
        if tokens.shape[1] != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {tokens.shape[1]}."
            )
        if spatial_size <= 0:
            raise ValueError(f"spatial_size must be positive, got {spatial_size}.")

        num_positions, _, _ = tokens.shape
        grid_cells = spatial_size * spatial_size
        if num_positions % grid_cells != 0:
            raise ValueError(
                f"Token count {num_positions} is not divisible by "
                f"spatial_size² ({grid_cells})."
            )
        batch_size = num_positions // grid_cells

        query = self._reshape_heads(self.q_proj(tokens))
        key = self._reshape_heads(self.k_proj(tokens))
        value = self._reshape_heads(self.v_proj(tokens))

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        attn_bias = self._build_thermal_guided_bias(
            batch_size=batch_size,
            spatial_size=spatial_size,
            thermal_stability=thermal_stability,
            device=tokens.device,
            dtype=scores.dtype,
        )
        scores = scores + attn_bias.unsqueeze(1)

        if modality_mask is not None:
            key_mask = modality_mask == 0  # (B, M)
            key_mask = key_mask.unsqueeze(1).expand(-1, grid_cells, -1)
            key_mask = key_mask.reshape(num_positions, 1, 1, self.num_modalities)
            scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        context = self._merge_heads(context)
        return self.out_proj(context)


class CrossModalAttentionBlock(nn.Module):
    """Pre-norm transformer block for cross-modal fusion at a single pyramid level.

    Architecture:
        T'  = T + Dropout(SelfAttn(LN(T)))
        T'' = T' + Dropout(MLP(LN(T')))

    Each spatial location holds a length-``M`` sequence of modality tokens.
    """

    def __init__(
        self,
        embed_dim: int = DEFAULT_FPN_OUT_CHANNELS,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_modalities: int = len(DEFAULT_MODALITIES),
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        dropout: float = DEFAULT_DROPOUT,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.modalities: tuple[str, ...] = tuple(modalities)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CrossModalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_modalities=num_modalities,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        modality_features: ModalityFeatures,
        modality_mask: Optional[torch.Tensor] = None,
        thermal_stability: Optional[torch.Tensor] = None,
        spatial_size: Optional[int] = None,
    ) -> ModalityFeatures:
        """Fuse modality features via cross-modal self-attention.

        Args:
            modality_features: Mapping from modality name to tensor
                ``(B, C, H, W)``.
            modality_mask: Optional mask ``(B, M)`` where ``1`` is valid and ``0``
                is missing.
            thermal_stability: Optional stability map ``(B, 1, H, W)`` in ``[0, 1]``.
            spatial_size: Expected ``H = W``; inferred from tensors if omitted.

        Returns:
            Updated modality feature mapping with unchanged shapes.
        """
        if spatial_size is None:
            sample = modality_features[self.modalities[0]]
            spatial_size = sample.shape[-1]

        batch_size = _validate_modality_features(
            modality_features,
            self.modalities,
            self.embed_dim,
            spatial_size,
            context="CrossModalAttentionBlock",
        )

        if modality_mask is not None:
            modality_mask = _validate_modality_mask(
                modality_mask,
                batch_size=batch_size,
                num_modalities=self.num_modalities,
                context="CrossModalAttentionBlock",
            )

        if thermal_stability is not None:
            thermal_stability = _validate_stability_map(
                thermal_stability,
                batch_size=batch_size,
                spatial_size=spatial_size,
                context="CrossModalAttentionBlock",
            )

        tokens = _stack_modalities(modality_features, self.modalities)
        batch, height, width, num_modalities, channels = tokens.shape
        flat = tokens.view(batch * height * width, num_modalities, channels)

        attn_out = self.attn(
            self.norm1(flat),
            spatial_size=spatial_size,
            modality_mask=modality_mask,
            thermal_stability=thermal_stability,
        )
        flat = flat + self.dropout(attn_out)

        mlp_out = self.mlp(self.norm2(flat))
        flat = flat + self.dropout(mlp_out)

        updated = flat.view(batch, height, width, num_modalities, channels)
        return _unstack_modalities(updated, self.modalities)


class CrossModalMultiScaleAttention(nn.Module):
    """Cross-Modal Multi-Scale Attention (CMMA) over F3, F4, and F5.

    Applies an independent ``CrossModalAttentionBlock`` at each coarse fused
    pyramid level. Fine levels (F1, F2) are intentionally excluded to preserve
    pixel-scale detail and limit compute.
    """

    def __init__(
        self,
        embed_dim: int = DEFAULT_FPN_OUT_CHANNELS,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        dropout: float = DEFAULT_DROPOUT,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.modalities: tuple[str, ...] = tuple(modalities)

        self.blocks = nn.ModuleDict(
            {
                level: CrossModalAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_modalities=len(self.modalities),
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    modalities=self.modalities,
                )
                for level in CMMA_LEVELS
            }
        )

    def forward(
        self,
        multi_scale_features: MultiScaleModalityFeatures,
        modality_masks: Optional[dict[str, torch.Tensor]] = None,
        thermal_stability_maps: Optional[dict[str, torch.Tensor]] = None,
    ) -> MultiScaleModalityFeatures:
        """Run CMMA at pyramid levels F3, F4, and F5.

        Args:
            multi_scale_features: Mapping ``level -> {modality -> tensor}`` for
                each ``F3``, ``F4``, ``F5`` level.  Every tensor must have shape
                ``(B, embed_dim, H_l, W_l)``.
            modality_masks: Optional per-level masks ``(B, M)`` keyed by ``F3``,
                ``F4``, or ``F5``.
            thermal_stability_maps: Optional per-level stability maps
                ``(B, 1, H_l, W_l)`` keyed by level.

        Returns:
            Mapping with the same structure as ``multi_scale_features`` containing
            attention-updated modality tensors.
        """
        if not multi_scale_features:
            raise ValueError("multi_scale_features must not be empty.")

        missing_levels = [
            level for level in CMMA_LEVELS if level not in multi_scale_features
        ]
        if missing_levels:
            raise ValueError(
                "CMMA requires pyramid levels "
                f"{', '.join(CMMA_LEVELS)}; missing: {', '.join(missing_levels)}."
            )

        outputs: MultiScaleModalityFeatures = {}
        for level_idx, level_name in enumerate(CMMA_LEVELS):
            spatial_size = CMMA_SPATIAL[level_idx]
            level_features = multi_scale_features[level_name]

            mask = None
            if modality_masks is not None:
                if level_name not in modality_masks:
                    raise ValueError(
                        f"modality_masks is missing an entry for level '{level_name}'."
                    )
                mask = modality_masks[level_name]

            stability = None
            if thermal_stability_maps is not None:
                if level_name not in thermal_stability_maps:
                    raise ValueError(
                        "thermal_stability_maps is missing an entry for level "
                        f"'{level_name}'."
                    )
                stability = thermal_stability_maps[level_name]

            outputs[level_name] = self.blocks[level_name](
                level_features,
                modality_mask=mask,
                thermal_stability=stability,
                spatial_size=spatial_size,
            )

        return outputs


__all__ = [
    "CMMA_LEVELS",
    "CMMA_SPATIAL",
    "DEFAULT_DROPOUT",
    "DEFAULT_MLP_RATIO",
    "DEFAULT_NUM_HEADS",
    "MODALITY_INDEX",
    "CrossModalAttentionBlock",
    "CrossModalMultiScaleAttention",
    "CrossModalSelfAttention",
    "ModalityFeatures",
    "MultiScaleModalityFeatures",
]
