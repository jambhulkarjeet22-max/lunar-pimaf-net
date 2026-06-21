"""Lunar geospatial preprocessing pipeline for LUNAR-PIMAF-Net.

Orchestrates instrument loaders, builds the 59-channel fusion tensor, applies
group-wise normalization, extracts training patches, and exports Zarr datasets.

Reference: docs/DATA_PREPROCESSING_PIPELINE.md §4–§10
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

import numpy as np
import xarray as xr

from src.data.common import (
    DEFAULT_PIXEL_SIZE_M,
    Pole,
    attach_reference_grid,
    build_reference_grid,
    pole_to_epsg,
    save_metadata_json,
    valid_pixel_fraction,
    validate_crs,
)
from src.data.diviner_loader import DIVINER_EXPORT_BANDS, DivinerLoader, DivinerLoaderConfig
from src.data.lamp_loader import LAMP_EXPORT_BANDS, LAMPLoader, LAMPLoaderConfig
from src.data.lend_loader import LEND_EXPORT_BANDS, LENDLoader, LENDLoaderConfig
from src.data.lola_loader import LOLA_EXPORT_BANDS, LOLALoader, LOLALoaderConfig
from src.data.m3_loader import M3_EXPORT_BANDS, M3Loader, M3LoaderConfig
from src.data.mini_rf_loader import MINI_RF_EXPORT_BANDS, MiniRFLoader, MiniRFLoaderConfig

logger = logging.getLogger(__name__)

PATCH_SIZE: Final[int] = 128
NUM_FUSION_CHANNELS: Final[int] = 59
STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8
T_TRAP_K: Final[float] = 110.0

FUSION_CHANNELS: Final[tuple[str, ...]] = (
    *LOLA_EXPORT_BANDS,
    *MINI_RF_EXPORT_BANDS,
    *(band for band in DIVINER_EXPORT_BANDS if band != "div_cold_trap_mask"),
    *LAMP_EXPORT_BANDS,
    *LEND_EXPORT_BANDS,
    *M3_EXPORT_BANDS,
    "phys_stefan_residual",
    "phys_ice_retention_prob",
    "phys_radar_ice_likelihood",
    "phys_neutron_h_anomaly",
    "phys_uv_frost_likelihood",
    "phys_multi_sensor_agreement",
    "phys_subsurface_accessibility",
)

MODALITY_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "topography": tuple(LOLA_EXPORT_BANDS),
    "radar": tuple(MINI_RF_EXPORT_BANDS),
    "thermal": tuple(band for band in DIVINER_EXPORT_BANDS if band != "div_cold_trap_mask"),
    "uv": tuple(LAMP_EXPORT_BANDS),
    "neutron": tuple(LEND_EXPORT_BANDS),
    "spectral": tuple(M3_EXPORT_BANDS),
    "physics": (
        "phys_stefan_residual",
        "phys_ice_retention_prob",
        "phys_radar_ice_likelihood",
        "phys_neutron_h_anomaly",
        "phys_uv_frost_likelihood",
        "phys_multi_sensor_agreement",
        "phys_subsurface_accessibility",
    ),
}

ScalerMethod = Literal["robust", "standard", "minmax", "identity"]


@dataclass
class GroupScalerState:
    """Fitted normalization parameters for a feature group."""

    method: ScalerMethod
    center: list[float]
    scale: list[float]
    clip_min: list[float] | None = None
    clip_max: list[float] | None = None


@dataclass
class NormalizationState:
    """Full normalization state for the 59-channel fusion stack."""

    pole: Pole
    channel_names: tuple[str, ...]
    groups: dict[str, GroupScalerState]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "pole": self.pole,
            "channel_names": list(self.channel_names),
            "groups": {name: asdict(state) for name, state in self.groups.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NormalizationState:
        """Deserialize normalization state from JSON."""
        groups = {
            name: GroupScalerState(**state)
            for name, state in payload["groups"].items()
        }
        return cls(
            pole=payload["pole"],
            channel_names=tuple(payload["channel_names"]),
            groups=groups,
        )


@dataclass
class DatasetStatistics:
    """Per-channel dataset statistics for quality control."""

    channel_means: dict[str, float]
    channel_stds: dict[str, float]
    valid_fractions: dict[str, float]
    total_pixels: int


@dataclass
class QualityControlReport:
    """Quality-control summary for an ingested tile."""

    pole: Pole
    crs_valid: bool
    spatial_alignment_valid: bool
    channel_integrity_valid: bool
    missing_value_report: dict[str, float]
    statistics: DatasetStatistics
    warnings: list[str] = field(default_factory=list)


@dataclass
class LunarPreprocessingConfig:
    """Configuration for the end-to-end preprocessing pipeline."""

    pole: Pole
    lola: LOLALoaderConfig
    mini_rf: MiniRFLoaderConfig
    diviner: DivinerLoaderConfig
    lend: LENDLoaderConfig
    lamp: LAMPLoaderConfig
    m3: M3LoaderConfig
    patch_size: int = PATCH_SIZE
    output_dir: Path = Path("data/processed")
    normalization_method_overrides: dict[str, ScalerMethod] = field(default_factory=dict)


@dataclass
class LunarPreprocessingPipeline:
    """Production preprocessing pipeline for LUNAR-PIMAF-Net."""

    config: LunarPreprocessingConfig
    _reference: xr.DataArray | None = field(default=None, init=False, repr=False)
    _products: dict[str, xr.DataArray] = field(default_factory=dict, init=False, repr=False)
    _fusion_stack: xr.DataArray | None = field(default=None, init=False, repr=False)
    _validity_masks: dict[str, xr.DataArray] = field(default_factory=dict, init=False, repr=False)
    _normalization_state: NormalizationState | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(FUSION_CHANNELS) != NUM_FUSION_CHANNELS:
            raise ValueError(
                f"FUSION_CHANNELS length {len(FUSION_CHANNELS)} != "
                f"expected {NUM_FUSION_CHANNELS}."
            )

    @staticmethod
    def _default_group_method(group: str) -> ScalerMethod:
        mapping: dict[str, ScalerMethod] = {
            "topography": "robust",
            "radar": "standard",
            "thermal": "minmax",
            "uv": "standard",
            "neutron": "standard",
            "spectral": "standard",
            "physics": "identity",
        }
        return mapping[group]

    def _instantiate_loaders(
        self,
    ) -> tuple[LOLALoader, MiniRFLoader, DivinerLoader, LENDLoader, LAMPLoader, M3Loader]:
        lola = LOLALoader(self.config.lola)
        mini_rf = MiniRFLoader(self.config.mini_rf)
        diviner = DivinerLoader(self.config.diviner)
        lend = LENDLoader(self.config.lend)
        lamp = LAMPLoader(self.config.lamp)
        m3 = M3Loader(self.config.m3)
        return lola, mini_rf, diviner, lend, lamp, m3

    def load_all_datasets(self) -> dict[str, xr.DataArray]:
        """Load and align all instrument products on the LOLA reference grid."""
        lola, mini_rf, diviner, lend, lamp, m3 = self._instantiate_loaders()

        lola.load()
        lola_products = lola.preprocess()
        reference = lola.reference_elevation
        self._reference = reference
        validate_crs(reference, pole_to_epsg(self.config.pole), context="LOLA reference")

        mini_rf.set_reference_grid(reference)
        mini_rf.config.lola_slope = lola_products["lola_slope_deg"]
        mini_rf.load()
        mini_rf_products = mini_rf.preprocess()

        for loader in (diviner, lend, lamp):
            loader.set_reference_grid(reference)
            loader.load()

        m3.set_reference_grid(reference)
        m3.config.psr_fraction = lola_products["lola_psr_fraction"]
        m3.load()

        diviner_products = diviner.preprocess()
        lend_products = lend.preprocess()
        lamp_products = lamp.preprocess()
        m3_products = m3.preprocess()

        products: dict[str, xr.DataArray] = {}
        products.update(lola_products)
        products.update(mini_rf_products)
        products.update({k: v for k, v in diviner_products.items() if k != "div_cold_trap_mask"})
        products.update(lend_products)
        products.update(lamp_products)
        products.update(m3_products)
        products.update(self._compute_physics_channels(products))

        self._products = products
        logger.info("Loaded %d aligned product layers.", len(products))
        return dict(products)

    def _compute_physics_channels(self, products: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
        """Derive physics-informed channels 52–58 from aligned products."""
        tmax = products["div_tbol_max"]
        emissivity = products["div_emissivity_ch7"].fillna(0.95)
        insolation = products["div_insolation"]
        albedo = products["lamp_albedo_ratio"].fillna(0.1)
        cpr = products["mrf_cpr_rough_corrected"]
        slope = products["lola_slope_deg"]
        hydrogen = products["lend_h_wt_pct"]
        psr = products["lola_psr_fraction"]
        h2o_depth = products["lamp_h2o_depth"]
        permafrost = products["div_permafrost_depth_m"]
        ice_stability = products["div_ice_stability"]

        t_vals = np.asarray(tmax.values, dtype=np.float64)
        predicted = np.power(t_vals, 4)
        expected = (1.0 - np.asarray(albedo.values)) * np.asarray(insolation.values) / (
            np.asarray(emissivity.values) * STEFAN_BOLTZMANN + 1e-8
        )
        stefan_residual = np.abs(predicted - expected)
        stefan_residual = stefan_residual / (np.abs(expected) + 1.0)

        ice_retention = 1.0 / (1.0 + np.exp((t_vals - T_TRAP_K) / 5.0))
        radar_likelihood = np.clip(np.asarray(cpr.values) / 1.5, 0.0, 1.0) * (
            np.asarray(slope.values) < 10.0
        ).astype(np.float64)

        psr_mask = np.asarray(psr.values) >= 0.5
        h_vals = np.asarray(hydrogen.values, dtype=np.float64)
        if np.any(psr_mask & np.isfinite(h_vals)):
            psr_h = h_vals[psr_mask & np.isfinite(h_vals)]
            mean_h = float(np.mean(psr_h))
            std_h = float(np.std(psr_h) + 1e-6)
            neutron_anomaly = (h_vals - mean_h) / std_h
        else:
            neutron_anomaly = np.zeros_like(h_vals)

        h2o_vals = np.asarray(h2o_depth.values, dtype=np.float64)
        if np.any(psr_mask & np.isfinite(h2o_vals)):
            psr_h2o = h2o_vals[psr_mask & np.isfinite(h2o_vals)]
            uv_frost = (h2o_vals - float(np.min(psr_h2o))) / (
                float(np.max(psr_h2o) - np.min(psr_h2o) + 1e-6)
            )
        else:
            uv_frost = np.zeros_like(h2o_vals)

        agreement = (
            (ice_stability.values >= 0.8).astype(np.float64)
            + (products["lend_neutron_suppression"].values >= 0.7).astype(np.float64)
            + (np.asarray(cpr.values) >= 1.2).astype(np.float64)
            + (h2o_vals >= np.nanpercentile(h2o_vals[np.isfinite(h2o_vals)], 80)).astype(np.float64)
            + (products["m3_bd_2800"].values >= np.nanpercentile(
                products["m3_bd_2800"].values[np.isfinite(products["m3_bd_2800"].values)],
                80,
            )).astype(np.float64)
        ) / 5.0

        subsurface_access = np.asarray(psr.values) * np.asarray(permafrost.values)

        template = tmax
        return {
            "phys_stefan_residual": xr.DataArray(stefan_residual, coords=template.coords, dims=template.dims),
            "phys_ice_retention_prob": xr.DataArray(ice_retention, coords=template.coords, dims=template.dims),
            "phys_radar_ice_likelihood": xr.DataArray(radar_likelihood, coords=template.coords, dims=template.dims),
            "phys_neutron_h_anomaly": xr.DataArray(neutron_anomaly, coords=template.coords, dims=template.dims),
            "phys_uv_frost_likelihood": xr.DataArray(uv_frost, coords=template.coords, dims=template.dims),
            "phys_multi_sensor_agreement": xr.DataArray(agreement, coords=template.coords, dims=template.dims),
            "phys_subsurface_accessibility": xr.DataArray(
                subsurface_access,
                coords=template.coords,
                dims=template.dims,
            ),
        }

    def build_fusion_stack(self) -> xr.DataArray:
        """Stack aligned products into a 59-channel ``(band, y, x)`` DataArray."""
        if not self._products:
            self.load_all_datasets()

        missing = [name for name in FUSION_CHANNELS if name not in self._products]
        if missing:
            raise ValueError(f"Missing products required for fusion stack: {', '.join(missing)}.")

        arrays = [self._products[name].astype("float32") for name in FUSION_CHANNELS]
        stack = xr.concat(arrays, dim="band")
        stack = stack.assign_coords(band=list(FUSION_CHANNELS))
        self._fusion_stack = stack
        return stack

    def generate_validity_masks(self) -> dict[str, xr.DataArray]:
        """Generate per-modality validity masks from finite-value checks."""
        if self._fusion_stack is None:
            self.build_fusion_stack()

        assert self._fusion_stack is not None
        masks: dict[str, xr.DataArray] = {}
        for group, channels in MODALITY_GROUPS.items():
            group_stack = self._fusion_stack.sel(band=list(channels))
            valid = np.isfinite(group_stack.values).any(axis=0).astype(np.float32)
            masks[group] = xr.DataArray(
                valid,
                coords={"y": self._fusion_stack.coords["y"], "x": self._fusion_stack.coords["x"]},
                dims=("y", "x"),
                name=f"{group}_valid",
            )
        self._validity_masks = masks
        return dict(masks)

    @staticmethod
    def _fit_group_scaler(values: np.ndarray, method: ScalerMethod) -> GroupScalerState:
        if values.ndim != 2:
            raise ValueError("Expected values with shape (n_samples, n_features).")

        centers: list[float] = []
        scales: list[float] = []
        clip_min: list[float] | None = [] if method == "minmax" else None
        clip_max: list[float] | None = [] if method == "minmax" else None

        for feature_idx in range(values.shape[1]):
            column = values[:, feature_idx]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                centers.append(0.0)
                scales.append(1.0)
                if clip_min is not None and clip_max is not None:
                    clip_min.append(0.0)
                    clip_max.append(1.0)
                continue

            if method == "robust":
                center = float(np.median(finite))
                q75, q25 = np.percentile(finite, [75, 25])
                scale = float(q75 - q25) or 1.0
            elif method == "standard":
                center = float(np.mean(finite))
                scale = float(np.std(finite)) or 1.0
            elif method == "minmax":
                min_val = float(np.min(finite))
                max_val = float(np.max(finite))
                center = min_val
                scale = (max_val - min_val) or 1.0
                if clip_min is not None and clip_max is not None:
                    clip_min.append(min_val)
                    clip_max.append(max_val)
            else:
                center = 0.0
                scale = 1.0

            centers.append(center)
            scales.append(scale)

        return GroupScalerState(
            method=method,
            center=centers,
            scale=scales,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    @staticmethod
    def _apply_group_scaler(values: np.ndarray, state: GroupScalerState) -> np.ndarray:
        transformed = np.empty_like(values, dtype=np.float64)
        for idx in range(values.shape[1]):
            column = values[:, idx]
            if state.method == "minmax":
                transformed[:, idx] = (column - state.center[idx]) / state.scale[idx]
            elif state.method == "identity":
                transformed[:, idx] = column
            else:
                transformed[:, idx] = (column - state.center[idx]) / state.scale[idx]
        return transformed

    def fit(self, sample_mask: xr.DataArray | None = None) -> NormalizationState:
        """Fit group-wise normalization parameters on training pixels."""
        if self._fusion_stack is None:
            self.build_fusion_stack()

        assert self._fusion_stack is not None
        stack = self._fusion_stack
        if sample_mask is not None:
            mask = np.asarray(sample_mask.values, dtype=bool)
        else:
            mask = np.ones(stack.shape[1:], dtype=bool)

        groups: dict[str, GroupScalerState] = {}
        for group, channels in MODALITY_GROUPS.items():
            method = self.config.normalization_method_overrides.get(
                group,
                self._default_group_method(group),
            )
            group_data = stack.sel(band=list(channels)).values.reshape(len(channels), -1).T
            group_data = group_data[mask.ravel()]
            groups[group] = self._fit_group_scaler(group_data, method)

        self._normalization_state = NormalizationState(
            pole=self.config.pole,
            channel_names=FUSION_CHANNELS,
            groups=groups,
        )
        self._fitted = True
        return self._normalization_state

    def transform(self, stack: xr.DataArray | None = None) -> xr.DataArray:
        """Apply fitted normalization to the fusion stack."""
        if self._normalization_state is None:
            raise RuntimeError("Call fit() before transform().")

        data = stack if stack is not None else self._fusion_stack
        if data is None:
            raise RuntimeError("No fusion stack available for transform().")

        normalized_layers: list[xr.DataArray] = []
        for group, channels in MODALITY_GROUPS.items():
            group_stack = data.sel(band=list(channels))
            values = group_stack.values.reshape(len(channels), -1).T
            transformed = self._apply_group_scaler(values, self._normalization_state.groups[group])
            transformed = transformed.T.reshape(len(channels), *group_stack.shape[1:])
            normalized_layers.append(
                xr.DataArray(
                    transformed,
                    coords=group_stack.coords,
                    dims=group_stack.dims,
                )
            )

        result = xr.concat(normalized_layers, dim="band")
        ordered_channels = [ch for group in MODALITY_GROUPS for ch in MODALITY_GROUPS[group]]
        result = result.assign_coords(band=ordered_channels)
        return result

    def fit_transform(self, sample_mask: xr.DataArray | None = None) -> xr.DataArray:
        """Fit normalization and return the transformed fusion stack."""
        self.fit(sample_mask=sample_mask)
        return self.transform()

    def extract_patches(
        self,
        stack: xr.DataArray | None = None,
        stride: int | None = None,
        max_patches: int | None = None,
    ) -> np.ndarray:
        """Extract non-overlapping or strided ``128×128`` patches.

        Args:
            stack: Normalized fusion stack. Uses the internal stack if omitted.
            stride: Patch stride in pixels. Defaults to ``patch_size``.
            max_patches: Optional cap on number of returned patches.

        Returns:
            Array of shape ``(N, 59, 128, 128)``.
        """
        data = stack if stack is not None else self._fusion_stack
        if data is None:
            raise RuntimeError("No fusion stack available for patch extraction.")

        patch_size = self.config.patch_size
        stride = stride or patch_size
        values = np.asarray(data.values, dtype=np.float32)
        bands, height, width = values.shape

        if height < patch_size or width < patch_size:
            raise ValueError(
                f"Stack spatial size ({height}, {width}) is smaller than "
                f"patch size {patch_size}."
            )

        patches: list[np.ndarray] = []
        for row in range(0, height - patch_size + 1, stride):
            for col in range(0, width - patch_size + 1, stride):
                patch = values[:, row : row + patch_size, col : col + patch_size]
                if np.isfinite(patch).mean() < 0.7:
                    continue
                patches.append(patch)
                if max_patches is not None and len(patches) >= max_patches:
                    return np.stack(patches, axis=0)

        if not patches:
            raise ValueError("No valid patches extracted from fusion stack.")

        return np.stack(patches, axis=0)

    def export_patches_zarr(
        self,
        patches: np.ndarray,
        output_path: Path | str,
        chunk_size: int = 32,
    ) -> Path:
        """Export extracted patches to Zarr format.

        Args:
            patches: Patch array ``(N, 59, 128, 128)``.
            output_path: Zarr store path.
            chunk_size: Zarr chunk size for spatial dimensions.

        Returns:
            Resolved Zarr path.
        """
        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "zarr is required for patch export. Install with: pip install zarr"
            ) from exc

        if patches.ndim != 4 or patches.shape[1:] != (NUM_FUSION_CHANNELS, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f"Expected patches shape (N, {NUM_FUSION_CHANNELS}, "
                f"{PATCH_SIZE}, {PATCH_SIZE}), got {patches.shape}."
            )

        resolved = Path(output_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(resolved), mode="w")
        root.create_dataset(
            "X",
            data=patches,
            chunks=(1, NUM_FUSION_CHANNELS, chunk_size, chunk_size),
            dtype="float32",
        )
        root.attrs["channels"] = list(FUSION_CHANNELS)
        root.attrs["patch_size"] = PATCH_SIZE
        logger.info("Exported %d patches to Zarr: %s", patches.shape[0], resolved)
        return resolved

    def compute_statistics(self, stack: xr.DataArray | None = None) -> DatasetStatistics:
        """Compute per-channel dataset statistics."""
        data = stack if stack is not None else self._fusion_stack
        if data is None:
            raise RuntimeError("No fusion stack available.")

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        valid_fractions: dict[str, float] = {}
        for channel in FUSION_CHANNELS:
            band = data.sel(band=channel)
            values = np.asarray(band.values, dtype=np.float64)
            finite = values[np.isfinite(values)]
            means[channel] = float(np.mean(finite)) if finite.size else float("nan")
            stds[channel] = float(np.std(finite)) if finite.size else float("nan")
            valid_fractions[channel] = valid_pixel_fraction(band)

        total_pixels = int(data.shape[1] * data.shape[2])
        return DatasetStatistics(
            channel_means=means,
            channel_stds=stds,
            valid_fractions=valid_fractions,
            total_pixels=total_pixels,
        )

    def run_quality_control(self) -> QualityControlReport:
        """Run CRS, alignment, channel integrity, and missing-value checks."""
        warnings: list[str] = []
        if self._reference is None or self._fusion_stack is None:
            self.load_all_datasets()
            self.build_fusion_stack()
            self.generate_validity_masks()

        assert self._reference is not None
        assert self._fusion_stack is not None

        crs_valid = True
        try:
            validate_crs(self._reference, pole_to_epsg(self.config.pole), context="reference")
        except ValueError as exc:
            crs_valid = False
            warnings.append(str(exc))

        spatial_alignment_valid = True
        reference_shape = tuple(self._reference.shape)
        for channel in FUSION_CHANNELS:
            if tuple(self._products[channel].shape) != reference_shape:
                spatial_alignment_valid = False
                warnings.append(
                    f"Channel '{channel}' shape {tuple(self._products[channel].shape)} "
                    f"!= reference {reference_shape}."
                )

        missing = [name for name in FUSION_CHANNELS if name not in self._products]
        channel_integrity_valid = len(missing) == 0
        if missing:
            warnings.append(f"Missing channels: {', '.join(missing)}")

        missing_report = {
            channel: 1.0 - valid_pixel_fraction(self._products[channel])
            for channel in FUSION_CHANNELS
            if channel in self._products
        }

        statistics = self.compute_statistics()
        return QualityControlReport(
            pole=self.config.pole,
            crs_valid=crs_valid,
            spatial_alignment_valid=spatial_alignment_valid,
            channel_integrity_valid=channel_integrity_valid,
            missing_value_report=missing_report,
            statistics=statistics,
            warnings=warnings,
        )

    def save(self, directory: Path | str) -> Path:
        """Persist normalization state and pipeline metadata."""
        resolved = Path(directory)
        resolved.mkdir(parents=True, exist_ok=True)

        if self._normalization_state is not None:
            save_metadata_json(self._normalization_state.to_dict(), resolved / "normalization_state.json")

        save_metadata_json(
            {
                "pole": self.config.pole,
                "patch_size": self.config.patch_size,
                "channels": list(FUSION_CHANNELS),
                "modality_groups": {k: list(v) for k, v in MODALITY_GROUPS.items()},
            },
            resolved / "pipeline_metadata.json",
        )
        return resolved

    @classmethod
    def load(cls, directory: Path | str, config: LunarPreprocessingConfig) -> LunarPreprocessingPipeline:
        """Load a saved pipeline state into a new instance."""
        resolved = Path(directory)
        pipeline = cls(config=config)
        norm_path = resolved / "normalization_state.json"
        if norm_path.is_file():
            with norm_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            pipeline._normalization_state = NormalizationState.from_dict(payload)
            pipeline._fitted = True
        return pipeline


__all__ = [
    "FUSION_CHANNELS",
    "LunarPreprocessingConfig",
    "LunarPreprocessingPipeline",
    "MODALITY_GROUPS",
    "NUM_FUSION_CHANNELS",
    "NormalizationState",
    "PATCH_SIZE",
    "QualityControlReport",
    "DatasetStatistics",
]
