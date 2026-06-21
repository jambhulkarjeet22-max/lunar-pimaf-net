"""Mini-RF radar loader for LUNAR-PIMAF-Net.

Loads Mini-SAR / Mini-RF polar products, applies slope-aware CPR correction using
LOLA terrain, and exports aligned radar feature stacks.

Reference: docs/DATA_PREPROCESSING_PIPELINE.md §3.2, §4.2 (channels 8–17)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr

from src.data.common import (
    DEFAULT_PIXEL_SIZE_M,
    Pole,
    normalize_nodata,
    read_geotiff,
    reproject_to_polar_stereographic,
    resample_to_reference,
    save_metadata_json,
    validate_crs,
    write_cog,
)

logger = logging.getLogger(__name__)

MINI_RF_EXPORT_BANDS: Final[tuple[str, ...]] = (
    "mrf_cpr",
    "mrf_cpr_rough_corrected",
    "mrf_sc_db",
    "mrf_oc_db",
    "mrf_s0",
    "mrf_s1_norm",
    "mrf_s2_norm",
    "mrf_s3_norm",
    "mrf_mchi_odd",
    "mrf_mchi_vol",
)


@dataclass
class MiniRFLoaderConfig:
    """Configuration for Mini-RF ingestion."""

    cpr_path: Path
    sc_path: Path
    oc_path: Path
    pole: Pole
    s0_path: Path | None = None
    s1_path: Path | None = None
    s2_path: Path | None = None
    s3_path: Path | None = None
    mchi_odd_path: Path | None = None
    mchi_vol_path: Path | None = None
    slope_path: Path | None = None
    lola_slope: xr.DataArray | None = None
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M
    max_slope_correction_deg: float = 45.0
    output_dir: Path | None = None


@dataclass
class MiniRFLoader:
    """Load and preprocess Mini-RF radar observables."""

    config: MiniRFLoaderConfig
    _reference: xr.DataArray | None = field(default=None, init=False, repr=False)
    _products: dict[str, xr.DataArray] = field(default_factory=dict, init=False, repr=False)

    def set_reference_grid(self, reference: xr.DataArray) -> None:
        """Attach the LOLA reference grid for resampling."""
        if reference.ndim != 2:
            raise ValueError("reference grid must be 2-D.")
        self._reference = reference

    def _align_band(self, path: Path, name: str) -> xr.DataArray:
        if self._reference is None:
            raise RuntimeError("Reference grid not set. Call set_reference_grid() first.")
        band = read_geotiff(path)
        band = normalize_nodata(band)
        band = reproject_to_polar_stereographic(
            band,
            pole=self.config.pole,
            resolution_m=self.config.pixel_size_m,
        )
        band = resample_to_reference(band, self._reference)
        band.name = name
        return band

    @staticmethod
    def _safe_ratio(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
        num = np.asarray(numerator.values, dtype=np.float64)
        den = np.asarray(denominator.values, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(den) > 1e-12, num / den, np.nan)
        return xr.DataArray(ratio, coords=numerator.coords, dims=numerator.dims)

    @staticmethod
    def _normalize_stokes(component: xr.DataArray, total: xr.DataArray) -> xr.DataArray:
        normalized = MiniRFLoader._safe_ratio(component, total)
        normalized = normalized.clip(-1.0, 1.0)
        return normalized

    def _resolve_slope(self) -> xr.DataArray:
        if self.config.lola_slope is not None:
            return self.config.lola_slope
        if self.config.slope_path is not None:
            if self._reference is None:
                raise RuntimeError("Reference grid required to align slope raster.")
            return self._align_band(self.config.slope_path, "lola_slope_deg")
        raise ValueError(
            "Mini-RF slope correction requires lola_slope DataArray or slope_path."
        )

    @staticmethod
    def _correct_cpr_for_roughness(
        cpr: xr.DataArray,
        slope_deg: xr.DataArray,
        max_slope_deg: float,
    ) -> xr.DataArray:
        slope = np.asarray(slope_deg.values, dtype=np.float64)
        cpr_values = np.asarray(cpr.values, dtype=np.float64)
        slope_clipped = np.clip(slope, 0.0, max_slope_deg)
        correction = np.cos(np.radians(slope_clipped))
        with np.errstate(divide="ignore", invalid="ignore"):
            corrected = np.where(correction > 1e-3, cpr_values / correction, np.nan)
        corrected = np.clip(corrected, 0.0, 3.0)
        return xr.DataArray(
            corrected,
            coords=cpr.coords,
            dims=cpr.dims,
            name="mrf_cpr_rough_corrected",
        )

    def load(self) -> None:
        """Validate source paths and ensure reference grid availability."""
        required = (self.config.cpr_path, self.config.sc_path, self.config.oc_path)
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Mini-RF input not found: {path}")
        if self._reference is None:
            logger.warning(
                "MiniRFLoader.load(): reference grid not set; call set_reference_grid() "
                "before preprocess()."
            )

    def preprocess(self) -> dict[str, xr.DataArray]:
        """Load radar bands, apply slope correction, and normalize Stokes terms.

        Returns:
            Mapping of Mini-RF feature names to aligned rasters.
        """
        if self._reference is None:
            raise RuntimeError("Call set_reference_grid() before preprocess().")

        cpr = self._align_band(self.config.cpr_path, "mrf_cpr")
        sc = self._align_band(self.config.sc_path, "mrf_sc_db")
        oc = self._align_band(self.config.oc_path, "mrf_oc_db")

        slope = self._resolve_slope()
        cpr_corrected = self._correct_cpr_for_roughness(
            cpr,
            slope,
            self.config.max_slope_correction_deg,
        )

        s0 = (
            self._align_band(self.config.s0_path, "mrf_s0")
            if self.config.s0_path is not None
            else xr.full_like(cpr, np.nan)
        )
        s0 = s0.where(np.isfinite(s0), other=sc + oc)

        s1 = (
            self._align_band(self.config.s1_path, "mrf_s1_norm")
            if self.config.s1_path is not None
            else xr.zeros_like(cpr)
        )
        s2 = (
            self._align_band(self.config.s2_path, "mrf_s2_norm")
            if self.config.s2_path is not None
            else xr.zeros_like(cpr)
        )
        s3 = (
            self._align_band(self.config.s3_path, "mrf_s3_norm")
            if self.config.s3_path is not None
            else xr.zeros_like(cpr)
        )

        s1_norm = self._normalize_stokes(s1, s0).rename("mrf_s1_norm")
        s2_norm = self._normalize_stokes(s2, s0).rename("mrf_s2_norm")
        s3_norm = self._normalize_stokes(s3, s0).rename("mrf_s3_norm")

        mchi_odd = (
            self._align_band(self.config.mchi_odd_path, "mrf_mchi_odd")
            if self.config.mchi_odd_path is not None
            else xr.zeros_like(cpr)
        ).clip(0.0, 1.0)
        mchi_vol = (
            self._align_band(self.config.mchi_vol_path, "mrf_mchi_vol")
            if self.config.mchi_vol_path is not None
            else xr.zeros_like(cpr)
        ).clip(0.0, 1.0)

        self._products = {
            "mrf_cpr": cpr,
            "mrf_cpr_rough_corrected": cpr_corrected,
            "mrf_sc_db": sc,
            "mrf_oc_db": oc,
            "mrf_s0": s0,
            "mrf_s1_norm": s1_norm,
            "mrf_s2_norm": s2_norm,
            "mrf_s3_norm": s3_norm,
            "mrf_mchi_odd": mchi_odd,
            "mrf_mchi_vol": mchi_vol,
        }
        validate_crs(cpr, int(self._reference.rio.crs.to_epsg()), context="Mini-RF CPR")
        return dict(self._products)

    def export(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Export Mini-RF products as COGs."""
        if not self._products:
            self.preprocess()

        destination = output_dir or self.config.output_dir
        if destination is None:
            raise ValueError("output_dir must be provided via config or export().")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        written = {name: write_cog(array, destination / f"{name}.tif") for name, array in self._products.items()}
        save_metadata_json(
            {"instrument": "Mini-RF", "pole": self.config.pole, "bands": list(self._products)},
            destination / "mini_rf_metadata.json",
        )
        return written


__all__ = ["MINI_RF_EXPORT_BANDS", "MiniRFLoader", "MiniRFLoaderConfig"]
