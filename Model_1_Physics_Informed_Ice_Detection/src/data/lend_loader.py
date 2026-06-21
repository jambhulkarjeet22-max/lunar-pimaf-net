"""LEND neutron loader for LUNAR-PIMAF-Net.

Loads coarse LEND epithermal products, derives hydrogen abundance and neutron
suppression metrics, and upsamples to the LOLA reference grid.

Reference: docs/DATA_PREPROCESSING_PIPELINE.md §3.5, §4.2 (channels 39–44)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr
from rasterio.enums import Resampling

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

LEND_EXPORT_BANDS: Final[tuple[str, ...]] = (
    "lend_epi_counts",
    "lend_epi_collimated",
    "lend_fast_counts",
    "lend_h_wt_pct",
    "lend_neutron_suppression",
    "lend_h_gradient",
)


@dataclass
class LENDLoaderConfig:
    """Configuration for LEND neutron ingestion."""

    epithermal_path: Path
    pole: Pole
    collimated_path: Path | None = None
    fast_path: Path | None = None
    hydrogen_path: Path | None = None
    confidence_path: Path | None = None
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M
    min_confidence: float = 0.3
    upsampling: Resampling = Resampling.cubic
    output_dir: Path | None = None


@dataclass
class LENDLoader:
    """Load and preprocess LEND neutron observables with coarse-to-fine resampling."""

    config: LENDLoaderConfig
    _reference: xr.DataArray | None = field(default=None, init=False, repr=False)
    _products: dict[str, xr.DataArray] = field(default_factory=dict, init=False, repr=False)

    def set_reference_grid(self, reference: xr.DataArray) -> None:
        """Attach the LOLA reference grid."""
        if reference.ndim != 2:
            raise ValueError("reference grid must be 2-D.")
        self._reference = reference

    def _align_band(self, path: Path, name: str) -> xr.DataArray:
        if self._reference is None:
            raise RuntimeError("Reference grid not set.")
        band = read_geotiff(path)
        band = normalize_nodata(band)
        band = reproject_to_polar_stereographic(
            band,
            pole=self.config.pole,
            resolution_m=self.config.pixel_size_m,
        )
        band = resample_to_reference(band, self._reference, resampling=self.config.upsampling)
        band.name = name
        return band

    @staticmethod
    def _neutron_suppression(epithermal: xr.DataArray) -> xr.DataArray:
        values = np.asarray(epithermal.values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            suppression = np.zeros_like(values)
        else:
            median = float(np.median(finite))
            with np.errstate(divide="ignore", invalid="ignore"):
                suppression = np.clip(1.0 - values / (median + 1e-6), 0.0, 1.0)
        return xr.DataArray(
            suppression,
            coords=epithermal.coords,
            dims=epithermal.dims,
            name="lend_neutron_suppression",
        )

    @staticmethod
    def _hydrogen_gradient(hydrogen: xr.DataArray, pixel_size_m: float) -> xr.DataArray:
        values = np.asarray(hydrogen.values, dtype=np.float64)
        grad_y, grad_x = np.gradient(values, pixel_size_m / 1000.0)
        gradient = np.hypot(grad_x, grad_y)
        return xr.DataArray(
            gradient,
            coords=hydrogen.coords,
            dims=hydrogen.dims,
            name="lend_h_gradient",
        )

    def load(self) -> None:
        """Validate LEND source paths."""
        if not self.config.epithermal_path.is_file():
            raise FileNotFoundError(f"LEND epithermal raster not found: {self.config.epithermal_path}")

    def preprocess(self) -> dict[str, xr.DataArray]:
        """Upsample LEND products and derive hydrogen metrics."""
        if self._reference is None:
            raise RuntimeError("Call set_reference_grid() before preprocess().")

        epithermal = self._align_band(self.config.epithermal_path, "lend_epi_counts")
        collimated = (
            self._align_band(self.config.collimated_path, "lend_epi_collimated")
            if self.config.collimated_path is not None
            else epithermal.copy()
        )
        fast = (
            self._align_band(self.config.fast_path, "lend_fast_counts")
            if self.config.fast_path is not None
            else xr.full_like(epithermal, np.nan)
        )

        if self.config.confidence_path is not None:
            confidence = self._align_band(self.config.confidence_path, "lend_confidence")
            epithermal = epithermal.where(confidence >= self.config.min_confidence)
            collimated = collimated.where(confidence >= self.config.min_confidence)

        if self.config.hydrogen_path is not None:
            hydrogen = self._align_band(self.config.hydrogen_path, "lend_h_wt_pct")
        else:
            hydrogen = self._neutron_suppression(epithermal).rename("lend_h_wt_pct") * 10.0

        suppression = self._neutron_suppression(epithermal)
        gradient = self._hydrogen_gradient(hydrogen.fillna(0.0), self.config.pixel_size_m)

        self._products = {
            "lend_epi_counts": epithermal,
            "lend_epi_collimated": collimated,
            "lend_fast_counts": fast,
            "lend_h_wt_pct": hydrogen,
            "lend_neutron_suppression": suppression,
            "lend_h_gradient": gradient,
        }
        validate_crs(epithermal, int(self._reference.rio.crs.to_epsg()), context="LEND epithermal")
        return dict(self._products)

    def export(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Export LEND products as COGs."""
        if not self._products:
            self.preprocess()

        destination = output_dir or self.config.output_dir
        if destination is None:
            raise ValueError("output_dir must be provided via config or export().")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        written = {name: write_cog(array, destination / f"{name}.tif") for name, array in self._products.items()}
        save_metadata_json(
            {"instrument": "LEND", "pole": self.config.pole, "bands": list(self._products)},
            destination / "lend_metadata.json",
        )
        return written


__all__ = ["LEND_EXPORT_BANDS", "LENDLoader", "LENDLoaderConfig"]
