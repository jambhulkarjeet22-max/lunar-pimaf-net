"""M3 spectral loader for LUNAR-PIMAF-Net.

Loads Moon Mineralogy Mapper reflectance and hydration indices, excludes
permanently shadowed regions, and exports aligned spectral feature stacks.

Reference: docs/DATA_PREPROCESSING_PIPELINE.md §3.6, §4.2 (channels 45–51)
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

M3_EXPORT_BANDS: Final[tuple[str, ...]] = (
    "m3_r750",
    "m3_r1500",
    "m3_bd_1250",
    "m3_bd_2800",
    "m3_water_index",
    "m3_slope_1um",
    "m3_slope_2um",
)


@dataclass
class M3LoaderConfig:
    """Configuration for M3 spectral ingestion."""

    r750_path: Path
    r1500_path: Path
    pole: Pole
    bd1250_path: Path | None = None
    bd2800_path: Path | None = None
    sunlit_path: Path | None = None
    psr_fraction: xr.DataArray | None = None
    psr_threshold: float = 0.5
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M
    output_dir: Path | None = None


@dataclass
class M3Loader:
    """Load and preprocess M3 spectral observables."""

    config: M3LoaderConfig
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
        band = resample_to_reference(band, self._reference)
        band.name = name
        return band

    def _psr_mask(self, template: xr.DataArray) -> xr.DataArray:
        if self.config.psr_fraction is not None:
            psr = resample_to_reference(self.config.psr_fraction, template)
            return psr >= self.config.psr_threshold
        if self.config.sunlit_path is not None:
            sunlit = self._align_band(self.config.sunlit_path, "m3_sunlit")
            return sunlit > 0.5
        logger.warning("No PSR or sunlit mask provided; M3 PSR exclusion disabled.")
        return xr.zeros_like(template, dtype=bool)

    @staticmethod
    def _band_depth(r750: xr.DataArray, r1500: xr.DataArray, center: float) -> xr.DataArray:
        values = np.asarray(r750.values, dtype=np.float64)
        ref = np.asarray(r1500.values, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(ref > 1e-12, 1.0 - values / ref, np.nan)
        return xr.DataArray(depth, coords=r750.coords, dims=r750.dims)

    @staticmethod
    def _water_index(bd2800: xr.DataArray, bd1250: xr.DataArray) -> xr.DataArray:
        values = np.asarray(bd2800.values, dtype=np.float64) - np.asarray(bd1250.values, dtype=np.float64)
        return xr.DataArray(values, coords=bd2800.coords, dims=bd2800.dims, name="m3_water_index")

    @staticmethod
    def _spectral_slope(r750: xr.DataArray, r1500: xr.DataArray, name: str) -> xr.DataArray:
        v750 = np.asarray(r750.values, dtype=np.float64)
        v1500 = np.asarray(r1500.values, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (v1500 - v750) / (1500.0 - 750.0)
        return xr.DataArray(slope, coords=r750.coords, dims=r750.dims, name=name)

    def load(self) -> None:
        """Validate M3 source paths."""
        for path in (self.config.r750_path, self.config.r1500_path):
            if not path.is_file():
                raise FileNotFoundError(f"M3 raster not found: {path}")

    def preprocess(self) -> dict[str, xr.DataArray]:
        """Load spectral bands, mask PSR pixels, and derive hydration indices."""
        if self._reference is None:
            raise RuntimeError("Call set_reference_grid() before preprocess().")

        r750 = self._align_band(self.config.r750_path, "m3_r750")
        r1500 = self._align_band(self.config.r1500_path, "m3_r1500")

        bd1250 = (
            self._align_band(self.config.bd1250_path, "m3_bd_1250")
            if self.config.bd1250_path is not None
            else self._band_depth(r750, r1500, 1250.0).rename("m3_bd_1250")
        )
        bd2800 = (
            self._align_band(self.config.bd2800_path, "m3_bd_2800")
            if self.config.bd2800_path is not None
            else self._band_depth(r750, r1500, 2800.0).rename("m3_bd_2800")
        )

        psr_mask = self._psr_mask(r750)
        r750 = r750.where(~psr_mask)
        r1500 = r1500.where(~psr_mask)
        bd1250 = bd1250.where(~psr_mask)
        bd2800 = bd2800.where(~psr_mask)

        water_index = self._water_index(bd2800, bd1250).rename("m3_water_index")
        slope_1um = self._spectral_slope(r750, r1500, "m3_slope_1um")
        slope_2um = self._spectral_slope(bd1250, bd2800, "m3_slope_2um")

        self._products = {
            "m3_r750": r750,
            "m3_r1500": r1500,
            "m3_bd_1250": bd1250,
            "m3_bd_2800": bd2800,
            "m3_water_index": water_index,
            "m3_slope_1um": slope_1um,
            "m3_slope_2um": slope_2um,
        }
        validate_crs(r750, int(self._reference.rio.crs.to_epsg()), context="M3 r750")
        return dict(self._products)

    def export(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Export M3 products as COGs."""
        if not self._products:
            self.preprocess()

        destination = output_dir or self.config.output_dir
        if destination is None:
            raise ValueError("output_dir must be provided via config or export().")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        written = {name: write_cog(array, destination / f"{name}.tif") for name, array in self._products.items()}
        save_metadata_json(
            {"instrument": "M3", "pole": self.config.pole, "bands": list(self._products)},
            destination / "m3_metadata.json",
        )
        return written


__all__ = ["M3_EXPORT_BANDS", "M3Loader", "M3LoaderConfig"]
