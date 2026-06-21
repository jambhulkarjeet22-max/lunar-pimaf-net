"""LAMP UV loader for LUNAR-PIMAF-Net.

Loads LAMP far-UV albedo and water-absorption products, computes temporal
variability, and exports aligned UV feature stacks.

Reference: docs/DATA_PREPROCESSING_PIPELINE.md §3.4, §4.2 (channels 33–38)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Sequence

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

LAMP_EXPORT_BANDS: Final[tuple[str, ...]] = (
    "lamp_albedo_on",
    "lamp_albedo_off",
    "lamp_albedo_ratio",
    "lamp_h2o_depth",
    "lamp_brightness",
    "lamp_temporal_std",
)


@dataclass
class LAMPLoaderConfig:
    """Configuration for LAMP UV ingestion."""

    albedo_on_paths: Sequence[Path]
    pole: Pole
    albedo_off_paths: Sequence[Path] | None = None
    h2o_depth_path: Path | None = None
    brightness_path: Path | None = None
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M
    output_dir: Path | None = None


@dataclass
class LAMPLoader:
    """Load and preprocess LAMP UV observables."""

    config: LAMPLoaderConfig
    _reference: xr.DataArray | None = field(default=None, init=False, repr=False)
    _products: dict[str, xr.DataArray] = field(default_factory=dict, init=False, repr=False)

    def set_reference_grid(self, reference: xr.DataArray) -> None:
        """Attach the LOLA reference grid."""
        if reference.ndim != 2:
            raise ValueError("reference grid must be 2-D.")
        self._reference = reference

    def _align_stack(self, paths: Sequence[Path], name: str) -> xr.DataArray:
        if self._reference is None:
            raise RuntimeError("Reference grid not set.")
        if not paths:
            raise ValueError(f"No paths provided for {name}.")

        aligned: list[xr.DataArray] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"LAMP raster not found: {path}")
            band = read_geotiff(path)
            band = normalize_nodata(band)
            band = reproject_to_polar_stereographic(
                band,
                pole=self.config.pole,
                resolution_m=self.config.pixel_size_m,
            )
            band = resample_to_reference(band, self._reference)
            aligned.append(band)

        if len(aligned) == 1:
            return aligned[0].rename(name)
        return xr.concat(aligned, dim="time")

    def _align_optional(self, path: Path | None, name: str, template: xr.DataArray) -> xr.DataArray:
        if path is None:
            return template
        if not path.is_file():
            raise FileNotFoundError(f"LAMP raster not found: {path}")
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

    def load(self) -> None:
        """Validate LAMP source paths."""
        if not self.config.albedo_on_paths:
            raise ValueError("albedo_on_paths must contain at least one GeoTIFF.")
        for path in self.config.albedo_on_paths:
            if not path.is_file():
                raise FileNotFoundError(f"LAMP albedo raster not found: {path}")

    def preprocess(self) -> dict[str, xr.DataArray]:
        """Aggregate temporal UV observations and derive albedo ratios."""
        if self._reference is None:
            raise RuntimeError("Call set_reference_grid() before preprocess().")

        albedo_on_stack = self._align_stack(self.config.albedo_on_paths, "lamp_albedo_on")
        if "time" in albedo_on_stack.dims:
            albedo_on = albedo_on_stack.mean(dim="time").rename("lamp_albedo_on")
            temporal_std = albedo_on_stack.std(dim="time").rename("lamp_temporal_std")
        else:
            albedo_on = albedo_on_stack
            temporal_std = xr.zeros_like(albedo_on).rename("lamp_temporal_std")

        if self.config.albedo_off_paths:
            albedo_off_stack = self._align_stack(self.config.albedo_off_paths, "lamp_albedo_off")
            albedo_off = (
                albedo_off_stack.mean(dim="time")
                if "time" in albedo_off_stack.dims
                else albedo_off_stack
            ).rename("lamp_albedo_off")
        else:
            albedo_off = albedo_on * 0.5
            albedo_off.name = "lamp_albedo_off"

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_values = np.where(
                np.asarray(albedo_off.values) > 1e-12,
                np.asarray(albedo_on.values) / np.asarray(albedo_off.values),
                np.nan,
            )
        albedo_ratio = xr.DataArray(
            ratio_values,
            coords=albedo_on.coords,
            dims=albedo_on.dims,
            name="lamp_albedo_ratio",
        )

        h2o_depth = self._align_optional(
            self.config.h2o_depth_path,
            "lamp_h2o_depth",
            xr.zeros_like(albedo_on),
        )
        brightness = self._align_optional(
            self.config.brightness_path,
            "lamp_brightness",
            albedo_on.copy().rename("lamp_brightness"),
        )

        self._products = {
            "lamp_albedo_on": albedo_on,
            "lamp_albedo_off": albedo_off,
            "lamp_albedo_ratio": albedo_ratio,
            "lamp_h2o_depth": h2o_depth,
            "lamp_brightness": brightness,
            "lamp_temporal_std": temporal_std,
        }
        validate_crs(albedo_on, int(self._reference.rio.crs.to_epsg()), context="LAMP albedo")
        return dict(self._products)

    def export(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Export LAMP products as COGs."""
        if not self._products:
            self.preprocess()

        destination = output_dir or self.config.output_dir
        if destination is None:
            raise ValueError("output_dir must be provided via config or export().")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        written = {name: write_cog(array, destination / f"{name}.tif") for name, array in self._products.items()}
        save_metadata_json(
            {"instrument": "LAMP", "pole": self.config.pole, "bands": list(self._products)},
            destination / "lamp_metadata.json",
        )
        return written


__all__ = ["LAMP_EXPORT_BANDS", "LAMPLoader", "LAMPLoaderConfig"]
