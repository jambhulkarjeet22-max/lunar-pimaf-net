"""Diviner thermal loader for LUNAR-PIMAF-Net.

Aggregates multi-temporal Diviner radiometry, derives thermal inertia and ice
stability proxies, and exports aligned thermal feature stacks.

Reference: docs/DATA_PREPROCESSING_PIPELINE.md §3.3, §4.2 (channels 18–32)
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

DIVINER_EXPORT_BANDS: Final[tuple[str, ...]] = (
    "div_tbol_max",
    "div_tbol_min",
    "div_tbol_mean",
    "div_tbol_std",
    "div_ch3_tb",
    "div_ch4_tb",
    "div_ch5_tb",
    "div_ch6_tb",
    "div_ch7_tb",
    "div_ch8_tb",
    "div_emissivity_ch7",
    "div_thermal_inertia",
    "div_ice_stability",
    "div_permafrost_depth_m",
    "div_insolation",
)

T_TRAP_K: Final[float] = 110.0
STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8


@dataclass
class DivinerLoaderConfig:
    """Configuration for Diviner thermal ingestion."""

    tbol_paths: Sequence[Path]
    pole: Pole
    emissivity_path: Path | None = None
    thermal_inertia_path: Path | None = None
    insolation_path: Path | None = None
    cold_trap_path: Path | None = None
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M
    temperature_clip_k: tuple[float, float] = (20.0, 400.0)
    output_dir: Path | None = None


@dataclass
class DivinerLoader:
    """Load and preprocess Diviner thermal radiometry."""

    config: DivinerLoaderConfig
    _reference: xr.DataArray | None = field(default=None, init=False, repr=False)
    _products: dict[str, xr.DataArray] = field(default_factory=dict, init=False, repr=False)

    def set_reference_grid(self, reference: xr.DataArray) -> None:
        """Attach the LOLA reference grid."""
        if reference.ndim != 2:
            raise ValueError("reference grid must be 2-D.")
        self._reference = reference

    def _align_optional(self, path: Path | None, name: str, default: xr.DataArray) -> xr.DataArray:
        if path is None:
            return default
        if not path.is_file():
            raise FileNotFoundError(f"Diviner input not found: {path}")
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

    def _load_temporal_stack(self) -> xr.DataArray:
        if not self.config.tbol_paths:
            raise ValueError("At least one Diviner tbol path is required.")
        if self._reference is None:
            raise RuntimeError("Reference grid not set.")

        aligned: list[xr.DataArray] = []
        for path in self.config.tbol_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Diviner tbol raster not found: {path}")
            band = read_geotiff(path)
            band = normalize_nodata(band)
            band = reproject_to_polar_stereographic(
                band,
                pole=self.config.pole,
                resolution_m=self.config.pixel_size_m,
            )
            band = resample_to_reference(band, self._reference)
            low, high = self.config.temperature_clip_k
            band = band.clip(low, high)
            aligned.append(band)

        stack = xr.concat(aligned, dim="time")
        return stack

    @staticmethod
    def _ice_stability_proxy(tmax: xr.DataArray, trap_k: float = T_TRAP_K) -> xr.DataArray:
        values = np.asarray(tmax.values, dtype=np.float64)
        stability = 1.0 / (1.0 + np.exp((values - trap_k) / 5.0))
        return xr.DataArray(
            stability,
            coords=tmax.coords,
            dims=tmax.dims,
            name="div_ice_stability",
        )

    @staticmethod
    def _permafrost_depth_proxy(
        tmax: xr.DataArray,
        thermal_inertia: xr.DataArray,
    ) -> xr.DataArray:
        ti = np.asarray(thermal_inertia.values, dtype=np.float64)
        t_vals = np.asarray(tmax.values, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(ti > 0.0, np.sqrt(np.maximum(T_TRAP_K - t_vals, 0.0)) * (ti / 100.0), 0.0)
        return xr.DataArray(
            depth,
            coords=tmax.coords,
            dims=tmax.dims,
            name="div_permafrost_depth_m",
        )

    def load(self) -> None:
        """Validate Diviner source paths."""
        if not self.config.tbol_paths:
            raise ValueError("tbol_paths must contain at least one GeoTIFF path.")
        for path in self.config.tbol_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Diviner tbol raster not found: {path}")

    def preprocess(self) -> dict[str, xr.DataArray]:
        """Aggregate temporal observations and derive thermal products."""
        if self._reference is None:
            raise RuntimeError("Call set_reference_grid() before preprocess().")

        stack = self._load_temporal_stack()
        tmax = stack.max(dim="time").rename("div_tbol_max")
        tmin = stack.min(dim="time").rename("div_tbol_min")
        tmean = stack.mean(dim="time").rename("div_tbol_mean")
        tstd = stack.std(dim="time").rename("div_tbol_std")

        template = xr.full_like(tmax, np.nan)
        emissivity = self._align_optional(
            self.config.emissivity_path,
            "div_emissivity_ch7",
            template,
        ).clip(0.0, 1.0)
        thermal_inertia = self._align_optional(
            self.config.thermal_inertia_path,
            "div_thermal_inertia",
            template,
        )
        insolation = self._align_optional(
            self.config.insolation_path,
            "div_insolation",
            template,
        )
        cold_trap = self._align_optional(
            self.config.cold_trap_path,
            "div_cold_trap_mask",
            xr.zeros_like(tmax),
        ).clip(0.0, 1.0)

        ice_stability = self._ice_stability_proxy(tmax)
        permafrost_depth = self._permafrost_depth_proxy(tmax, thermal_inertia.fillna(50.0))

        channel_names = [
            "div_ch3_tb",
            "div_ch4_tb",
            "div_ch5_tb",
            "div_ch6_tb",
            "div_ch7_tb",
            "div_ch8_tb",
        ]
        channel_arrays: dict[str, xr.DataArray] = {}
        if stack.sizes.get("time", 0) >= len(channel_names):
            for idx, name in enumerate(channel_names):
                channel_arrays[name] = stack.isel(time=idx).rename(name)
        else:
            for name in channel_names:
                channel_arrays[name] = tmean.rename(name)

        self._products = {
            "div_tbol_max": tmax,
            "div_tbol_min": tmin,
            "div_tbol_mean": tmean,
            "div_tbol_std": tstd,
            **channel_arrays,
            "div_emissivity_ch7": emissivity,
            "div_thermal_inertia": thermal_inertia,
            "div_ice_stability": ice_stability,
            "div_permafrost_depth_m": permafrost_depth,
            "div_insolation": insolation,
            "div_cold_trap_mask": cold_trap,
        }
        validate_crs(tmax, int(self._reference.rio.crs.to_epsg()), context="Diviner tmax")
        return dict(self._products)

    def export(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Export Diviner products as COGs."""
        if not self._products:
            self.preprocess()

        destination = output_dir or self.config.output_dir
        if destination is None:
            raise ValueError("output_dir must be provided via config or export().")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        written = {name: write_cog(array, destination / f"{name}.tif") for name, array in self._products.items()}
        save_metadata_json(
            {
                "instrument": "Diviner",
                "pole": self.config.pole,
                "temporal_observations": len(self.config.tbol_paths),
                "bands": list(self._products),
            },
            destination / "diviner_metadata.json",
        )
        return written


__all__ = ["DIVINER_EXPORT_BANDS", "DivinerLoader", "DivinerLoaderConfig"]
