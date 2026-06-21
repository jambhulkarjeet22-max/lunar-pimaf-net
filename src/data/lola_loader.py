"""LOLA topography loader for LUNAR-PIMAF-Net.

Loads LOLA DEM products, derives terrain metrics, and exports Cloud Optimized
GeoTIFFs on the lunar polar stereographic reference grid.

Reference: docs/DATA_PREPROCESSING_PIPELINE.md §3.1, §4.2 (channels 0–7)
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
    compute_aspect,
    compute_curvature,
    compute_roughness,
    compute_slope,
    compute_topographic_position_index,
    normalize_nodata,
    pole_to_epsg,
    read_geotiff,
    reproject_to_polar_stereographic,
    resample_to_reference,
    save_metadata_json,
    validate_crs,
    write_cog,
)

logger = logging.getLogger(__name__)

LOLA_EXPORT_BANDS: Final[tuple[str, ...]] = (
    "lola_elev_m",
    "lola_slope_deg",
    "lola_aspect_deg_sin",
    "lola_aspect_deg_cos",
    "lola_roughness_m",
    "lola_tpi",
    "lola_curvature",
    "lola_psr_fraction",
)


@dataclass
class LOLALoaderConfig:
    """Configuration for LOLA DEM ingestion."""

    dem_path: Path
    pole: Pole
    psr_path: Path | None = None
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M
    slope_window: int = 5
    tpi_window: int = 15
    output_dir: Path | None = None


@dataclass
class LOLALoader:
    """Load and preprocess LOLA digital elevation and derived terrain layers.

    The LOLA DEM defines the reference grid for all downstream sensor alignment.
    Derived layers match the 8-channel topography group in the fusion tensor.
    """

    config: LOLALoaderConfig
    _elevation: xr.DataArray | None = field(default=None, init=False, repr=False)
    _psr_fraction: xr.DataArray | None = field(default=None, init=False, repr=False)
    _products: dict[str, xr.DataArray] = field(default_factory=dict, init=False, repr=False)

    def load(self) -> None:
        """Load LOLA DEM and optional PSR fraction raster."""
        if not self.config.dem_path.is_file():
            raise FileNotFoundError(f"LOLA DEM not found: {self.config.dem_path}")

        elevation = read_geotiff(self.config.dem_path)
        elevation = normalize_nodata(elevation)
        elevation = reproject_to_polar_stereographic(
            elevation,
            pole=self.config.pole,
            resolution_m=self.config.pixel_size_m,
        )
        elevation.name = "lola_elev_m"
        validate_crs(elevation, pole_to_epsg(self.config.pole), context="LOLA DEM")
        self._elevation = elevation

        if self.config.psr_path is not None:
            if not self.config.psr_path.is_file():
                raise FileNotFoundError(f"PSR raster not found: {self.config.psr_path}")
            psr = read_geotiff(self.config.psr_path)
            psr = normalize_nodata(psr, fill_value=0.0)
            psr = reproject_to_polar_stereographic(
                psr,
                pole=self.config.pole,
                resolution_m=self.config.pixel_size_m,
            )
            psr = resample_to_reference(psr, elevation)
            psr = psr.clip(0.0, 1.0)
            psr.name = "lola_psr_fraction"
            self._psr_fraction = psr
        else:
            logger.warning("No PSR raster provided; using zero PSR fraction.")
            self._psr_fraction = xr.zeros_like(elevation, dtype=np.float32)
            self._psr_fraction.name = "lola_psr_fraction"

        logger.info(
            "Loaded LOLA DEM from %s with shape %s",
            self.config.dem_path,
            tuple(elevation.shape),
        )

    def preprocess(self) -> dict[str, xr.DataArray]:
        """Derive slope, aspect, curvature, roughness, TPI, and PSR fraction.

        Returns:
            Mapping of LOLA product names to aligned ``xarray.DataArray`` objects.

        Raises:
            RuntimeError: If :meth:`load` has not been called.
        """
        if self._elevation is None or self._psr_fraction is None:
            raise RuntimeError("Call load() before preprocess().")

        elevation = self._elevation
        slope = compute_slope(elevation, pixel_size_m=self.config.pixel_size_m)
        aspect_sin, aspect_cos = compute_aspect(elevation)
        roughness = compute_roughness(
            elevation,
            window_size=self.config.slope_window,
            pixel_size_m=self.config.pixel_size_m,
        )
        tpi = compute_topographic_position_index(
            elevation,
            window_size=self.config.tpi_window,
        )
        curvature = compute_curvature(elevation, pixel_size_m=self.config.pixel_size_m)

        self._products = {
            "lola_elev_m": elevation,
            "lola_slope_deg": slope,
            "lola_aspect_deg_sin": aspect_sin,
            "lola_aspect_deg_cos": aspect_cos,
            "lola_roughness_m": roughness,
            "lola_tpi": tpi,
            "lola_curvature": curvature,
            "lola_psr_fraction": self._psr_fraction,
        }
        return dict(self._products)

    def export(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Export LOLA products as Cloud Optimized GeoTIFFs.

        Args:
            output_dir: Destination directory. Defaults to ``config.output_dir``.

        Returns:
            Mapping of product names to written COG paths.

        Raises:
            RuntimeError: If preprocessing has not been run.
            ValueError: If no output directory is configured.
        """
        if not self._products:
            self.preprocess()

        destination = output_dir or self.config.output_dir
        if destination is None:
            raise ValueError("output_dir must be provided via config or export().")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}
        for name, array in self._products.items():
            path = destination / f"{name}.tif"
            written[name] = write_cog(array, path)

        save_metadata_json(
            {
                "instrument": "LOLA",
                "pole": self.config.pole,
                "source_dem": str(self.config.dem_path),
                "bands": list(self._products.keys()),
                "pixel_size_m": self.config.pixel_size_m,
            },
            destination / "lola_metadata.json",
        )
        logger.info("Exported %d LOLA COGs to %s", len(written), destination)
        return written

    @property
    def reference_elevation(self) -> xr.DataArray:
        """Return the reference elevation grid used for downstream alignment."""
        if self._elevation is None:
            raise RuntimeError("Call load() before accessing reference_elevation.")
        return self._elevation

    @property
    def products(self) -> dict[str, xr.DataArray]:
        """Return the latest preprocessed product mapping."""
        return dict(self._products)


__all__ = ["LOLA_EXPORT_BANDS", "LOLALoader", "LOLALoaderConfig"]
