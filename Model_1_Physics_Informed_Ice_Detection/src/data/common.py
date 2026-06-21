"""Shared geospatial utilities for LUNAR-PIMAF-Net data ingestion.

Provides CRS constants, GeoTIFF I/O, polar reprojection, terrain derivatives,
and quality-control helpers used across instrument loaders.

Reference: docs/DATA_PREPROCESSING_PIPELINE.md §5–§8
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers the .rio accessor on xarray
import xarray as xr
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import calculate_default_transform, reproject
from scipy import ndimage

logger = logging.getLogger(__name__)

Pole = Literal["north", "south"]

# IAU Moon polar stereographic CRS (240 m GDR reference grid).
POLAR_CRS: Final[dict[Pole, int]] = {
    "north": 104905,
    "south": 104906,
}

DEFAULT_PIXEL_SIZE_M: Final[float] = 240.0
DEFAULT_NODATA: Final[float] = -3.4028235e38


@dataclass(frozen=True)
class GridSpec:
    """Reference raster grid specification."""

    crs_epsg: int
    width: int
    height: int
    transform: Affine
    bounds: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize grid metadata to a JSON-compatible mapping."""
        return {
            "crs_epsg": self.crs_epsg,
            "width": self.width,
            "height": self.height,
            "transform": list(self.transform),
            "bounds": list(self.bounds),
        }


@dataclass(frozen=True)
class RasterMetadata:
    """Lightweight raster provenance record."""

    source_path: str
    crs_epsg: int | None
    width: int
    height: int
    nodata: float | None
    dtype: str


def pole_to_epsg(pole: Pole) -> int:
    """Return the EPSG code for a lunar pole."""
    if pole not in POLAR_CRS:
        raise ValueError(f"Unknown pole '{pole}'. Expected one of: {', '.join(POLAR_CRS)}.")
    return POLAR_CRS[pole]


def crs_from_authority_code(code: int) -> CRS:
    """Resolve a lunar polar CRS from its IAU authority code.

    IAU Moon CRS definitions are registered under ESRI in many PROJ builds and
    may not resolve via ``CRS.from_epsg`` even when GeoTIFFs label them as
    ``EPSG:<code>``.
    """
    try:
        return CRS.from_epsg(code)
    except Exception:
        return CRS.from_authority("ESRI", str(code))


def _crs_authority_code(crs_obj: CRS) -> int | None:
    """Extract numeric authority code from a CRS, if available."""
    epsg = crs_obj.to_epsg()
    if epsg is not None:
        return epsg
    authority = crs_obj.to_authority()
    if authority is not None:
        _, code = authority
        try:
            return int(code)
        except (TypeError, ValueError):
            return None
    return None


def validate_crs(data: xr.DataArray, expected_epsg: int, context: str = "raster") -> None:
    """Validate that a raster carries the expected projected CRS.

    Args:
        data: Input ``xarray`` raster with the ``rio`` accessor.
        expected_epsg: Required EPSG integer code.
        context: Label used in error messages.

    Raises:
        ValueError: If CRS is missing or does not match ``expected_epsg``.
        TypeError: If ``data`` is not an ``xarray.DataArray``.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{context} must be an xarray.DataArray, got {type(data).__name__}.")
    if not hasattr(data, "rio"):
        raise ValueError(f"{context} is missing the rioxarray accessor.")

    crs = data.rio.crs
    if crs is None:
        raise ValueError(f"{context} has no CRS metadata.")

    parsed = CRS.from_user_input(crs)
    actual = _crs_authority_code(parsed)
    if actual != expected_epsg:
        raise ValueError(
            f"{context} CRS EPSG:{actual} does not match expected EPSG:{expected_epsg}."
        )


def read_geotiff(
    path: Path | str,
    band: int | None = 1,
    masked: bool = True,
) -> xr.DataArray:
    """Read a single-band or multi-band GeoTIFF into an ``xarray.DataArray``.

    Args:
        path: Path to the GeoTIFF file.
        band: 1-based band index for single-band reads. ``None`` keeps all bands.
        masked: Whether to apply the raster nodata mask.

    Returns:
        ``xarray.DataArray`` with spatial coordinates and CRS metadata.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be read or has zero bands.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"GeoTIFF not found: {resolved}")

    try:
        with rioxarray.open_rasterio(resolved, masked=masked) as opened:
            data = opened.load()
    except rasterio.errors.RasterioIOError as exc:
        raise ValueError(f"Failed to read GeoTIFF '{resolved}': {exc}") from exc

    if data.sizes.get("band", 0) == 0:
        raise ValueError(f"GeoTIFF '{resolved}' contains no bands.")

    if band is not None:
        if band < 1 or band > int(data.sizes["band"]):
            raise ValueError(
                f"Band {band} out of range for '{resolved}' "
                f"(file has {int(data.sizes['band'])} bands)."
            )
        data = data.sel(band=band, drop=True)

    data.name = resolved.stem
    logger.debug("Read GeoTIFF %s with shape %s", resolved, tuple(data.shape))
    return data


def write_cog(
    data: xr.DataArray,
    path: Path | str,
    nodata: float | None = DEFAULT_NODATA,
    compress: str = "deflate",
    overview_levels: Sequence[int] = (2, 4, 8, 16),
) -> Path:
    """Write a Cloud Optimized GeoTIFF (COG).

    Args:
        data: Single-band ``xarray.DataArray`` with CRS and transform metadata.
        path: Output file path.
        nodata: Nodata value written to the COG profile.
        compress: Rasterio compression method.
        overview_levels: Internal overview decimation factors.

    Returns:
        Resolved output path.

    Raises:
        ValueError: If CRS/transform are missing or array is not 2-D.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(data, xr.DataArray):
        raise TypeError(f"write_cog expects xarray.DataArray, got {type(data).__name__}.")
    if data.ndim != 2:
        raise ValueError(f"write_cog expects a 2-D array, got shape {tuple(data.shape)}.")
    if data.rio.crs is None or data.rio.transform() is None:
        raise ValueError("COG export requires CRS and geotransform on the DataArray.")

    array = data.astype("float32")
    if nodata is not None:
        array = array.rio.write_nodata(nodata)

    profile = {
        "driver": "COG",
        "dtype": "float32",
        "compress": compress,
        "blocksize": 512,
        "overview_resampling": "nearest",
    }
    array.rio.to_raster(resolved, **profile)
    logger.info("Wrote COG: %s", resolved)
    return resolved


def normalize_nodata(
    data: xr.DataArray,
    nodata: float | None = None,
    fill_value: float = np.nan,
) -> xr.DataArray:
    """Replace nodata values with ``fill_value`` and return a float copy.

    Args:
        data: Input raster.
        nodata: Explicit nodata value. If ``None``, uses raster metadata.
        fill_value: Replacement value for invalid pixels.

    Returns:
        Float ``xarray.DataArray`` with invalid pixels replaced.
    """
    values = data.astype("float64").copy()
    if nodata is None:
        nodata = values.rio.nodata

    if nodata is not None:
        values = values.where(values != nodata)

    if np.isfinite(fill_value):
        values = values.fillna(fill_value)
    return values


def reproject_to_polar_stereographic(
    data: xr.DataArray,
    pole: Pole,
    resolution_m: float = DEFAULT_PIXEL_SIZE_M,
    resampling: Resampling = Resampling.bilinear,
    dst_bounds: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Reproject a raster to the lunar polar stereographic reference grid.

    Args:
        data: Source raster with CRS metadata.
        pole: Lunar pole identifier (``north`` or ``south``).
        resolution_m: Target pixel size in meters.
        resampling: Rasterio resampling kernel.
        dst_bounds: Optional destination bounds ``(minx, miny, maxx, maxy)``.

    Returns:
        Reprojected ``xarray.DataArray`` in the target CRS.

    Raises:
        ValueError: If source CRS is missing.
    """
    if data.rio.crs is None:
        raise ValueError("Cannot reproject raster without CRS metadata.")

    dst_crs = crs_from_authority_code(pole_to_epsg(pole))
    src_crs = CRS.from_user_input(data.rio.crs)

    if _crs_authority_code(src_crs) == pole_to_epsg(pole):
        return data.copy()

    if dst_bounds is None:
        dst_bounds = tuple(data.rio.bounds())

    transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        int(data.rio.width),
        int(data.rio.height),
        *data.rio.bounds(),
        dst_bounds=dst_bounds,
        resolution=resolution_m,
    )

    destination = np.full((height, width), np.nan, dtype=np.float32)
    source = np.asarray(data.values, dtype=np.float32)
    if source.ndim == 3:
        source = source[0]

    reproject(
        source=source,
        destination=destination,
        src_transform=data.rio.transform(),
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=data.rio.nodata,
        dst_nodata=np.nan,
    )

    coords = {
        "y": np.arange(height) * transform.e + transform.f,
        "x": np.arange(width) * transform.a + transform.c,
    }
    result = xr.DataArray(destination, coords=coords, dims=("y", "x"), name=data.name)
    result = result.rio.write_crs(dst_crs)
    result = result.rio.write_transform(transform)
    return result


def clip_to_polar_region(
    data: xr.DataArray,
    bounds: tuple[float, float, float, float],
) -> xr.DataArray:
    """Clip a raster to axis-aligned projected bounds.

    Args:
        data: Input raster with spatial coordinates.
        bounds: ``(minx, miny, maxx, maxy)`` in projected meters.

    Returns:
        Clipped ``xarray.DataArray``.
    """
    minx, miny, maxx, maxy = bounds
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"Invalid clip bounds: {bounds}")

    return data.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)


def _pixel_size_from_transform(transform: Affine) -> float:
    return float(abs(transform.a))


def compute_slope(
    elevation: xr.DataArray,
    pixel_size_m: float | None = None,
) -> xr.DataArray:
    """Compute terrain slope in degrees using Horn's method.

    Args:
        elevation: Digital elevation model in meters.
        pixel_size_m: Ground sampling distance. Inferred from transform if omitted.

    Returns:
        Slope raster in degrees clipped to ``[0, 90]``.
    """
    if elevation.ndim != 2:
        raise ValueError(f"elevation must be 2-D, got shape {tuple(elevation.shape)}.")

    transform = elevation.rio.transform()
    dx = pixel_size_m or _pixel_size_from_transform(transform)
    dy = pixel_size_m or abs(transform.e)

    dem = np.asarray(elevation.values, dtype=np.float64)
    dz_dy, dz_dx = np.gradient(dem, dy, dx)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope_rad)
    slope_deg = np.clip(slope_deg, 0.0, 90.0)

    return xr.DataArray(
        slope_deg,
        coords=elevation.coords,
        dims=elevation.dims,
        name="lola_slope_deg",
        attrs=elevation.attrs,
    )


def compute_aspect(elevation: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute sine and cosine of terrain aspect.

    Args:
        elevation: Digital elevation model in meters.

    Returns:
        Tuple ``(aspect_sin, aspect_cos)`` each in ``[-1, 1]``.
    """
    if elevation.ndim != 2:
        raise ValueError(f"elevation must be 2-D, got shape {tuple(elevation.shape)}.")

    transform = elevation.rio.transform()
    dx = _pixel_size_from_transform(transform)
    dy = abs(transform.e)

    dem = np.asarray(elevation.values, dtype=np.float64)
    dz_dy, dz_dx = np.gradient(dem, dy, dx)
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    coords = elevation.coords
    dims = elevation.dims
    sin_da = xr.DataArray(aspect_sin, coords=coords, dims=dims, name="lola_aspect_deg_sin")
    cos_da = xr.DataArray(aspect_cos, coords=coords, dims=dims, name="lola_aspect_deg_cos")
    return sin_da, cos_da


def compute_curvature(elevation: xr.DataArray, pixel_size_m: float | None = None) -> xr.DataArray:
    """Compute plan curvature from the DEM using discrete Laplacian.

    Args:
        elevation: Digital elevation model in meters.
        pixel_size_m: Ground sampling distance.

    Returns:
        Curvature raster (1/m).
    """
    if elevation.ndim != 2:
        raise ValueError(f"elevation must be 2-D, got shape {tuple(elevation.shape)}.")

    transform = elevation.rio.transform()
    dx = pixel_size_m or _pixel_size_from_transform(transform)
    dy = pixel_size_m or abs(transform.e)

    dem = np.asarray(elevation.values, dtype=np.float64)
    dz_dy, dz_dx = np.gradient(dem, dy, dx)
    d2z_dy2, _ = np.gradient(dz_dy, dy, dx)
    _, d2z_dx2 = np.gradient(dz_dx, dy, dx)
    curvature = d2z_dx2 + d2z_dy2

    return xr.DataArray(
        curvature,
        coords=elevation.coords,
        dims=elevation.dims,
        name="lola_curvature",
    )


def compute_roughness(
    elevation: xr.DataArray,
    window_size: int = 5,
    pixel_size_m: float | None = None,
) -> xr.DataArray:
    """Compute surface roughness as local elevation standard deviation.

    Args:
        elevation: Digital elevation model in meters.
        window_size: Moving-window size in pixels (odd integer).
        pixel_size_m: Unused, retained for API symmetry with other terrain ops.

    Returns:
        Roughness raster in meters.
    """
    del pixel_size_m  # Roughness is computed in elevation units over pixel windows.
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3.")

    dem = np.asarray(elevation.values, dtype=np.float64)
    rough = ndimage.generic_filter(
        dem,
        np.nanstd,
        size=window_size,
        mode="nearest",
    )
    return xr.DataArray(
        rough,
        coords=elevation.coords,
        dims=elevation.dims,
        name="lola_roughness_m",
    )


def compute_topographic_position_index(
    elevation: xr.DataArray,
    window_size: int = 15,
) -> xr.DataArray:
    """Compute topographic position index (TPI).

    Args:
        elevation: Digital elevation model in meters.
        window_size: Neighborhood size in pixels (odd integer).

    Returns:
        TPI raster in meters.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3.")

    dem = np.asarray(elevation.values, dtype=np.float64)
    local_mean = ndimage.uniform_filter(dem, size=window_size, mode="nearest")
    tpi = dem - local_mean
    return xr.DataArray(
        tpi,
        coords=elevation.coords,
        dims=elevation.dims,
        name="lola_tpi",
    )


def resample_to_reference(
    data: xr.DataArray,
    reference: xr.DataArray,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    """Resample ``data`` to the grid of ``reference``.

    Args:
        data: Source raster with CRS metadata.
        reference: Target grid definition.
        resampling: Rasterio resampling kernel.

    Returns:
        Resampled ``xarray.DataArray`` aligned to ``reference``.
    """
    if data.rio.crs is None or reference.rio.crs is None:
        raise ValueError("Both source and reference rasters require CRS metadata.")

    destination = np.full(
        (int(reference.rio.height), int(reference.rio.width)),
        np.nan,
        dtype=np.float32,
    )
    source = np.asarray(data.values, dtype=np.float32)
    if source.ndim == 3:
        source = source[0]

    reproject(
        source=source,
        destination=destination,
        src_transform=data.rio.transform(),
        src_crs=data.rio.crs,
        dst_transform=reference.rio.transform(),
        dst_crs=reference.rio.crs,
        resampling=resampling,
        src_nodata=data.rio.nodata,
        dst_nodata=np.nan,
    )

    return xr.DataArray(
        destination,
        coords=reference.coords,
        dims=reference.dims,
        name=data.name,
    )


def build_reference_grid(
    pole: Pole,
    bounds: tuple[float, float, float, float],
    resolution_m: float = DEFAULT_PIXEL_SIZE_M,
) -> GridSpec:
    """Construct a reference polar stereographic grid specification.

    Args:
        pole: Lunar pole.
        bounds: Projected bounds ``(minx, miny, maxx, maxy)``.
        resolution_m: Pixel size in meters.

    Returns:
        ``GridSpec`` describing the target grid.
    """
    minx, miny, maxx, maxy = bounds
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"Invalid bounds: {bounds}")

    width = int(np.ceil((maxx - minx) / resolution_m))
    height = int(np.ceil((maxy - miny) / resolution_m))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return GridSpec(
        crs_epsg=pole_to_epsg(pole),
        width=width,
        height=height,
        transform=transform,
        bounds=bounds,
    )


def save_metadata_json(metadata: Mapping[str, Any], path: Path | str) -> Path:
    """Persist metadata to a JSON file.

    Args:
        metadata: JSON-serializable mapping.
        path: Output path.

    Returns:
        Resolved output path.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_default)

    logger.info("Wrote metadata JSON: %s", resolved)
    return resolved


def raster_metadata_from_path(path: Path | str) -> RasterMetadata:
    """Extract lightweight metadata from a GeoTIFF without loading the full array."""
    resolved = Path(path)
    with rasterio.open(resolved) as src:
        crs_epsg = CRS.from_user_input(src.crs).to_epsg() if src.crs else None
        return RasterMetadata(
            source_path=str(resolved),
            crs_epsg=crs_epsg,
            width=src.width,
            height=src.height,
            nodata=src.nodata,
            dtype=src.dtypes[0],
        )


def attach_reference_grid(array: xr.DataArray, grid: GridSpec) -> xr.DataArray:
    """Attach CRS and transform from a ``GridSpec`` to a 2-D array."""
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {tuple(array.shape)}.")
    result = array.copy()
    result = result.rio.write_crs(crs_from_authority_code(grid.crs_epsg))
    result = result.rio.write_transform(grid.transform)
    return result


def valid_pixel_fraction(data: xr.DataArray) -> float:
    """Return the fraction of finite pixels in a raster."""
    values = np.asarray(data.values)
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).mean())


__all__ = [
    "DEFAULT_NODATA",
    "DEFAULT_PIXEL_SIZE_M",
    "POLAR_CRS",
    "GridSpec",
    "Pole",
    "RasterMetadata",
    "attach_reference_grid",
    "build_reference_grid",
    "clip_to_polar_region",
    "compute_aspect",
    "compute_curvature",
    "compute_roughness",
    "compute_slope",
    "compute_topographic_position_index",
    "crs_from_authority_code",
    "normalize_nodata",
    "pole_to_epsg",
    "raster_metadata_from_path",
    "read_geotiff",
    "resample_to_reference",
    "reproject_to_polar_stereographic",
    "save_metadata_json",
    "valid_pixel_fraction",
    "validate_crs",
    "write_cog",
]
