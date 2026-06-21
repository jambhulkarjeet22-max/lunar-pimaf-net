"""Geospatial helpers for lunar polar stereographic products."""

from __future__ import annotations

import xarray as xr
from pyproj import CRS

from shared.lunar_constants import Pole, POLAR_CRS


def pole_to_epsg(pole: Pole) -> int:
    """Return the IAU Moon polar stereographic authority code for a pole."""
    if pole not in POLAR_CRS:
        raise ValueError(f"Unknown pole '{pole}'. Expected one of: {', '.join(POLAR_CRS)}.")
    return POLAR_CRS[pole]


def crs_from_authority_code(code: int) -> CRS:
    """Resolve lunar CRS codes via EPSG or ESRI authority fallback."""
    try:
        return CRS.from_epsg(code)
    except Exception:
        return CRS.from_authority("ESRI", str(code))


def _crs_authority_code(crs_obj: CRS) -> int | None:
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
    """Validate that a raster carries the expected projected CRS."""
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{context} must be an xarray.DataArray, got {type(data).__name__}.")
    if not hasattr(data, "rio"):
        raise ValueError(f"{context} is missing the rioxarray accessor.")

    crs = data.rio.crs
    if crs is None:
        raise ValueError(f"{context} has no CRS metadata.")

    actual = _crs_authority_code(CRS.from_user_input(crs))
    if actual != expected_epsg:
        raise ValueError(
            f"{context} CRS EPSG:{actual} does not match expected EPSG:{expected_epsg}."
        )
