"""Cross-model utilities shared by all LUNAR OS model packages."""

from shared.dataset_utils import collate_tensor_dict, resolve_repo_paths
from shared.geospatial_utils import crs_from_authority_code, pole_to_epsg, validate_crs
from shared.lunar_constants import (
    DEFAULT_NODATA,
    DEFAULT_PIXEL_SIZE_M,
    PATCH_SIZE,
    POLAR_CRS,
    Pole,
)
from shared.uncertainty_utils import dirichlet_entropy, normalize_uncertainty_map
from shared.visualization import save_probability_png

__all__ = [
    "DEFAULT_NODATA",
    "DEFAULT_PIXEL_SIZE_M",
    "PATCH_SIZE",
    "POLAR_CRS",
    "Pole",
    "collate_tensor_dict",
    "crs_from_authority_code",
    "dirichlet_entropy",
    "normalize_uncertainty_map",
    "pole_to_epsg",
    "resolve_repo_paths",
    "save_probability_png",
    "validate_crs",
]
