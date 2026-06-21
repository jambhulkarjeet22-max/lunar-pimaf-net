"""Lunar geospatial data ingestion for LUNAR-PIMAF-Net."""

from src.data.common import (
    DEFAULT_PIXEL_SIZE_M,
    POLAR_CRS,
    GridSpec,
    build_reference_grid,
    compute_aspect,
    compute_curvature,
    compute_roughness,
    compute_slope,
    compute_topographic_position_index,
    pole_to_epsg,
    read_geotiff,
    reproject_to_polar_stereographic,
    resample_to_reference,
    save_metadata_json,
    validate_crs,
    write_cog,
)
from src.data.diviner_loader import DIVINER_EXPORT_BANDS, DivinerLoader, DivinerLoaderConfig
from src.data.lamp_loader import LAMP_EXPORT_BANDS, LAMPLoader, LAMPLoaderConfig
from src.data.lend_loader import LEND_EXPORT_BANDS, LENDLoader, LENDLoaderConfig
from src.data.lola_loader import LOLA_EXPORT_BANDS, LOLALoader, LOLALoaderConfig
from src.data.m3_loader import M3_EXPORT_BANDS, M3Loader, M3LoaderConfig
from src.data.mini_rf_loader import MINI_RF_EXPORT_BANDS, MiniRFLoader, MiniRFLoaderConfig
from src.data.preprocessing import (
    FUSION_CHANNELS,
    NUM_FUSION_CHANNELS,
    LunarPreprocessingConfig,
    LunarPreprocessingPipeline,
    MODALITY_GROUPS,
    NormalizationState,
    PATCH_SIZE,
)

__all__ = [
    "DEFAULT_PIXEL_SIZE_M",
    "DIVINER_EXPORT_BANDS",
    "FUSION_CHANNELS",
    "GridSpec",
    "LAMP_EXPORT_BANDS",
    "LEND_EXPORT_BANDS",
    "LOLA_EXPORT_BANDS",
    "LunarPreprocessingConfig",
    "LunarPreprocessingPipeline",
    "M3_EXPORT_BANDS",
    "MINI_RF_EXPORT_BANDS",
    "MODALITY_GROUPS",
    "NUM_FUSION_CHANNELS",
    "NormalizationState",
    "PATCH_SIZE",
    "POLAR_CRS",
    "DivinerLoader",
    "DivinerLoaderConfig",
    "LAMPLoader",
    "LAMPLoaderConfig",
    "LENDLoader",
    "LENDLoaderConfig",
    "LOLALoader",
    "LOLALoaderConfig",
    "M3Loader",
    "M3LoaderConfig",
    "MiniRFLoader",
    "MiniRFLoaderConfig",
    "build_reference_grid",
    "compute_aspect",
    "compute_curvature",
    "compute_roughness",
    "compute_slope",
    "compute_topographic_position_index",
    "pole_to_epsg",
    "read_geotiff",
    "reproject_to_polar_stereographic",
    "resample_to_reference",
    "save_metadata_json",
    "validate_crs",
    "write_cog",
]
