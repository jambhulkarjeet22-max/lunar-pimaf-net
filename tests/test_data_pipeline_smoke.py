"""Smoke tests for the LUNAR-PIMAF-Net geospatial ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from src.data.common import (
    build_reference_grid,
    compute_slope,
    pole_to_epsg,
    read_geotiff,
    save_metadata_json,
    validate_crs,
    write_cog,
)
from src.data.diviner_loader import DivinerLoader, DivinerLoaderConfig
from src.data.lamp_loader import LAMPLoader, LAMPLoaderConfig
from src.data.lend_loader import LENDLoader, LENDLoaderConfig
from src.data.lola_loader import LOLALoader, LOLALoaderConfig
from src.data.m3_loader import M3Loader, M3LoaderConfig
from src.data.mini_rf_loader import MiniRFLoader, MiniRFLoaderConfig
from src.data.preprocessing import (
    FUSION_CHANNELS,
    NUM_FUSION_CHANNELS,
    LunarPreprocessingConfig,
    LunarPreprocessingPipeline,
    PATCH_SIZE,
)


def _write_synthetic_geotiff(path: Path, height: int = 32, width: int = 32) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width)
    transform = from_origin(0.0, float(height), 1.0, 1.0)
    crs = f"EPSG:{pole_to_epsg('north')}"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def test_common_imports_and_geotiff_roundtrip(tmp_path: Path) -> None:
    tif = _write_synthetic_geotiff(tmp_path / "synthetic.tif")
    array = read_geotiff(tif)
    assert array.ndim == 2
    validate_crs(array, pole_to_epsg("north"), context="synthetic")
    assert write_cog(array, tmp_path / "out.tif").is_file()
    assert save_metadata_json({"test": True}, tmp_path / "meta.json").is_file()


def test_terrain_derivatives_on_synthetic_grid() -> None:
    height, width = 16, 16
    dem = np.tile(np.linspace(-2.0, 2.0, width), (height, 1)).astype(np.float32)
    elevation = xr.DataArray(
        dem,
        coords={"y": np.arange(height), "x": np.arange(width)},
        dims=("y", "x"),
    )
    grid = build_reference_grid("north", bounds=(0.0, 0.0, width, height))
    elevation = elevation.rio.write_crs(grid.crs_epsg)
    elevation = elevation.rio.write_transform(grid.transform)
    slope = compute_slope(elevation, pixel_size_m=1.0)
    assert slope.shape == elevation.shape


def test_lola_loader_initialize_and_preprocess(tmp_path: Path) -> None:
    dem = _write_synthetic_geotiff(tmp_path / "lola_dem.tif")
    loader = LOLALoader(LOLALoaderConfig(dem_path=dem, pole="north"))
    loader.load()
    products = loader.preprocess()
    assert "lola_elev_m" in products
    assert products["lola_elev_m"].shape == loader.reference_elevation.shape


def test_fusion_channel_schema() -> None:
    assert len(FUSION_CHANNELS) == NUM_FUSION_CHANNELS == 59
    assert len(set(FUSION_CHANNELS)) == NUM_FUSION_CHANNELS


def test_patch_extraction_on_synthetic_stack() -> None:
    height = PATCH_SIZE * 2
    width = PATCH_SIZE * 2
    stack = np.random.randn(NUM_FUSION_CHANNELS, height, width).astype(np.float32)
    data = xr.DataArray(
        stack,
        coords={"band": list(FUSION_CHANNELS), "y": np.arange(height), "x": np.arange(width)},
        dims=("band", "y", "x"),
    )
    config = LunarPreprocessingConfig(
        pole="north",
        lola=LOLALoaderConfig(dem_path=Path("unused.tif"), pole="north"),
        mini_rf=MiniRFLoaderConfig(
            cpr_path=Path("unused.tif"),
            sc_path=Path("unused.tif"),
            oc_path=Path("unused.tif"),
            pole="north",
        ),
        diviner=DivinerLoaderConfig(tbol_paths=[Path("unused.tif")], pole="north"),
        lend=LENDLoaderConfig(epithermal_path=Path("unused.tif"), pole="north"),
        lamp=LAMPLoaderConfig(albedo_on_paths=[Path("unused.tif")], pole="north"),
        m3=M3LoaderConfig(
            r750_path=Path("unused.tif"),
            r1500_path=Path("unused.tif"),
            pole="north",
        ),
    )
    pipeline = LunarPreprocessingPipeline(config=config)
    pipeline._fusion_stack = data
    patches = pipeline.extract_patches(max_patches=2)
    assert patches.shape == (2, NUM_FUSION_CHANNELS, PATCH_SIZE, PATCH_SIZE)


def test_loader_classes_initialize() -> None:
    assert LOLALoader(LOLALoaderConfig(dem_path=Path("a.tif"), pole="north"))
    assert MiniRFLoader(
        MiniRFLoaderConfig(
            cpr_path=Path("a.tif"),
            sc_path=Path("b.tif"),
            oc_path=Path("c.tif"),
            pole="north",
        )
    )
    assert DivinerLoader(DivinerLoaderConfig(tbol_paths=[Path("a.tif")], pole="north"))
    assert LENDLoader(LENDLoaderConfig(epithermal_path=Path("a.tif"), pole="north"))
    assert LAMPLoader(LAMPLoaderConfig(albedo_on_paths=[Path("a.tif")], pole="north"))
    assert M3Loader(
        M3LoaderConfig(
            r750_path=Path("a.tif"),
            r1500_path=Path("b.tif"),
            pole="north",
        )
    )
