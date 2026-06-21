"""Lightweight smoke runner for data pipeline (no pytest required)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.transform import from_origin

from src.data.common import pole_to_epsg, read_geotiff, validate_crs, write_cog
from src.data.diviner_loader import DivinerLoader, DivinerLoaderConfig
from src.data.lamp_loader import LAMPLoader, LAMPLoaderConfig
from src.data.lend_loader import LENDLoader, LENDLoaderConfig
from src.data.lola_loader import LOLALoader, LOLALoaderConfig
from src.data.m3_loader import M3Loader, M3LoaderConfig
from src.data.mini_rf_loader import MiniRFLoader, MiniRFLoaderConfig
from src.data.preprocessing import (
    FUSION_CHANNELS,
    LunarPreprocessingConfig,
    LunarPreprocessingPipeline,
    PATCH_SIZE,
)


def main() -> None:
    print("imports ok")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tif = tmp_path / "synthetic.tif"
        data = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
        epsg = pole_to_epsg("north")
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=32,
            width=32,
            count=1,
            dtype="float32",
            crs=f"EPSG:{epsg}",
            transform=from_origin(0, 32, 1, 1),
        ) as dst:
            dst.write(data, 1)

        arr = read_geotiff(tif)
        validate_crs(arr, epsg)
        assert write_cog(arr, tmp_path / "out.tif").is_file()

        loader = LOLALoader(LOLALoaderConfig(dem_path=tif, pole="north"))
        loader.load()
        products = loader.preprocess()
        assert "lola_elev_m" in products

    assert len(FUSION_CHANNELS) == 59
    height = PATCH_SIZE * 2
    stack = np.random.randn(59, height, height).astype(np.float32)
    da = xr.DataArray(
        stack,
        coords={"band": list(FUSION_CHANNELS), "y": np.arange(height), "x": np.arange(height)},
        dims=("band", "y", "x"),
    )
    cfg = LunarPreprocessingConfig(
        pole="north",
        lola=LOLALoaderConfig(dem_path=Path("x.tif"), pole="north"),
        mini_rf=MiniRFLoaderConfig(
            cpr_path=Path("a.tif"),
            sc_path=Path("b.tif"),
            oc_path=Path("c.tif"),
            pole="north",
        ),
        diviner=DivinerLoaderConfig(tbol_paths=[Path("a.tif")], pole="north"),
        lend=LENDLoaderConfig(epithermal_path=Path("a.tif"), pole="north"),
        lamp=LAMPLoaderConfig(albedo_on_paths=[Path("a.tif")], pole="north"),
        m3=M3LoaderConfig(r750_path=Path("a.tif"), r1500_path=Path("b.tif"), pole="north"),
    )
    pipe = LunarPreprocessingPipeline(config=cfg)
    pipe._fusion_stack = da
    patches = pipe.extract_patches(max_patches=2)
    assert patches.shape == (2, 59, 128, 128)

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
    assert M3Loader(M3LoaderConfig(r750_path=Path("a.tif"), r1500_path=Path("b.tif"), pole="north"))

    print("smoke tests passed")


if __name__ == "__main__":
    main()
