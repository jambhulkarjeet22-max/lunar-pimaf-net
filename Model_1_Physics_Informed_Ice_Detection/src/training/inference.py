"""Inference utilities and export helpers for LUNAR-PIMAF-Net."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import torch
import xarray as xr
from rasterio.transform import from_origin

from src.data.common import crs_from_authority_code, pole_to_epsg, write_cog
from src.models.lunar_pimaf_net import LunarPIMAFNet
from src.training.checkpoint import load_checkpoint
from src.training.config import PredictConfig
from src.training.dataset import LunarPatchDataset, collate_batch
from src.training.trainer import resolve_device

logger = logging.getLogger(__name__)


def build_model_from_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
    fpn_channels: int = 256,
    dropout: float = 0.1,
) -> tuple[LunarPIMAFNet, dict]:
    """Instantiate the model and load weights from a checkpoint."""
    model = LunarPIMAFNet(dropout=dropout, fpn_channels=fpn_channels).to(device)
    payload = load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()
    return model, payload


@torch.no_grad()
def predict_batch(
    model: LunarPIMAFNet,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run inference on a collated batch."""
    inputs = batch["inputs"].to(device)
    outputs = model(inputs)
    return {
        "segmentation_logits": outputs["segmentation_logits"].detach().cpu(),
        "surface_ice_probability": outputs["surface_ice_probability"].detach().cpu(),
        "subsurface_ice_probability": outputs["subsurface_ice_probability"].detach().cpu(),
        "total_uncertainty": outputs["total_uncertainty"].detach().cpu(),
        "confidence": outputs["confidence"].detach().cpu(),
    }


def _probability_to_dataarray(
    probability: np.ndarray,
    pole: str,
) -> xr.DataArray:
    height, width = probability.shape
    transform = from_origin(0.0, float(height), 240.0, 240.0)
    coords = {
        "y": np.arange(height) * transform.e + transform.f,
        "x": np.arange(width) * transform.a + transform.c,
    }
    data_array = xr.DataArray(probability.astype(np.float32), coords=coords, dims=("y", "x"))
    data_array = data_array.rio.write_crs(crs_from_authority_code(pole_to_epsg(pole)))  # type: ignore[arg-type]
    data_array = data_array.rio.write_transform(transform)
    return data_array


def export_probability_geotiff(
    probability: np.ndarray,
    output_path: Path | str,
    *,
    pole: str,
) -> Path:
    """Write a single-band probability map as a Cloud Optimized GeoTIFF."""
    data_array = _probability_to_dataarray(probability, pole=pole)
    return write_cog(data_array, output_path)


def export_probability_png(
    probability: np.ndarray,
    output_path: Path | str,
    *,
    title: str = "Subsurface Ice Probability",
) -> Path:
    """Save a quick-look PNG visualization of a probability map."""
    import matplotlib.pyplot as plt

    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(5, 5), dpi=120)
    image = axis.imshow(probability, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(image, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(resolved, bbox_inches="tight")
    plt.close(fig)
    return resolved


def run_prediction(config: PredictConfig) -> list[Path]:
    """Execute batch inference and export probability maps."""
    device = resolve_device(config.device)
    model, _payload = build_model_from_checkpoint(
        config.checkpoint,
        device=device,
        fpn_channels=config.fpn_channels,
        dropout=config.dropout,
    )

    dataset = LunarPatchDataset(
        data_path=config.data_path,
        synthetic_samples=config.synthetic_samples if config.data_path is None else 0,
        seed=42,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []

    for batch_idx, batch in enumerate(loader):
        outputs = predict_batch(model, batch, device=device)
        batch_size = outputs["subsurface_ice_probability"].shape[0]

        for sample_idx in range(batch_size):
            subsurface = outputs["subsurface_ice_probability"][sample_idx, 0].numpy()
            surface = outputs["surface_ice_probability"][sample_idx, 0].numpy()
            prefix = output_dir / f"patch_{batch_idx:04d}_{sample_idx:02d}"

            if config.export_geotiff:
                geotiff_path = export_probability_geotiff(
                    subsurface,
                    prefix.with_name(prefix.name + "_subsurface.tif"),
                    pole=config.pole,
                )
                exported.append(geotiff_path)
                surface_path = export_probability_geotiff(
                    surface,
                    prefix.with_name(prefix.name + "_surface.tif"),
                    pole=config.pole,
                )
                exported.append(surface_path)

            if config.export_png:
                png_path = export_probability_png(
                    subsurface,
                    prefix.with_name(prefix.name + "_subsurface.png"),
                    title="Subsurface Ice Probability",
                )
                exported.append(png_path)

    logger.info("Exported %d artifacts to %s", len(exported), output_dir)
    return exported


__all__ = [
    "build_model_from_checkpoint",
    "export_probability_geotiff",
    "export_probability_png",
    "predict_batch",
    "run_prediction",
]
