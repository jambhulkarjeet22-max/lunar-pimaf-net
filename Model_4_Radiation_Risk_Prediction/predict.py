#!/usr/bin/env python3
"""Run inference with a trained Radiation Risk Prediction checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repository root to sys.path to enable imports of shared and Model_4_Radiation_Risk_Prediction
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Automatically create shared/__init__.py if missing
shared_init_path = repo_root / "shared" / "__init__.py"
if not shared_init_path.exists():
    shared_init_path.parent.mkdir(parents=True, exist_ok=True)
    shared_init_path.write_text(
        '"""Cross-model utilities shared by all LUNAR OS model packages."""\n\n'
        'from .dataset_utils import collate_tensor_dict, resolve_repo_paths\n'
        'from .geospatial_utils import crs_from_authority_code, pole_to_epsg, validate_crs\n'
        'from .lunar_constants import (\n'
        '    DEFAULT_NODATA,\n'
        '    DEFAULT_PIXEL_SIZE_M,\n'
        '    PATCH_SIZE,\n'
        '    POLAR_CRS,\n'
        '    Pole,\n'
        ')\n'
        'from .uncertainty_utils import dirichlet_entropy, normalize_uncertainty_map\n'
        'from .visualization import save_probability_png\n\n'
        '__all__ = [\n'
        '    "DEFAULT_NODATA",\n'
        '    "DEFAULT_PIXEL_SIZE_M",\n'
        '    "PATCH_SIZE",\n'
        '    "POLAR_CRS",\n'
        '    "Pole",\n'
        '    "collate_tensor_dict",\n'
        '    "crs_from_authority_code",\n'
        '    "dirichlet_entropy",\n'
        '    "normalize_uncertainty_map",\n'
        '    "pole_to_epsg",\n'
        '    "resolve_repo_paths",\n'
        '    "save_probability_png",\n'
        '    "validate_crs",\n'
        ']\n',
        encoding="utf-8"
    )

from shared.dataset_utils import ensure_import_paths
ensure_import_paths(Path(__file__).resolve().parent)

from Model_4_Radiation_Risk_Prediction.src.training.config import PredictConfig
from Model_4_Radiation_Risk_Prediction.src.training.inference import run_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict radiation risk maps.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--output-dir", type=str, default="predictions")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PredictConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        device=args.device,
    )
    summary = run_prediction(config)
    print("Prediction complete:", summary)


if __name__ == "__main__":
    main()
