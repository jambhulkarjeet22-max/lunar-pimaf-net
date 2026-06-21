#!/usr/bin/env python3
"""Train the lunar Rover Hazard & Navigation Intelligence model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repository root to sys.path to enable imports of shared and Model_5_Rover_Hazard_Navigation
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

from Model_5_Rover_Hazard_Navigation.src.training.config import TrainingConfig
from Model_5_Rover_Hazard_Navigation.src.training.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rover Hazard & Navigation model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        patch_size=args.patch_size,
        val_fraction=args.val_fraction,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        use_amp=args.use_amp,
    )
    metrics = run_training(config)
    print("Training complete:", metrics)


if __name__ == "__main__":
    main()
