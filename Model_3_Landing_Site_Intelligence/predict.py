#!/usr/bin/env python3
"""Run inference with a trained Landing Site Intelligence checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.dataset_utils import ensure_import_paths

ensure_import_paths(Path(__file__).resolve().parent)

from Model_3_Landing_Site_Intelligence.src.training.config import PredictConfig
from Model_3_Landing_Site_Intelligence.src.training.inference import run_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict landing suitability scores.")
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
