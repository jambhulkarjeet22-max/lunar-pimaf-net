#!/usr/bin/env python3
"""Train the lunar Landing Site Intelligence model."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.dataset_utils import ensure_import_paths

ensure_import_paths(Path(__file__).resolve().parent)

from Model_3_Landing_Site_Intelligence.src.training.config import TrainingConfig
from Model_3_Landing_Site_Intelligence.src.training.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Landing Site Intelligence model.")
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
