"""
LUNAR OS — Inference Entry Point
Physics-Informed Ice Detection Engine

Runs batch ice-detection inference and exports probability maps.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.training.config import PredictConfig
from src.training.inference import run_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


_DEFAULT = PredictConfig()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LUNAR-PIMAF-Net inference")
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT.checkpoint)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT.output_dir)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT.batch_size)
    parser.add_argument("--num-workers", type=int, default=_DEFAULT.num_workers)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--pole", type=str, default=_DEFAULT.pole, choices=("north", "south"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-geotiff", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--fpn-channels", type=int, default=_DEFAULT.fpn_channels)
    parser.add_argument("--dropout", type=float, default=_DEFAULT.dropout)
    return parser


def config_from_args(args: argparse.Namespace) -> PredictConfig:
    return PredictConfig(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        synthetic_samples=args.synthetic_samples,
        pole=args.pole,
        device=args.device,
        export_geotiff=not args.no_geotiff,
        export_png=not args.no_png,
        fpn_channels=args.fpn_channels,
        dropout=args.dropout,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    logger.info("Running inference from checkpoint: %s", config.checkpoint)
    exported = run_prediction(config)
    logger.info("Wrote %d files to %s", len(exported), config.output_dir)


if __name__ == "__main__":
    main()
