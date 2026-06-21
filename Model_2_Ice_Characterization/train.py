"""Model 2 — Ice Characterization training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure shared imports work when running from Model 2 root
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shared.dataset_utils import ensure_import_paths
ensure_import_paths()

from Model_2_Ice_Characterization.src.training.config import TrainingConfig
from Model_2_Ice_Characterization.src.training.trainer import Trainer

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ice Characterization Model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir="Model_2_Ice_Characterization/logs/",
        checkpoint_dir="Model_2_Ice_Characterization/saved_models/",
    )
    
    print("Initializing Training...")
    trainer = Trainer(config)
    trainer.train()
    print("Training Complete!")

if __name__ == "__main__":
    main()
