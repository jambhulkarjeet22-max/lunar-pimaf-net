"""Model 2 — Ice Characterization inference entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shared.dataset_utils import ensure_import_paths
ensure_import_paths()

from Model_2_Ice_Characterization.src.training.inference import InferencePipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ice Characterization Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    pipeline = InferencePipeline(args.checkpoint)
    
    # Generate dummy input to demonstrate inference
    print("Running inference on dummy data...")
    dummy_inputs = {
        "mini_rf": torch.randn(1, 3, 64, 64),
        "diviner": torch.randn(1, 1, 64, 64),
        "lola": torch.randn(1, 1, 64, 64),
        "lend": torch.randn(1, 1, 64, 64),
        "lamp": torch.randn(1, 1, 64, 64),
        "m3": torch.randn(1, 2, 64, 64),
    }
    
    outputs = pipeline.predict(dummy_inputs)
    print("Inference successful. Output keys:", outputs.keys())
    for k, v in outputs.items():
        print(f" - {k}: shape {v.shape}")

if __name__ == "__main__":
    main()
