"""Training smoke test for Model 2."""

import sys
from pathlib import Path
import os
import shutil

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_2_Ice_Characterization.src.training.config import TrainingConfig
from Model_2_Ice_Characterization.src.training.trainer import Trainer

def main():
    test_logs = "test_logs"
    test_checkpoints = "test_saved_models"
    
    os.makedirs(test_logs, exist_ok=True)
    os.makedirs(test_checkpoints, exist_ok=True)
    
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        output_dir=test_logs,
        checkpoint_dir=test_checkpoints,
    )
    
    try:
        trainer = Trainer(config)
        trainer.train()
        print("training smoke tests passed")
    finally:
        if os.path.exists(test_logs):
            shutil.rmtree(test_logs)
        if os.path.exists(test_checkpoints):
            shutil.rmtree(test_checkpoints)

if __name__ == "__main__":
    main()
