"""Predict smoke test for Model 2."""

import sys
from pathlib import Path
import os
import torch
import shutil

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_2_Ice_Characterization.src.models.ice_characterization_net import IceCharacterizationNet
from Model_2_Ice_Characterization.src.training.checkpoint import save_checkpoint
from Model_2_Ice_Characterization.src.training.inference import InferencePipeline

def main():
    test_checkpoint_dir = "test_smoke_checkpoints"
    os.makedirs(test_checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(test_checkpoint_dir, "dummy_checkpoint.pt")
    
    try:
        model = IceCharacterizationNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        save_checkpoint(model, optimizer, 1, checkpoint_path)
        
        pipeline = InferencePipeline(checkpoint_path)
        
        inputs = {
            "mini_rf": torch.randn(2, 3, 32, 32),
            "diviner": torch.randn(2, 1, 32, 32),
            "lola": torch.randn(2, 1, 32, 32),
            "lend": torch.randn(2, 1, 32, 32),
            "lamp": torch.randn(2, 1, 32, 32),
            "m3": torch.randn(2, 2, 32, 32),
        }
        
        outputs = pipeline.predict(inputs)
        assert "ice_type_probs" in outputs
        assert "ice_type_class" in outputs
        
        print("predict smoke tests passed")
        
    finally:
        if os.path.exists(test_checkpoint_dir):
            shutil.rmtree(test_checkpoint_dir)

if __name__ == "__main__":
    main()
