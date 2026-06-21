"""Smoke tests for Model 2 — Ice Characterization."""

from __future__ import annotations

import sys
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shared.dataset_utils import ensure_import_paths
ensure_import_paths()

from Model_2_Ice_Characterization.src.models.ice_characterization_net import IceCharacterizationNet
from Model_2_Ice_Characterization.src.data.dataset import IceCharacterizationDataset, collate_dict_batch
from torch.utils.data import DataLoader

def test_model_forward():
    print("Testing Model Forward Pass...")
    model = IceCharacterizationNet()
    inputs = {
        "mini_rf": torch.randn(2, 3, 32, 32),
        "diviner": torch.randn(2, 1, 32, 32),
        "lola": torch.randn(2, 1, 32, 32),
        "lend": torch.randn(2, 1, 32, 32),
        "lamp": torch.randn(2, 1, 32, 32),
        "m3": torch.randn(2, 2, 32, 32),
    }
    outputs = model(inputs)
    
    expected_keys = ["purity_percentage", "ice_depth", "ice_type", "stability_score", "confidence"]
    for key in expected_keys:
        assert key in outputs, f"Missing output: {key}"
        
    assert outputs["purity_percentage"].shape == (2, 1, 16, 16)
    assert outputs["ice_type"].shape == (2, 3, 16, 16)
    print("Forward Pass OK!")

def test_dataset():
    print("Testing Dataset and Dataloader...")
    dataset = IceCharacterizationDataset(num_samples=4, img_size=32)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_dict_batch)
    
    batch = next(iter(loader))
    assert "inputs" in batch and "targets" in batch
    assert batch["inputs"]["mini_rf"].shape == (2, 3, 32, 32)
    assert batch["targets"]["ice_type"].shape == (2, 32, 32)
    print("Dataset OK!")

def main():
    try:
        test_model_forward()
        test_dataset()
        print("ALL SMOKE TESTS PASSED!")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
