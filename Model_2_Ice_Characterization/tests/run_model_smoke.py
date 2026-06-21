"""Model smoke test for Model 2."""

import sys
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Model_2_Ice_Characterization.src.models.ice_characterization_net import IceCharacterizationNet

def main():
    model = IceCharacterizationNet()
    model.eval()

    inputs = {
        "mini_rf": torch.randn(2, 3, 32, 32),
        "diviner": torch.randn(2, 1, 32, 32),
        "lola": torch.randn(2, 1, 32, 32),
        "lend": torch.randn(2, 1, 32, 32),
        "lamp": torch.randn(2, 1, 32, 32),
        "m3": torch.randn(2, 2, 32, 32),
    }

    with torch.no_grad():
        outputs = model(inputs)

    expected_keys = ["purity_percentage", "ice_depth", "ice_type", "stability_score", "confidence"]
    for key in expected_keys:
        assert key in outputs, f"Missing output: {key}"

    print("model smoke tests passed")

if __name__ == "__main__":
    main()
