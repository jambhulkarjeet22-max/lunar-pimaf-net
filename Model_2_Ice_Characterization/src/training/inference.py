from __future__ import annotations
import torch
from pathlib import Path

from src.models.ice_characterization_net import IceCharacterizationNet
from src.training.checkpoint import load_checkpoint

class InferencePipeline:
    def __init__(self, checkpoint_path: Path | str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = IceCharacterizationNet().to(self.device)
        load_checkpoint(self.model, checkpoint_path)
        self.model.eval()
        
    def predict(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(inputs)
            # Apply softmax to classification outputs for convenience
            outputs["ice_type_probs"] = torch.nn.functional.softmax(outputs["ice_type"], dim=1)
            outputs["ice_type_class"] = torch.argmax(outputs["ice_type"], dim=1)
            
            # Move to CPU for further processing
            return {k: v.cpu() for k, v in outputs.items()}
