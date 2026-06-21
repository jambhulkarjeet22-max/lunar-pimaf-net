from __future__ import annotations

import torch
from torch.utils.data import Dataset


class IceCharacterizationDataset(Dataset):
    """Dataset for Lunar Ice Characterization."""
    def __init__(self, num_samples: int = 100, img_size: int = 64):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        
        # Modalities and their respective channels
        self.modalities = {
            "mini_rf": 3,
            "diviner": 1,
            "lola": 1,
            "lend": 1,
            "lamp": 1,
            "m3": 2,
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        # Generate synthetic input data
        inputs = {}
        for mod, channels in self.modalities.items():
            inputs[mod] = torch.randn(channels, self.img_size, self.img_size)
            
        # Generate synthetic targets
        targets = {
            "purity_percentage": torch.rand(1, self.img_size, self.img_size),
            "ice_depth": torch.rand(1, self.img_size, self.img_size) * 10.0, # 0-10 meters
            "ice_type": torch.randint(0, 3, (self.img_size, self.img_size), dtype=torch.long),
            "stability_score": torch.rand(1, self.img_size, self.img_size),
        }
        
        return {"inputs": inputs, "targets": targets}


def collate_dict_batch(batch: list[dict]) -> dict:
    """Collate a batch of dictionaries containing inputs and targets."""
    collated = {"inputs": {}, "targets": {}}
    
    first_sample = batch[0]
    for mod in first_sample["inputs"].keys():
        collated["inputs"][mod] = torch.stack([item["inputs"][mod] for item in batch])
        
    for target in first_sample["targets"].keys():
        collated["targets"][target] = torch.stack([item["targets"][target] for item in batch])
        
    return collated
