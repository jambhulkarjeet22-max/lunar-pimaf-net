from __future__ import annotations
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.ice_characterization_net import IceCharacterizationNet
from src.data.dataset import IceCharacterizationDataset, collate_dict_batch
from src.training.losses import IceCharacterizationLoss
from src.training.metrics import MetricsCalculator
from src.training.config import TrainingConfig
from src.training.checkpoint import save_checkpoint

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = IceCharacterizationNet().to(self.device)
        self.criterion = IceCharacterizationLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.metrics = MetricsCalculator()
        
    def train(self):
        # Using synthetic dataset for training demonstration
        dataset = IceCharacterizationDataset(num_samples=16)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=collate_dict_batch)
        
        self.model.train()
        for epoch in range(self.config.epochs):
            self.metrics.reset()
            total_loss = 0.0
            
            for batch in loader:
                inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                import torch.nn.functional as F
                target_size = targets["purity_percentage"].shape[-2:]
                for k in outputs:
                    if outputs[k].shape[-2:] != target_size:
                        outputs[k] = F.interpolate(
                            outputs[k],
                            size=target_size,
                            mode="bilinear",
                            align_corners=False
                        )
                
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["total_loss"]
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                self.metrics.update(outputs, targets)
                
            avg_loss = total_loss / len(loader)
            epoch_metrics = self.metrics.compute()
            print(f"Epoch {epoch+1}/{self.config.epochs} - Loss: {avg_loss:.4f} - Purity MSE: {epoch_metrics['purity_mse']:.4f}")
            
            # Save checkpoint
            checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(self.model, self.optimizer, epoch + 1, checkpoint_path)
