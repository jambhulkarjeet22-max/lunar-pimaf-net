from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for training the Ice Characterization Net."""
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    device: str = "cpu" # Default for smoke tests
    output_dir: str = "logs/"
    checkpoint_dir: str = "saved_models/"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
