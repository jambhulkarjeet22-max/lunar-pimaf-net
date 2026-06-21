# Model 5 — Lunar Rover Hazard & Navigation Intelligence

This package implements a production-ready PyTorch deep learning pipeline to predict safe rover traversability, crater/boulder/slope hazards, and navigation cost maps on the lunar surface.

## Features

- **Multi-Modal Terrain Encoders**: Spatial CNN encoders for 7 input modalities (`lola`, `mini_rf`, `dem`, `slope`, `crater`, `boulder`, `illumination`).
- **Spatial Attention Fusion**: Combines features using learned attention weights to focus on the most hazardous features.
- **Multi-Task Scoring**: Predicts traversability score, crater hazard, boulder hazard, slope hazard, navigation cost, and a final safety score.
- **Slope & Boulder Physics Loss**: Custom constraints penalizing unsafe rover traversability on steep slopes or high boulder densities.
- **High-Performance Training**: Mixed-precision (AMP) support, gradient clipping, early stopping, and automatic dataloading.
- **Standardized Exports**: Exports NumPy arrays for all scored maps and detailed JSON run summaries.

## Requirements

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

## Quick Start

### Model Training
To train the model from the repository root:
```bash
python Model_5_Rover_Hazard_Navigation/train.py --epochs 5 --num-samples 64 --batch-size 4
```

### Model Inference
To run inference and export navigation maps:
```bash
python Model_5_Rover_Hazard_Navigation/predict.py --checkpoint Model_5_Rover_Hazard_Navigation/checkpoints/best.pt --num-samples 8
```

## Code Layout

- `src/models/`: Neural network layers (`terrain_encoder.py`, `fusion.py`, `heads.py`, `rover_navigation_net.py`).
- `src/data/`: Dataset loading and physical labels correlation logic (`dataset.py`).
- `src/training/`: Training and prediction pipelines (`losses.py`, `metrics.py`, `trainer.py`, `config.py`, `checkpoint.py`, `inference.py`).
- `tests/`: Smoke tests checking model layers, trainer optimization, and inference map export (`run_model_smoke.py`, `run_train_smoke.py`, `run_predict_smoke.py`).
- `docs/`: Technical architectural details and schemas (`docs/ARCHITECTURE.md`).
