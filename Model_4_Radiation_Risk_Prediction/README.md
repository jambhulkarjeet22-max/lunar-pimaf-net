# Model 4 — Lunar Radiation Risk Prediction AI System

This package implements a production-ready PyTorch deep learning pipeline to predict lunar surface radiation exposure, shielding effectiveness, and habitat safety.

## Features

- **Multi-Modal Spatial Encoders**: Independent 2D CNN encoders for 7 orbital and terrain modalities.
- **Attention-Based Fusion**: Fuses encoded features using a learned spatial/channel attention bottleneck.
- **Multi-Task Prediction Heads**: Simultaneously predicts dose rate, risk score, shielding effectiveness, habitat safety, and a final hazard map.
- **Physics-Aware Regularization**: Combines supervised losses with physical constraints coupling regolith thickness, shadow regions, topography, and habitat safety.
- **Mixed-Precision Training**: Full support for mixed-precision (AMP) training.
- **Training and Inference Pipelines**: Command-line entry points for end-to-end model training, checkpoint saving/loading, early stopping, and metric calculation.

## Requirements

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

## Quick Start

### Model Training
To train the model from the repository root:
```bash
python Model_4_Radiation_Risk_Prediction/train.py --epochs 5 --num-samples 64 --batch-size 4
```

### Model Inference
To run inference and export prediction maps:
```bash
python Model_4_Radiation_Risk_Prediction/predict.py --checkpoint Model_4_Radiation_Risk_Prediction/checkpoints/best.pt --num-samples 8
```

## Code Layout

- `src/models/`: Neural network layers (`radiation_encoder.py`, `fusion.py`, `heads.py`, `radiation_net.py`).
- `src/data/`: Dataset classes and synthetic physically correlated data generation (`dataset.py`).
- `src/training/`: Training and evaluation pipeline logic (`losses.py`, `metrics.py`, `trainer.py`, `config.py`, `checkpoint.py`, `inference.py`).
- `tests/`: Smoke tests verifying the network, train loops, and inference pipeline (`run_model_smoke.py`, `run_train_smoke.py`, `run_predict_smoke.py`).
- `docs/`: Technical architectural details and schemas (`docs/ARCHITECTURE.md`).
