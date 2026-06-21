# Model 2: Ice Characterization

This module implements a complete deep learning pipeline for lunar ice characterization using multi-modal sensor fusion.

## Features
- Multi-modal CNN Encoder (Mini-RF, Diviner, LOLA, LEND, LAMP, M3)
- Attention Fusion Network
- Multi-Task Heads:
  - Purity Percentage (0-100%)
  - Ice Depth (meters)
  - Ice Type (Surface, Subsurface, Mixed)
  - Stability Score (0-1)
  - Confidence Score

## Getting Started

To train on synthetic data for a quick test:
```bash
python train.py
```

To run smoke tests:
```bash
python tests/run_smoke_tests.py
```

To run inference:
```bash
python predict.py --checkpoint saved_models/checkpoint_epoch_2.pt
```
