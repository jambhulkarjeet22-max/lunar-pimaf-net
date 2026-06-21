# Model 3 — Landing Site Intelligence

Production-ready PyTorch system for predicting safe lunar landing zones from multi-modal orbital data.

## Inputs

| Modality | Channels | Description |
|----------|----------|-------------|
| `lola` | 1 | LOLA topography (DEM) |
| `mini_rf` | 3 | Mini-RF radar (CPR / SC / OC) |
| `diviner` | 1 | Diviner surface temperature |
| `lend` | 1 | LEND hydrogen abundance |
| `illumination` | 1 | Illumination map |
| `psr` | 1 | Permanently shadowed region map |

## Outputs

All outputs are per-pixel maps in `[0, 1]`:

1. **landing_safety_score** — Landing Safety Score
2. **hazard_probability** — Hazard Probability
3. **illumination_score** — Illumination Score
4. **resource_accessibility_score** — Resource Accessibility Score
5. **final_suitability_score** — Final Landing Suitability Score

## Architecture

- **MultiModalTerrainEncoder** — Independent CNN encoders per modality
- **AttentionFusion** — Learned modality weighting and feature fusion
- **LandingHeads** — Multi-task sigmoid prediction heads

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design report.

## Installation

From the repository root:

```bash
pip install -r Model_3_Landing_Site_Intelligence/requirements.txt
```

## Training

```bash
python Model_3_Landing_Site_Intelligence/train.py \
  --epochs 5 \
  --batch-size 4 \
  --num-samples 64 \
  --patch-size 64 \
  --checkpoint-dir Model_3_Landing_Site_Intelligence/checkpoints
```

Checkpoints are written to `checkpoints/` (`best.pt` when validation is enabled).

## Inference

```bash
python Model_3_Landing_Site_Intelligence/predict.py \
  --checkpoint Model_3_Landing_Site_Intelligence/checkpoints/best.pt \
  --output-dir Model_3_Landing_Site_Intelligence/predictions
```

Predictions are exported as NumPy arrays plus `summary.json`.

## Smoke Tests

```bash
python Model_3_Landing_Site_Intelligence/tests/run_model_smoke.py
python Model_3_Landing_Site_Intelligence/tests/run_train_smoke.py
python Model_3_Landing_Site_Intelligence/tests/run_predict_smoke.py
```

## Physics-Aware Loss

`LandingSiteLoss` combines MSE/BCE multi-task supervision with a **slope penalty** that discourages high safety predictions on steep terrain (slope magnitude above a configurable threshold).

## Project Layout

```
Model_3_Landing_Site_Intelligence/
├── src/
│   ├── models/          # Encoders, fusion, heads, network
│   ├── data/            # Synthetic dataset and collate
│   └── training/        # Losses, metrics, trainer, inference
├── tests/               # Smoke tests
├── docs/                # Architecture report
├── train.py
├── predict.py
└── requirements.txt
```
