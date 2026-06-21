# LUNAR OS — Multi-Model Lunar Intelligence Platform

This repository hosts six specialized lunar ML models plus shared cross-cutting utilities.

## Repository layout

```
Model_1_Physics_Informed_Ice_Detection/   # Physics-informed ice detection (LUNAR-PIMAF-Net)
Model_2_Ice_Characterization/
Model_3_Landing_Site_Intelligence/
Model_4_Radiation_Risk_Prediction/
Model_5_Rover_Hazard_Navigation/
Model_6_ORACLE_Mission_Copilot/
shared/                                    # Cross-model utilities
LICENSE
.gitignore
```

## Running Model 1

From the Model 1 directory:

```powershell
cd Model_1_Physics_Informed_Ice_Detection
$env:PYTHONPATH=".;.."
pip install -r requirements.txt
python -m tests.run_data_smoke
python -m tests.run_model_smoke
python -m tests.run_train_smoke
python -m tests.run_predict_smoke
python train.py --synthetic-samples 64 --epochs 1
python predict.py --checkpoint saved_models/experiment/best.pt --synthetic-samples 1
```

Set `PYTHONPATH` to the model directory (`.`) and repository root (`..`) so `src.*` and `shared.*` imports resolve.

## Shared utilities

Import from repository root on `PYTHONPATH`:

```python
from shared.lunar_constants import POLAR_CRS, pole_to_epsg
from shared.geospatial_utils import validate_crs
from shared.dataset_utils import ensure_import_paths
```

## Models 2–6

Scaffold packages are in place with standard `src/`, `tests/`, `configs/`, `data/`, and `docs/` folders. Implementations are pending.
