# LUNAR OS — Physics-Informed Ice Detection Engine

> Research-grade ML architecture for detecting and characterizing subsurface and surface ice deposits on the lunar surface using physics-informed deep learning.

---

## Project Overview

**LUNAR OS** (Lunar Understanding & Neural Analysis for Regolith) is an advanced AI/ML system designed to identify, map, and quantify ice-bearing regions in permanently shadowed regions (PSRs) and other geologically relevant lunar terrains.

The engine combines:

- **Multi-modal remote sensing data** — orbital spectrometry, thermal inertia maps, radar backscatter, and altimetry
- **Physics-informed neural networks (PINNs)** — embedding thermodynamic and regolith transport constraints directly into the learning objective
- **Scalable MLOps practices** — reproducible pipelines, configuration-driven experiments, and modular source architecture

This repository contains the **project scaffold only**. Model implementations, data pipelines, and training logic are planned for subsequent development phases.

---

## Architecture Principles

The codebase follows **clean architecture** and **separation of concerns**:

| Layer | Responsibility |
|---|---|
| `src/data/` | Ingestion, validation, and versioning of raw and external datasets |
| `src/features/` | Feature extraction, spectral indices, and physics-derived descriptors |
| `src/models/` | Model definitions, PINN architectures, and inference wrappers |
| `src/training/` | Training loops, loss composition, and checkpoint management |
| `src/evaluation/` | Metrics, calibration analysis, and geospatial validation |
| `src/utils/` | Shared logging, configuration loading, and I/O helpers |

Top-level entry points (`train.py`, `predict.py`) orchestrate the pipeline without embedding business logic, keeping the core modules independently testable and reusable.

---

## Folder Structure

```
ML_MODEL/
│
├── data/
│   ├── raw/              # Unmodified source data (LRO, LCROSS, Diviner, etc.)
│   ├── processed/        # Cleaned, tiled, and normalized training artifacts
│   └── external/         # Third-party reference datasets and physics priors
│
├── notebooks/            # Exploratory analysis and research prototypes
│
├── src/
│   ├── data/             # Data loading, validation, and split logic
│   ├── features/         # Feature engineering and physics-derived inputs
│   ├── models/           # Neural architectures and PINN components
│   ├── training/         # Training orchestration and optimization
│   ├── evaluation/       # Metrics, uncertainty quantification, reporting
│   └── utils/            # Cross-cutting utilities (config, logging, paths)
│
├── configs/              # Hydra/YAML experiment and environment configs
├── saved_models/         # Serialized model checkpoints (git-ignored)
├── logs/                 # Training and inference logs (git-ignored)
├── tests/                # Unit and integration tests
│
├── requirements.txt      # Python dependencies (placeholder — pin before deploy)
├── train.py              # Training CLI entry point
├── predict.py            # Inference CLI entry point
├── README.md
├── .gitignore
└── .env.example          # Environment variable template
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or later
- Git
- (Recommended) CUDA-capable GPU for training workloads

### 1. Clone the repository

```bash
git clone <repository-url>
cd ML_MODEL
```

### 2. Create and activate a virtual environment

See [Virtual Environment Instructions](#virtual-environment-instructions) below.

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your local paths and credentials
```

### 4. Install dependencies

Uncomment and pin packages in `requirements.txt`, then install:

```bash
pip install -r requirements.txt
```

### 5. Verify the scaffold

```bash
python train.py    # Expected: NotImplementedError (pipeline not yet built)
python predict.py  # Expected: NotImplementedError (pipeline not yet built)
```

---

## Virtual Environment Instructions

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

### Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Deactivate

```bash
deactivate
```

> **Note:** The `.venv/` directory is excluded from version control. Each developer maintains their own local environment.

---

## Development Workflow (Planned)

```
Raw Data  →  src/data  →  src/features  →  src/models
                                              ↓
                         src/evaluation  ←  src/training
                                              ↓
                                        saved_models/
```

1. Place raw lunar datasets in `data/raw/`
2. Define preprocessing and feature configs in `configs/`
3. Implement modules under `src/` following the layer boundaries above
4. Run experiments via `python train.py --config configs/<experiment>.yaml`
5. Evaluate and export artifacts to `saved_models/`

---

## Future Roadmap

### Phase 1 — Data Foundation
- [ ] Define data schemas for LRO Diviner, Mini-RF, and LAMP products
- [ ] Implement ingestion pipelines with checksum validation and versioning
- [ ] Establish train/validation/test splits with spatial holdout for PSR regions

### Phase 2 — Feature Engineering
- [ ] Spectral index computation (water band ratios, thermal emissivity)
- [ ] Physics-derived features: thermal diffusion proxies, albedo–temperature coupling
- [ ] Geospatial tiling and coordinate reference system normalization

### Phase 3 — Model Development
- [ ] Baseline segmentation model (U-Net / SegFormer on multi-band rasters)
- [ ] Physics-informed loss terms: Stefan–Boltzmann consistency, latent heat constraints
- [ ] Uncertainty quantification via Monte Carlo dropout or deep ensembles

### Phase 4 — Training & Evaluation
- [ ] Configuration-driven training with MLflow experiment tracking
- [ ] Geospatial cross-validation respecting lunar tile boundaries
- [ ] Calibration curves and false-positive analysis for mission-critical regions

### Phase 5 — Deployment & MLOps
- [ ] ONNX/TorchScript export for edge inference on lunar mission hardware
- [ ] CI/CD pipeline with automated tests and model registry integration
- [ ] Inference API for integration with LUNAR OS mission planning modules

---

## License

TBD — specify license before public release.

## Contact

LUNAR OS Research Team — ice-detection@lunar-os.dev *(placeholder)*
