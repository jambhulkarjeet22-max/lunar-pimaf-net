# Model 2: Ice Characterization Architecture Report

## 1. Folder Structure
The module follows a highly structured, scalable architecture separated into functional namespaces:
```
Model_2_Ice_Characterization/
├── src/
│   ├── data/        # Data handling and batch collation (IceCharacterizationDataset)
│   ├── models/      # Neural network logic (encoders, fusion, heads)
│   ├── training/    # Training loops, loss functions, metrics, configurations
│   ├── evaluation/  # Evaluation scripts and metric pipelines
│   └── features/    # Feature engineering logic
├── tests/           # Dedicated smoke tests (model, train, predict)
├── configs/         # Configurations and hyperparameters
├── docs/            # Architecture reports
├── train.py         # Entry point for training execution
├── predict.py       # Entry point for inference execution
└── requirements.txt # Python dependencies
```

## 2. Neural Network Architecture
The primary architecture is defined within `IceCharacterizationNet`. It is composed of three interconnected sub-networks:
1. **Multi-Modal Encoder**: Independent convolutional pipelines (`ModalityEncoder`) applied parallelly to each instrument data stream. The encoders downsample the spatial resolution by a factor of 2.
2. **Attention Fusion**: A spatial attention mechanism (`AttentionFusion`) that dynamically weights the 6 fused modalities based on spatial relevance, generating a dense representation.
3. **Multi-Task Heads**: A branched convolutional decoder (`MultiTaskHeads`) terminating in specialized activation functions for diverse multi-task predictions.

## 3. Input Modalities
The network is designed to simultaneously ingest spatial feature maps from multiple lunar instruments:
- **Mini-RF radar**: 3 channels
- **Diviner thermal**: 1 channel
- **LOLA topography**: 1 channel
- **LEND neutron**: 1 channel
- **LAMP UV**: 1 channel
- **M3 spectral hydration**: 2 channels

## 4. Output Heads
The model estimates 5 critical attributes simultaneously:
- **Ice Purity (%)**: Regression head utilizing `Sigmoid` activation.
- **Ice Depth (meters)**: Regression head utilizing `Softplus` activation to guarantee strictly positive physical depth values.
- **Ice Type Classification**: 3-class categorical classification (Surface, Subsurface, Mixed).
- **Ice Stability Score**: Regression head utilizing `Sigmoid` activation (0-1 range).
- **Confidence Score**: Regression head predicting uncertainty/confidence via `Sigmoid` (0-1 range).

## 5. Training Pipeline
The central `Trainer` orchestrates the lifecycle:
- Defines an explicit initialization sequence wrapping `optim.Adam` and `DataLoader`.
- Features an automated upsampling block mapping lower-resolution model decoder predictions back to the target spatial size `(H, W)` using bilinear interpolation.
- Managed by a dynamic `TrainingConfig` dataclass routing logs and checkpoints.

## 6. Loss Functions
The custom `IceCharacterizationLoss` merges heterogeneous objectives using a weighted sum formula:
- Uses `MSELoss` for purity, depth, and stability.
- Uses `CrossEntropyLoss` for the ice type classification.
- **Physics-Aware Constraint**: Explicitly penalizes the network if it predicts large values for `ice_depth` while simultaneously classifying the pixel as `Surface Ice` with high probability, embedding physical priors directly into the gradient landscape.

## 7. Metrics
The `MetricsCalculator` object logs and aggregates batch statistics across epochs, currently tracking:
- **Purity MSE**: `F.mse_loss` over purity predictions.
- **Depth MSE**: `F.mse_loss` over depth targets.
- **Type Accuracy**: Categorical argmax equality computation.

## 8. Inference Pipeline
`InferencePipeline` manages seamless state reloading and execution via `torch.no_grad()`. It automatically applies `Softmax` over the classification logits to output structured probability distributions alongside standard tensor predictions.

## 9. Smoke Test Status
- `run_model_smoke.py`: **PASSED** (Model instantiation and synthetic forward pass).
- `run_train_smoke.py`: **PASSED** (Synthetic epoch, interpolation testing, loss convergence).
- `run_predict_smoke.py`: **PASSED** (Save dummy state, reload checkpoint, infer on synthetic map).

## 10. Files Created
- `src/models/encoder.py`, `fusion.py`, `heads.py`, `ice_characterization_net.py`, `__init__.py`
- `src/data/dataset.py`, `__init__.py`
- `src/training/losses.py`, `metrics.py`, `trainer.py`, `config.py`, `checkpoint.py`, `inference.py`, `__init__.py`
- `src/evaluation/__init__.py`, `src/features/__init__.py`
- `tests/run_model_smoke.py`, `run_train_smoke.py`, `run_predict_smoke.py`, `__init__.py`
- `train.py`, `predict.py`, `requirements.txt`, `README.md`

## 11. Remaining Limitations
1. **Synthetic Data Dependency**: The pipeline currently relies on randomized data tensors via `torch.randn`. Requires integration with the root-level `shared.dataset_utils` and physical data loaders to read actual GeoTIFF assets.
2. **Device Scalability**: Hardcoded to simple `cpu`/`cuda` fallback logic. Needs expansion into Multi-GPU via Distributed Data Parallel (DDP) for scalable training on dense orbital datasets.
3. **Telemetry & Logging**: The pipeline relies purely on terminal `print` streams. Integration with structured experiment tracking software like Weights & Biases (W&B) or TensorBoard is required for production visibility.
