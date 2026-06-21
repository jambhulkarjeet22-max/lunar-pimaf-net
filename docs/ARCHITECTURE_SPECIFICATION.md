# LUNAR OS — Deep Learning Architecture Specification

**Physics-Informed Ice Detection Engine (PI-IDE)**  
**Model codename:** `LUNAR-PIMAF-Net` (Physics-Informed Multi-sensor Attention Fusion Network)  
**Document version:** 1.0  
**Classification:** Research architecture specification — no implementation code  
**Companion document:** `DATA_PREPROCESSING_PIPELINE.md`

---

## 1. Mission Statement

`LUNAR-PIMAF-Net` predicts **per-pixel lunar ice probability** — decomposed into surface/volatile ice and subsurface ice classes — from a fused 59-channel orbital observation tensor. The architecture is designed for **spatial crater holdout evaluation**, **multi-sensor missingness**, and **mission-grade uncertainty quantification**.

**Primary output:**

```
P_subsurface_ice(x, y) ∈ [0, 1]   — mission decision variable
P_surface_ice(x, y)   ∈ [0, 1]
σ_epistemic(x, y)     ∈ [0, 1]   — model uncertainty (lack of evidence)
σ_aleatoric(x, y)     ∈ [0, 1]   — data/sensor ambiguity
conf(x, y)            ∈ [0, 1]   — composite prediction confidence
```

---

## 2. Architecture Selection Analysis

### 2.1 Candidate Comparison

| Architecture | Strengths for Lunar Ice | Weaknesses | Verdict |
|--------------|------------------------|------------|---------|
| **Pure CNN (ResNet/EfficientNet)** | Strong local texture/slope/radar feature extraction; proven on SAR/thermal RS | No explicit multi-scale context; poor long-range PSR-crater relationships | **Component only** |
| **U-Net** | Pixel-level segmentation; skip connections preserve crater rim geometry; standard in planetary RS | Single encoder treats all 59 channels identically — ignores sensor physics | **Decoder backbone** |
| **Vision Transformer (ViT/Swin)** | Global context; cross-patch relationships | Requires large labeled datasets; weak inductive bias for 240 m geophysics; overfits under crater holdout with weak labels | **Not recommended as sole backbone** |
| **Hybrid (Modality CNN + Attention + U-Net)** | Sensor-specific encoders respect physics; attention fuses cross-modal evidence; U-Net preserves spatial detail; physics layer injects domain constraints | Higher design complexity | **Recommended** |

### 2.2 Final Recommendation

**`LUNAR-PIMAF-Net` — Hybrid Modality-Encoded Attention U-Net with Physics Constraint Layer**

This architecture is selected because:

1. **Six sensors have incompatible statistics and missingness patterns** — grouped encoders are scientifically necessary, not optional.
2. **Ice signatures span multiple spatial scales** — crater-scale context (10–30 km) vs. pixel-scale roughness (240 m) requires multi-scale FPN + attention.
3. **Training labels are weak and spatially correlated** — physics-informed losses regularize against overfitting to CPR roughness artifacts.
4. **Crater holdout demands calibrated uncertainty** — evidential deep learning provides principled epistemic/aleatoric decomposition without Monte Carlo at inference (optional MC Dropout for validation).

**Target performance envelope (spatial crater holdout, weak-label agreement):**

| Metric | Target |
|--------|--------|
| Overall pixel accuracy | ≥ 90% |
| Macro-F1 (3-class) | ≥ 0.82 |
| Subsurface ice recall (class 2) | ≥ 0.78 |
| Cabeus LCROSS region recall | ≥ 0.80 |
| Expected Calibration Error (ECE) | ≤ 0.05 |

---

## 3. System Overview

### 3.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: X ∈ ℝ^(59×128×128)  +  M_valid ∈ {0,1}^(6×128×128)  [modality masks] │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  §4  INPUT LAYER — Modality Partitioning & Embedding                        │
│  7 sensor-group slices → 7 initial embedding maps ∈ ℝ^(64×128×128)          │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  §5  BACKBONE — Multi-Scale Modality Encoders + Shared FPN                  │
│  Scale levels: P1(128²) → P2(64²) → P3(32²) → P4(16²) → P5(8²)            │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  §6  FUSION — Cross-Modal Multi-Scale Attention (CMMA)                      │
│  At P3, P4, P5: 7 modality tokens attend across sensors per spatial cell    │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  §7  PHYSICS LAYER — Differentiable Constraint Module (DCM)                   │
│  Stefan residual · Ice stability gate · Radar-roughness consistency         │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  §8  DECODER — Attention-Gated U-Net with FPN Skip Connections              │
│  Upsample P5→P1 with skip fusion from CMMA outputs                          │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  §9  PREDICTION HEADS                                                        │
│  ├─ Segmentation Head      → logits ∈ ℝ^(3×128×128)                         │
│  ├─ Ice Probability Head   → P_ice ∈ ℝ^(2×128×128)  [surface, subsurface] │
│  ├─ Evidential Uncertainty   → Dirichlet α ∈ ℝ^(3×128×128)                  │
│  ├─ Confidence Head          → conf ∈ ℝ^(1×128×128)                         │
│  └─ Physics Violation Head   → v_phys ∈ ℝ^(3×128×128)  [monitoring only]   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Parameter Budget

| Component | Approx. Parameters |
|-----------|-------------------|
| Modality encoders (×7) | 4.2 M |
| Shared FPN + CMMA | 8.6 M |
| Physics Constraint Module | 0.4 M |
| U-Net decoder | 5.8 M |
| Prediction heads | 1.2 M |
| **Total** | **~20.2 M** |

Designed for single-GPU training (≥ 16 GB VRAM) with batch size 8 and mixed precision (FP16).

---

## 4. Input Layer

### 4.1 Modality Partitioning

The 59-channel input is partitioned into **seven scientifically homogeneous groups** aligned with the preprocessing schema. Each group is accompanied by a **modality validity mask** (from preprocessing validity channels, aggregated per group).

| Group ID | Channels (index) | Count | Encoder ID |
|----------|------------------|-------|------------|
| G_topo | 0–7 | 8 | `Enc_topo` |
| G_radar | 8–17 | 10 | `Enc_radar` |
| G_thermal | 18–32 | 15 | `Enc_thermal` |
| G_uv | 33–38 | 6 | `Enc_uv` |
| G_neutron | 39–44 | 6 | `Enc_neutron` |
| G_spectral | 45–51 | 7 | `Enc_spectral` |
| G_physics | 52–58 | 7 | `Enc_phys` |

### 4.2 Group Embedding

Each group `G_k` with `C_k` channels is mapped to a uniform **64-channel latent map** via a 1×1 grouped convolution followed by group normalization:

```
E_k⁽⁰⁾ = GN( ReLU( Conv1×1_{C_k→64}( G_k ) ) )     ∈ ℝ^(64×128×128)
```

**Missingness injection:** Before convolution, invalid pixels within a group are zeroed and a **learned missingness embedding** `η_k ∈ ℝ^64` is added:

```
G_k[i] = G_k[i] + η_k · (1 − mask_k[i])
```

This allows the network to distinguish "measured zero" from "not observed" — critical for M3 in PSR and LEND upsampling artifacts.

### 4.3 Positional Encoding

A **2D sinusoidal positional encoding** is added to `E_topo⁽⁰⁾` only (not all encoders), encoding absolute position within the 30.7 km patch:

```
PE(x, y, 2i)   = sin(x / 10000^(2i/d))
PE(x, y, 2i+1) = cos(y / 10000^(2i/d))
```

Rationale: Topography defines the spatial reference frame; other modalities are co-registered to LOLA.

### 4.4 Input Normalization

Per-group **instance normalization** (not batch normalization) is applied at the input boundary to handle covariate shift between north and south poles:

```
G_k ← InstanceNorm(G_k)
```

---

## 5. Backbone Architecture — Multi-Scale Modality Encoders

### 5.1 Encoder Block Design

Each modality encoder uses a **depthwise-separable ResNet block** stack (computationally efficient, proven on geospatial data):

**ResSepBlock:**
```
Input → DWConv3×3 → GN → ReLU → PWConv1×1 → GN → (+ residual) → ReLU
```

### 5.2 Per-Encoder Depth & Output Channels

| Encoder | Blocks at P1 | Blocks at P2 | Blocks at P3 | Output channels per scale |
|---------|-------------|-------------|-------------|--------------------------|
| `Enc_topo` | 2 | 2 | 2 | 64 / 128 / 256 |
| `Enc_radar` | 2 | 2 | 2 | 64 / 128 / 256 |
| `Enc_thermal` | 3 | 2 | 2 | 64 / 128 / 256 |
| `Enc_uv` | 2 | 1 | 1 | 64 / 128 / 256 |
| `Enc_neutron` | 2 | 1 | 1 | 64 / 128 / 256 |
| `Enc_spectral` | 2 | 2 | 1 | 64 / 128 / 256 |
| `Enc_phys` | 1 | 1 | 1 | 64 / 128 / 256 |

`Enc_thermal` receives the deepest stack (3 blocks at P1) because Diviner channels carry the strongest ice viability signal.

### 5.3 Shared Feature Pyramid Network (FPN)

After modality encoding, features at each scale are **concatenated** (7 × 256 = 1792 channels at P3/P4/P5) and compressed through a shared lateral connection:

```
FPN_l = Conv1×1_{1792→256}( Concat( E_k⁽ˡ⁾ ) )     l ∈ {3, 4, 5}
```

Top-down pathway with nearest-neighbor upsampling and element-wise addition (standard FPN; Lin et al. 2017):

```
FPN_4 = FPN_4 + Upsample(FPN_5)
FPN_3 = FPN_3 + Upsample(FPN_4)
```

**Multi-scale receptive fields at 240 m:**

| Level | Spatial size | Receptive field (approx.) | Captures |
|-------|-------------|--------------------------|----------|
| P1 | 128 × 128 | 240 m – 1.9 km | Pixel texture, slope facets |
| P2 | 64 × 64 | 1.9 – 3.8 km | Crater wall segments |
| P3 | 32 × 32 | 3.8 – 7.7 km | Crater floor PSR |
| P4 | 16 × 16 | 7.7 – 15.4 km | Full crater structure |
| P5 | 8 × 8 | 15.4 – 30.7 km | Full patch / multi-crater context |

---

## 6. Fusion Strategy — Cross-Modal Multi-Scale Attention (CMMA)

### 6.1 Motivation

Simple concatenation treats a 110 K Diviner pixel and a CPR = 1.8 Mini-RF pixel as independent features. Ice detection requires **conditional fusion**: high CPR is only informative when thermal stability confirms cold-trap conditions.

CMMA implements this via **cross-attention between modality tokens** at coarser scales (P3–P5) where LEND upsampling noise is averaged out.

### 6.2 Modality Token Construction

At scale level `l`, for each spatial position `(i, j)`:

```
t_k⁽ˡ⁾(i,j) = GAP( E_k⁽ˡ⁾(i,j) )     ∈ ℝ^256     (global average pool within 3×3 neighborhood)
```

Seven tokens per spatial cell form a **modality sequence** `T⁽ˡ⁾(i,j) ∈ ℝ^(7×256)`.

### 6.3 Cross-Modal Attention Block

Two-layer cross-attention with pre-norm (adapted from Perceiver IO):

```
T' = T + MHA(LN(T), LN(T), LN(T))          # self-attention across modalities
T'' = T' + FFN(LN(T'))                      # feed-forward, dim 256→512→256
```

**Thermal-guided attention bias:** A learned bias matrix `B ∈ ℝ^(7×7)` encodes prior modality coupling from planetary science literature:

|  | topo | radar | thermal | uv | neutron | spectral | phys |
|--|------|-------|---------|-----|---------|----------|------|
| **thermal** | 0.3 | 0.8 | — | 0.6 | 0.9 | 0.2 | 1.0 |
| **neutron** | 0.2 | 0.7 | 0.9 | 0.3 | — | 0.1 | 0.9 |
| **radar** | 0.9 | — | 0.8 | 0.2 | 0.7 | 0.1 | 0.8 |

Bias is added to attention logits before softmax: `A = softmax(QK^T / √d + B)`.

### 6.4 Gated Fusion Output

Fused feature at each scale:

```
F⁽ˡ⁾ = Conv1×1( Concat_k( Upsample( T_k⁽ˡ⁾ ) ) )     ∈ ℝ^(256×H_l×W_l)
```

A **learnable gate** `g ∈ [0,1]` blends FPN and CMMA paths:

```
F̃⁽ˡ⁾ = g · F⁽ˡ⁾_CMMA + (1 − g) · F⁽ˡ⁾_FPN
```

Initialized at `g = 0.5`; expected to converge to `g ≈ 0.7` (attention-dominant).

### 6.5 Scale Selection for CMMA

| Scale | CMMA applied? | Rationale |
|-------|--------------|-----------|
| P1, P2 | No — FPN only | Preserve fine spatial detail; attention too expensive at 128² |
| P3, P4, P5 | Yes | Crater-scale fusion; 32² + 16² + 8² = 1,344 cells (tractable) |

---

## 7. Physics Layer — Differentiable Constraint Module (DCM)

### 7.1 Design Philosophy

The DCM is not a post-processing filter. It is a **differentiable mid-network constraint** that:

1. Projects latent features onto a physically interpretable subspace.
2. Computes **analytic physics residuals** from predicted latent variables.
3. Feeds corrected features forward to the decoder.
4. Exposes residuals to the physics violation head and loss function.

This follows the Physics-Informed Neural Network (PINN) paradigm adapted for discriminative segmentation (Raissi et al. 2019; Karniadakis et al. 2021).

### 7.2 Latent Physics Variables

From fused features `F̃⁽⁴⁾`, a projection head extracts interpretable latent fields:

| Latent variable | Symbol | Dimension | Physical meaning |
|-----------------|--------|-----------|------------------|
| Surface temperature | `T̂_max` | 1 × H × W | Predicted annual max bolometric T (K) |
| Thermal emissivity | `ε̂` | 1 × H × W | Surface emissivity [0.9, 1.0] |
| Radar dielectric | `ε̂_r` | 1 × H × W | Effective dielectric constant |
| Hydrogen fraction | `ĥ` | 1 × H × W | Mass fraction hydrogen proxy |
| Ice stability | `ŝ` | 1 × H × W | Predicted stability score [0, 1] |

Projection: ` [T̂_max, ε̂, ε̂_r, ĥ, ŝ] = σ( Conv1×1_{256→5}( F̃⁽⁴⁾) ) ` with appropriate activations (ReLU for T, sigmoid for others).

### 7.3 Physics Residual Computations

**R1 — Stefan-Boltzmann energy balance:**

```
R_stefan = |T̂_max⁴ − (1−Â) · Î / (ε̂ · σ_SB)|     normalized to [0,1]
```

Where `Î` (insolation) and `Â` (albedo proxy from LAMP/M3) are read from input channels; `σ_SB = 5.67×10⁻⁸`.

**R2 — Ice stability gate (thermodynamic feasibility):**

```
R_stability = ReLU(T̂_max − T_trap) · P_ice_subsurface
```

Where `T_trap = 110 K` for H₂O; penalizes subsurface ice predictions in thermally unstable regions.

**R3 — Radar dielectric consistency:**

```
R_radar = |ε̂_r − f(ε_ice, ε_regolith, p_ice)| · (1 − roughness_mask)
```

Where `f` is the Maxwell-Garnett mixing model; `p_ice` is inferred ice fraction from `ĥ`; `roughness_mask` from input slope > 10°.

**R4 — Neutron-hydrogen coupling:**

```
R_neutron = |ĥ − h_LEND|² · mask_LEND_valid
```

Penalizes deviation from observed LEND hydrogen where LEND is trustworthy.

### 7.4 Feature Correction

Physics residuals are concatenated and passed through a correction gate:

```
F_phys = F̃⁽⁴⁾ + Conv1×1( [F̃⁽⁴⁾ ; R_stefan ; R_stability ; R_radar ; R_neutron] )
```

Corrected features `F_phys` feed the decoder; residuals feed the physics violation head.

### 7.5 Physics Bypass for Input Channels

The 7 input physics channels (indices 52–58) are **not re-learned** but used as **anchor targets** in the physics loss (§10.2), preventing the DCM from drifting from pre-computed geophysical priors.

---

## 8. Decoder — Attention-Gated U-Net

### 8.1 Architecture

Standard U-Net topology with **attention gates** (Oktay et al. 2018) on skip connections from CMMA/FPN features:

```
Decoder block: Upsample×2 → Conv3×3 → GN → ReLU → Conv3×3 → GN → ReLU
Skip fusion:   AttentionGate(decoder_feature, encoder_skip) → Concat → Conv1×1
```

### 8.2 Decoder Path

| Stage | Input | Skip from | Output size | Channels |
|-------|-------|-----------|-------------|----------|
| D5 | F_phys (P5) | — | 8 × 8 | 256 |
| D4 | D5 ↑ | F̃⁽⁴⁾ | 16 × 16 | 128 |
| D3 | D4 ↑ | F̃⁽³⁾ | 32 × 32 | 128 |
| D2 | D3 ↑ | FPN_2 | 64 × 64 | 64 |
| D1 | D2 ↑ | FPN_1 | 128 × 128 | 64 |

### 8.3 Attention Gate

For decoder signal `x` and skip `g`:

```
α = σ( Conv1×1( ReLU( W_x·x + W_g·g ) ) )
output = α · g
```

Gates suppress skip features from sunlit crater walls when decoding PSR floor ice — reducing false positives from terrain edges.

---

## 9. Prediction Heads

All heads branch from the D1 decoder output `D1 ∈ ℝ^(64×128×128)`.

### 9.1 Segmentation Head (Primary)

```
logits_seg = Conv1×1_{64→3}( D1 )     ∈ ℝ^(3×128×128)
```

Three classes: {0: no ice, 1: surface ice, 2: subsurface ice}.

### 9.2 Ice Probability Head (Mission Output)

Dedicated sigmoid head for operational probability maps (decoupled from softmax to allow overlapping surface + subsurface hypotheses):

```
P_ice = σ( Conv1×1_{64→2}( D1 ) )     ∈ ℝ^(2×128×128)
```

- Channel 0: `P_surface_ice`
- Channel 1: `P_subsurface_ice` ← **primary mission output**

### 9.3 Evidential Uncertainty Head

Models second-order probability via **Dirichlet distribution** over 3 classes (Sensoy et al. 2018):

```
α = Softplus( Conv1×1_{64→3}( D1 ) ) + 1     ∈ ℝ^(3×128×128),  α_k > 1
```

Derived quantities:

| Quantity | Formula | Interpretation |
|----------|---------|----------------|
| Expected probability | `p_k = α_k / S` | `S = Σα_k` |
| Total uncertainty | `u = K / S` | K=3; high when evidence is low |
| Epistemic (vacuity) | `u_epist = 3 / S` | Lack of evidence / distributional uncertainty |
| Aleatoric (discord) | `u_alea = −Σ p_k(log p_k − ψ(α_k+1) + ψ(S+1))` | Data ambiguity / sensor conflict |
| Predicted class | `ŷ = argmax(p_k)` | Point estimate |

### 9.4 Confidence Head

Composite confidence independent of the evidential head, trained with explicit confidence supervision from preprocessing label confidence `conf` (see DATA_PREPROCESSING_PIPELINE §9.5):

```
conf_pred = σ( Conv1×1_{64→1}( D1 ) )     ∈ ℝ^(1×128×128)
```

**Confidence definition (inference):**

```
conf = (1 − u_epist) · (1 − mean(R_phys)) · mask_valid
```

Where `mean(R_phys)` is the mean of the four physics residuals at that pixel.

### 9.5 Physics Violation Head (Monitoring)

Exposes DCM residuals as explicit spatial maps for mission operators:

```
v_phys = [R_stefan ; R_stability ; R_radar ; R_neutron]     ∈ ℝ^(4×128×128)
```

Not used for gradient descent directly (redundant with physics loss) but logged during inference for interpretability.

---

## 10. Loss Functions

### 10.1 Total Loss

```
L_total = L_seg + λ₁·L_evidential + λ₂·L_physics + λ₃·L_prob + λ₄·L_conf + λ₅·L_smooth
```

| Term | Weight | Purpose |
|------|--------|---------|
| `L_seg` | 1.0 | Primary segmentation against weak labels |
| `L_evidential` | 0.5 | Calibrated uncertainty |
| `L_physics` | 0.3 | Domain constraint enforcement |
| `L_prob` | 0.4 | Direct ice probability supervision |
| `L_conf` | 0.2 | Confidence calibration |
| `L_smooth` | 0.05 | Spatial regularization |

### 10.2 Segmentation Loss — Focal Soft Cross-Entropy

Weak labels use soft probability vectors `y_soft ∈ [0,1]^3` and per-pixel confidence `w_conf`:

```
L_seg = −(1/N) Σ_{i} w_conf(i) · w_class(i) · Σ_k y_soft_k(i) · (1 − p_k(i))^γ · log(p_k(i))
```

- `p_k` from softmax(logits_seg)
- `γ = 2` (focal exponent — down-weights easy background pixels)
- `w_class = [1.0, 2.5, 4.0]` for classes {0, 1, 2} — up-weights rare subsurface ice

Pixels with `w_conf < 0.4` are excluded from loss computation.

### 10.3 Evidential Loss

Type-II maximum likelihood for Dirichlet (Sensoy et al. 2018):

```
L_evidential = Σ_i [ log(S_i) − log(α_y(i), i) + Σ_k≠y α_k(i) · (log(α_y(i), i) − log(α_k(i))) ]
```

Plus **KL divergence regularizer** toward uniform Dirichlet (prevents overconfident evidence):

```
L_KL = KL( Dir(α) || Dir(1) )
```

Annealed during training: `λ_KL = min(1.0, epoch / 50)`.

### 10.4 Physics-Informed Loss

```
L_physics = μ₁·R̄_stefan + μ₂·R̄_stability + μ₃·R̄_radar + μ₄·R̄_neutron + μ₅·L_anchor
```

| Sub-term | Weight | Description |
|----------|--------|-------------|
| `R̄_stefan` | 1.0 | Mean Stefan residual |
| `R̄_stability` | 2.0 | Ice-in-warm-region penalty (critical) |
| `R̄_radar` | 0.8 | Dielectric consistency |
| `R̄_neutron` | 0.6 | Hydrogen coupling |
| `L_anchor` | 1.5 | MSE between DCM latents and input physics channels 52–58 |

`L_anchor` prevents the network from learning physics violations that happen to fit weak labels.

### 10.5 Ice Probability Loss

Binary cross-entropy on subsurface and surface channels against soft label marginals:

```
L_prob = BCE(P_subsurface, y_soft_2) + 0.5 · BCE(P_surface, y_soft_1)
```

### 10.6 Confidence Loss

```
L_conf = MSE(conf_pred, conf_target) + BCE(conf_pred, 𝟙[correct_prediction])
```

### 10.7 Spatial Smoothness Loss

Total variation penalty on subsurface probability within PSR interiors (not at terrain boundaries):

```
L_smooth = Σ_i |∇P_subsurface(i)| · mask_PSR_interior(i)
```

Prevents salt-and-pepper noise in neutron-dominated regions.

---

## 11. Uncertainty Modeling — Full Specification

### 11.1 Uncertainty Taxonomy

| Type | Source | Model mechanism | Mission use |
|------|--------|-----------------|-------------|
| **Epistemic** | Lack of training evidence; novel terrain | Dirichlet vacuity `u_epist`; optional MC Dropout (p=0.1, T=10 forward passes) | Flag low-confidence predictions for human review |
| **Aleatoric** | Inherent sensor ambiguity (CPR vs roughness; LEND resolution) | Dirichlet discord `u_alea`; modality missingness masks | Weight decision thresholds |
| **Physics uncertainty** | Thermodynamic inconsistency | DCM residuals `R_*` | Reject physically impossible detections |

### 11.2 Inference Protocol

**Standard inference (real-time):**
1. Single forward pass → `P_subsurface`, `u_epist`, `u_alea`, `conf`
2. Apply decision rule (§11.3)

**High-stakes inference (Cabeus-class targets):**
1. MC Dropout: T = 20 passes → `μ_P`, `σ_P`
2. Epistemic uncertainty = `σ_P + u_epist`
3. Require `σ_combined < 0.15` for positive ice declaration

### 11.3 Mission Decision Rule

```
DECLARE subsurface ice at pixel i IF:
    P_subsurface(i) > τ₁           (default τ₁ = 0.65)
    AND u_epist(i) < τ₂            (default τ₂ = 0.25)
    AND R_stability(i) < τ₃        (default τ₃ = 0.10)
    AND conf(i) > τ₄               (default τ₄ = 0.60)
```

Thresholds tuned on validation craters only; Cabeus never used for threshold selection.

### 11.4 Calibration

Post-training **temperature scaling** on validation Dirichlet evidence:

```
α_cal = α · exp(−T_cal),    T_cal learned to minimize ECE on val set
```

Target: ECE ≤ 0.05 on held-out craters.

---

## 12. Training Protocol (Specification Only)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimizer | AdamW | Standard for multimodal RS |
| Learning rate | 3 × 10⁻⁴ → cosine decay to 10⁻⁶ | |
| Weight decay | 10⁻⁴ | |
| Batch size | 8 | VRAM constraint |
| Epochs | 150 (early stop patience 20) | |
| Patch size | 128 × 128 | Per preprocessing spec |
| Augmentation | Random flip (H/V), 90° rotation, Gaussian noise σ=0.02 on radar/thermal only | Preserves physics symmetries at pole |
| No augmentation on | LOLA elevation (breaks slope), physics channels, labels | |
| Mixed precision | FP16 | |
| Gradient clipping | max norm 1.0 | Stabilizes physics loss |
| Pole strategy | Separate model weights per pole (north / south) | Different PSR statistics |

---

## 13. Scientific Validity Argument

### 13.1 Why This Architecture Is Physically Grounded

| Design choice | Scientific justification |
|---------------|-------------------------|
| Modality-grouped encoders | Each sensor measures a different physical observable (dielectric, thermal, nuclear, UV, spectral). Treating 59 channels as homogeneous RGB violates the measurement model. |
| Thermal-guided attention bias | Ice retention is thermodynamically gated. Neutron and radar evidence is only interpretable in cold-trap contexts (Paige et al. 2010; Feldman et al. 2010). |
| Physics Constraint Module | Prevents the network from learning spurious CPR–ice correlations in rocky terrain — the dominant failure mode in prior radar-only studies (Spudis et al. 2010 vs. Thompson et al. 2011). |
| Multi-scale FPN (P1–P5) | Subsurface ice is hypothesized at crater-scale (km) but must be resolved at orbital pixel scale (240 m). |
| Evidential uncertainty | Weak labels from proxy consensus carry inherent ambiguity; point estimates without uncertainty are scientifically misleading for mission planning. |
| Spatial crater holdout | The only honest evaluation protocol for spatially autocorrelated planetary data. |
| Separate pole models | North and south polar PSR geometries, illumination, and regolith properties are not exchangeable. |

### 13.2 Known Limitations (Documented)

1. **Subsurface ice labels are unverifiable at scale** — model learns proxy consensus, not ground truth.
2. **LEND resolution (~5 km)** — subsurface predictions below this scale are model extrapolation.
3. **M3 blindness in PSR** — spectral features cannot constrain ice in permanent shadow.
4. **CPR ambiguity** — even with roughness correction, blocky ejecta mimics ice radar signature.
5. **> 90% accuracy** — achievable on weak-label agreement under crater holdout, but does not constitute in-situ confirmation.

### 13.3 Validation Hierarchy

| Level | Method | Pass criterion |
|-------|--------|----------------|
| L1 — Proxy agreement | Spatial crater holdout test set | Accuracy ≥ 90%, macro-F1 ≥ 0.82 |
| L2 — Independent site | Cabeus LCROSS impact region | Subsurface recall ≥ 0.80 |
| L3 — Cross-mission | Agreement with Chandrayaan-2 DFSAR ice candidates (Peary, Shoemaker) | Spatial overlap ≥ 60% IoU |
| L4 — Physics consistency | Mean R_stability on predictions | < 0.05 |
| L5 — Calibration | ECE on validation | ≤ 0.05 |

---

## 14. Architecture Diagram (Layer Summary)

```
INPUT (59×128×128) + masks (6×128×128)
    │
    ├─ Enc_topo   (8→64)  ─┐
    ├─ Enc_radar  (10→64) ─┤
    ├─ Enc_thermal(15→64)─┤
    ├─ Enc_uv     (6→64)  ─┼─ FPN (P1–P5) ──┐
    ├─ Enc_neutron(6→64)  ─┤                 │
    ├─ Enc_spectral(7→64)─┤                 ├─ CMMA (P3–P5)
    └─ Enc_phys   (7→64)  ─┘                 │
                                              ▼
                                    Physics Constraint Module
                                    (Stefan · Stability · Radar · Neutron)
                                              │
                                              ▼
                              Attention-Gated U-Net Decoder (D5→D1)
                                              │
                    ┌─────────────┬───────────┼───────────┬──────────────┐
                    ▼             ▼           ▼           ▼              ▼
              Segmentation   P_ice      Dirichlet α   Confidence   Physics Violation
               (3-class)   (2-ch sig)   (uncertainty)  (1-ch sig)    (4-ch monitor)
```

---

## 15. Output Artifact Schema

| Output | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `logits_seg` | 3 × 128 × 128 | float32 | Raw class logits |
| `P_subsurface` | 1 × 128 × 128 | float32 | **Primary mission output** |
| `P_surface` | 1 × 128 × 128 | float32 | Surface ice probability |
| `u_epistemic` | 1 × 128 × 128 | float32 | Epistemic uncertainty |
| `u_aleatoric` | 1 × 128 × 128 | float32 | Aleatoric uncertainty |
| `conf` | 1 × 128 × 128 | float32 | Composite confidence |
| `v_phys` | 4 × 128 × 128 | float32 | Physics violation map |
| `latent_physics` | 5 × 128 × 128 | float32 | Interpretable DCM variables |

---

## 16. References

1. Ronneberger et al. (2015) — U-Net; MICCAI.
2. Lin et al. (2017) — Feature Pyramid Networks; CVPR.
3. Oktay et al. (2018) — Attention U-Net; MIDL.
4. Sensoy et al. (2018) — Evidential deep learning; NeurIPS.
5. Raissi et al. (2019) — Physics-informed neural networks; JCP.
6. Karniadakis et al. (2021) — Physics-informed machine learning; Nature Reviews Physics.
7. Paige et al. (2010) — Diviner; Science 330.
8. Feldman et al. (2010) — LEND; Science 330.
9. Colaprete et al. (2010) — LCROSS; Science 330.
10. Spudis et al. (2010) — Mini-SAR; GRL 37.
11. Li et al. (2023) — M3 water mosaics; PDS4/USGS.

---

## 17. Document Control

| Field | Value |
|-------|-------|
| Model name | `LUNAR-PIMAF-Net` |
| Architecture type | Hybrid Modality-Encoded Attention U-Net + Physics Constraint Module |
| Input tensor | `(59, 128, 128)` + `(6, 128, 128)` validity masks |
| Parameters | ~20.2 M |
| Primary metric | Subsurface ice probability under spatial crater holdout |
| Next revision | Integration of Chandrayaan-2 DFSAR as 8th modality encoder |

---

*Prepared by the LUNAR OS Autonomous Science Inference Working Group.*  
*Architecture specification only. Implementation deferred to `src/models/` development phase.*
