# Architecture Specification — Lunar Radiation Risk Prediction AI System

This document outlines the software and deep learning architecture of the Model 4 Lunar Radiation Risk Prediction AI system.

## System Block Diagram

```mermaid
graph TD
    %% Modalities
    subgraph Inputs ["Input Modalities (Each [B, 1, H, W])"]
        I1["LOLA DEM (lola)"]
        I2["Elevation Map (elevation)"]
        I3["Solar Illumination (illumination)"]
        I4["Permanent Shadow Region (psr)"]
        I5["Surface Temp (diviner)"]
        I6["Cosmic Ray Flux (flux)"]
        I7["Regolith Thickness (regolith)"]
    end

    %% Encoders
    subgraph Encoders ["MultiModalRadiationEncoder"]
        E1["RadiationEncoder (lola)"]
        E2["RadiationEncoder (elevation)"]
        E3["RadiationEncoder (illumination)"]
        E4["RadiationEncoder (psr)"]
        E5["RadiationEncoder (diviner)"]
        E6["RadiationEncoder (flux)"]
        E7["RadiationEncoder (regolith)"]
    end

    I1 --> E1
    I2 --> E2
    I3 --> E3
    I4 --> E4
    I5 --> E5
    I6 --> E6
    I7 --> E7

    %% Fusion
    subgraph Fusion ["AttentionFusion Layer"]
        F_Cat["Concat Features [B, 7 * F, H/2, W/2]"]
        F_Att["Softmax Attention Weights [B, 7, H/2, W/2]"]
        F_Proj["Fused Conv Projection [B, 2*F, H/2, W/2]"]
    end

    E1 & E2 & E3 & E4 & E5 & E6 & E7 --> F_Cat
    F_Cat --> F_Att
    F_Cat & F_Att --> F_Proj

    %% Heads
    subgraph Heads ["RadiationHeads (Multi-Task Output)"]
        H1["Dose Rate Head [B, 1, H/2, W/2] (Softplus)"]
        H2["Risk Score Head [B, 1, H/2, W/2] (Sigmoid)"]
        H3["Shielding Head [B, 1, H/2, W/2] (Sigmoid)"]
        H4["Safety Head [B, 1, H/2, W/2] (Sigmoid)"]
        H5["Hazard Map Head [B, 1, H/2, W/2] (Sigmoid)"]
    end

    F_Proj --> H1 & H2 & H3 & H4 & H5

    %% Upsampling
    subgraph Outputs ["Geospatial Outputs (Interpolated [B, 1, H, W])"]
        O1["Radiation Dose Rate (mSv/day)"]
        O2["Radiation Risk Score (0-1)"]
        O3["Shielding Effectiveness Score (0-1)"]
        O4["Habitat Safety Score (0-1)"]
        O5["Final Radiation Hazard Map (0-1)"]
    end

    H1 -->|Bilinear Interpolate| O1
    H2 -->|Bilinear Interpolate| O2
    H3 -->|Bilinear Interpolate| O3
    H4 -->|Bilinear Interpolate| O4
    H5 -->|Bilinear Interpolate| O5

    %% Physics Losses
    subgraph Regularization ["Physics-Aware Loss Coupling"]
        P1["Regolith & PSR Shielding Constraint"]
        P2["Elevation Exposure Constraint"]
        P3["Shielding-Regolith Direct Coupling"]
        P4["Habitat Safety Algebraic Coupling"]
    end

    O2 & I7 & I4 --> P1
    O2 & I2 --> P2
    O3 & I7 --> P3
    O4 & O3 & O2 --> P4
```

---

## Modality Specifications

The model processes 7 distinct geospatial layers corresponding to orbital instrument readings or simulated environments:

1. **lola**: Lunar Orbiter Laser Altimeter DEM mapping terrain profile.
2. **elevation**: Calibrated elevation measurements.
3. **illumination**: Cumulative solar illumination factor mapping exposure to solar radiation.
4. **psr**: Permanent Shadow Region mask designating crater interiors shielded from direct solar illumination.
5. **diviner**: Diviner Lunar Radiometer thermal maps correlating with heat and shadow.
6. **flux**: Solar particle events (SPE) and Galactic Cosmic Rays (GCR) exposure models.
7. **regolith**: Estimated depth/thickness profile of the lunar regolith.

Each modality is individually processed by an spatial `RadiationEncoder` consisting of 2D convolutions, batch normalization, and ReLU activations, reducing resolution by a factor of 2 (via stride=2) to increase receptive field.

---

## Attention Fusion Mechanism

The features are combined using a spatial/modality softmax attention mechanism:
1. **Concatenation**: Extracted features of dimension $F$ for the 7 modalities are stacked along the channel axis.
2. **Modality Weighting**: A $1\times 1$ convolution calculates spatial/modality attention coefficients.
3. **Softmax**: Attention logits are normalized using a softmax activation across modalities.
4. **Channel Projection**: Modalities are scaled by their spatial weights, concatenated, and projected through a $3\times 3$ convolutional block to a combined dimension of $2 \times F$.

---

## Physics-Aware Loss Formulation

To ensure the model learns physically consistent mappings even with noisy data, the loss objective incorporates regularizations:

$$L_{\text{total}} = L_{\text{supervised}} + \lambda_{\text{physics}} L_{\text{physics}}$$

Where $L_{\text{supervised}}$ is the multi-task MSE and BCE loss, and $L_{\text{physics}}$ comprises:

1. **Regolith & PSR Shielding constraint**:
   High regolith thickness and shadow zone interiors block radiation, defining a physical upper limit on radiation risk:
   $$L_{\text{shielding}} = \text{ReLU}\left(S_{\text{risk}} - (1.0 - 0.5 \cdot \text{regolith} - 0.3 \cdot \text{psr})\right)$$

2. **Elevation Exposure constraint**:
   Exposed elevated terrains lack local topography shielding, imposing a lower limit on radiation risk:
   $$L_{\text{exposure}} = \text{ReLU}\left(0.6 \cdot \text{elevation} - S_{\text{risk}}\right)$$

3. **Shielding Effectiveness coupling**:
   Alignspredicted shielding effectiveness directly with regolith thickness:
   $$L_{\text{shield\_eff}} = \text{MSE}\left(S_{\text{shielding}}, \text{regolith}\right)$$

4. **Habitat Safety coupling**:
   Habitat safety represents the interaction of shielding protection and radiation hazard:
   $$L_{\text{safety}} = \text{MSE}\left(S_{\text{safety}}, S_{\text{shielding}} \times (1.0 - S_{\text{risk}})\right)$$
