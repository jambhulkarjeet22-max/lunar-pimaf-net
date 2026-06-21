# Architecture Specification — Rover Hazard & Navigation Intelligence (Model 5)

This document describes the software and machine learning architecture of the Model 5 Rover Hazard & Navigation Intelligence system.

## System Block Diagram

```mermaid
graph TD
    %% Modalities
    subgraph Inputs ["Input Modalities"]
        I1["LOLA DEM (lola) [B, 1, H, W]"]
        I2["Mini-RF radar (mini_rf) [B, 3, H, W]"]
        I3["Hi-res DEM (dem) [B, 1, H, W]"]
        I4["Slope maps (slope) [B, 1, H, W]"]
        I5["Crater maps (crater) [B, 1, H, W]"]
        I6["Boulder maps (boulder) [B, 1, H, W]"]
        I7["Illumination maps (illumination) [B, 1, H, W]"]
    end

    %% Encoders
    subgraph Encoders ["MultiModalTerrainEncoder"]
        E1["TerrainEncoder (lola)"]
        E2["TerrainEncoder (mini_rf)"]
        E3["TerrainEncoder (dem)"]
        E4["TerrainEncoder (slope)"]
        E5["TerrainEncoder (crater)"]
        E6["TerrainEncoder (boulder)"]
        E7["TerrainEncoder (illumination)"]
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
    subgraph Heads ["NavigationHeads (Multi-Task Output)"]
        H1["Traversability Head [B, 1, H/2, W/2]"]
        H2["Crater Hazard Head [B, 1, H/2, W/2]"]
        H3["Boulder Hazard Head [B, 1, H/2, W/2]"]
        H4["Slope Hazard Head [B, 1, H/2, W/2]"]
        H5["Navigation Cost Head [B, 1, H/2, W/2]"]
        H6["Safety Head [B, 1, H/2, W/2]"]
    end

    F_Proj --> H1 & H2 & H3 & H4 & H5 & H6

    %% Upsampling
    subgraph Outputs ["Geospatial Outputs (Interpolated [B, 1, H, W])"]
        O1["Traversability Score"]
        O2["Crater Hazard Probability"]
        O3["Boulder Hazard Probability"]
        O4["Slope Hazard Score"]
        O5["Safe Navigation Cost Map"]
        O6["Final Rover Safety Score"]
    end

    H1 -->|Bilinear Interpolate| O1
    H2 -->|Bilinear Interpolate| O2
    H3 -->|Bilinear Interpolate| O3
    H4 -->|Bilinear Interpolate| O4
    H5 -->|Bilinear Interpolate| O5
    H6 -->|Bilinear Interpolate| O6
```

---

## Modality Specifications

The model integrates 7 orbital and terrain-based modalities to synthesize hazard and traversability cost maps:
1. **lola**: Low-resolution lunar orbiter laser altimeter topography data.
2. **mini_rf**: 3-channel Mini-RF polarimetric radar mapping surface roughness.
3. **dem**: High-resolution digital elevation model mapping local slopes and micro-topography.
4. **slope**: Pre-calculated surface slope maps.
5. **crater**: Crater probability or rim location density maps.
6. **boulder**: Boulder location/frequency maps.
7. **illumination**: Solar illumination mapping exposure.

---

## Physics-Aware Loss Formulation

To enforce consistent navigation safety and path cost metrics, the multi-task loss is augmented with several physics constraints:

1. **Slope & Boulder Traversability Limit**: Enforces that safe traversability cannot exceed the bounds determined by slopes and boulders:
   $$S_{\text{trav}} \leq 1.0 - \text{slope} - \text{boulder}$$
2. **Slope Hazard direct coupling**: Forces slope hazard predictions to reflect the raw physical input slopes.
3. **Multiplicative safety consistency**: Rover safety score must reflect the joint safety across all individual hazard predictions:
   $$\text{Safety} \approx (1.0 - \text{crater\_hazard}) \times (1.0 - \text{boulder\_hazard}) \times (1.0 - \text{slope\_hazard})$$
4. **Navigation cost map coupling**: Navigation cost is the inverse of predicted safety:
   $$\text{Cost} \approx 1.0 - \text{Safety}$$
5. **Illumination traversability lower limit**: Ensures that good solar illumination combined with low hazards guarantees a baseline traversability level.
