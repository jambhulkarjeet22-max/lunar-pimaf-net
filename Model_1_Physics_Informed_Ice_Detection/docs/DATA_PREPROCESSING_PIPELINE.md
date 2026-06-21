# LUNAR OS — Dataset Acquisition & Preprocessing Pipeline

**Physics-Informed Ice Detection Engine**  
**Document version:** 1.0  
**Scope:** Research-grade subsurface and surface ice detection using LRO (Mini-RF, Diviner, LOLA, LAMP, LEND) and Chandrayaan-1 M3  
**Status:** Architecture specification — no training code

---

## 1. Executive Summary

Lunar subsurface ice cannot be observed directly at orbital resolution. A research-grade model must therefore fuse **complementary geophysical proxies** — radar scattering, thermal stability, neutron hydrogen suppression, UV absorption, and near-infrared hydration — onto a **common topographic reference frame** derived from LOLA.

This pipeline defines:

1. Per-instrument feature extraction
2. A unified ML feature schema (47 base channels + 12 derived physics channels)
3. Spatial alignment and multi-sensor fusion
4. Preprocessing, missing-data policy, and normalization
5. Spatially honest train/validation/test partitioning
6. Multi-evidence weak-label construction for ice presence

**Primary study domain:** Lunar polar regions poleward of 80° latitude (north and south), where permanently shadowed regions (PSRs) and cold-trap thermodynamics make ice retention physically plausible.

**Secondary domain:** Equatorial and mid-latitude M3 hydration detections (surface-bound OH/H₂O), used for spectral calibration transfer only — not for subsurface ice supervision.

---

## 2. Data Inventory & Acquisition Order

| Priority | Dataset | PDS / Archive ID (representative) | Native Resolution | Primary Portal |
|----------|---------|-----------------------------------|-------------------|----------------|
| 1 | LOLA GDR / SLDEM | `LRO-L-LOLA-4-GDR-V1.0`, SLDEM2015 | 118 m (GDR); 60–240 m (polar) | [LOLA Data Node](https://imbrium.mit.edu/) / [Lunar ODE](https://ode.rsl.wustl.edu/moon/index.aspx) |
| 2 | LOLA GDRPSR | `LRO-L-LOLA-4-GDR-V1.0` (LPSR products) | 60–240 m | Lunar ODE |
| 3 | Diviner RDR + derived maps | `urn:nasa:pds:lro_diviner_derived` | Point-level; mosaics 2–240 ppd | [Diviner UCLA](https://www.diviner.ucla.edu/data) / Lunar ODE |
| 4 | Mini-RF Global Mosaic | `LRO-L-MRFLRO-5-GLOBAL-MOSAIC-V1.0`, PDS4 100 mpp bundle | 100 m (polar resample target) | [PDS Geosciences Mini-RF](https://pds-geosciences.wustl.edu/missions/lro/mrf.htm) |
| 5 | LAMP GDR | `LRO-L-LAMP-5-GDR-V1.0` | 240 m at pole | [PDS Imaging LRO](https://pds-imaging.jpl.nasa.gov/volumes/lro.html) |
| 6 | LEND averaged counts | `urn:nasa:pds:lro_lend` (averaged map products) | ~5–10 km (epithermal); collimated ~1–2 km | [PDS Geosciences LEND](https://pds-geosciences.wustl.edu/missions/lro/lend.htm) |
| 7 | M3 L2 reflectance + water mosaics | `CH1-ORB-L-M3-4-L2-REFLECTANCE-V3.0`, Li et al. 2023 PDS4 archive | 70 m (target) / 140 m (global) | [PDS Imaging M3](https://pds-imaging.jpl.nasa.gov/volumes/m3.html) / [USGS Astropedia](https://astrogeology.usgs.gov/search/map/lunar_m3_water_map_pds4_archive) |

**Acquisition sequence rationale:** LOLA topography and PSR geometry are acquired first because they define the reference grid, illumination masks, and spatial holdout units (crater polygons). All other sensors are warped to this frame.

---

## 3. Per-Instrument Feature Extraction

### 3.1 LRO Mini-RF

**Physical basis:** S-band (12.6 cm) synthetic aperture radar. Coherent backscatter from buried ice interfaces produces elevated Circular Polarization Ratio (CPR), but surface roughness and rocky slopes produce similar signatures. Roughness decoupling is mandatory before ice interpretation.

**Source products:**
- Global CPR mosaic (`GLOBAL_CPR_*_SIMP_0C.IMG`)
- Stokes parameters S0–S3 (100 mpp PDS4 bundle)
- Same-sense (SC) and opposite-sense (OC) power
- m-chi decomposition components (optional, if bundle available)

**Extracted features (per pixel):**

| Feature ID | Name | Derivation |
|------------|------|------------|
| `mrf_cpr` | Circular polarization ratio | SC / OC from mosaic |
| `mrf_sc_db` | Same-sense backscatter (dB) | 10·log₁₀(SC) |
| `mrf_oc_db` | Opposite-sense backscatter (dB) | 10·log₁₀(OC) |
| `mrf_s0` | Total power (Stokes S0) | Direct read |
| `mrf_s1_norm` | Normalized S1 | S1 / S0 |
| `mrf_s2_norm` | Normalized S2 | S2 / S0 |
| `mrf_s3_norm` | Normalized S3 | S3 / S0 |
| `mrf_mchi_odd` | Odd-bounce fraction | m-chi decomposition |
| `mrf_mchi_vol` | Volume scattering fraction | m-chi decomposition |
| `mrf_cpr_rough_corrected` | Roughness-decoupled CPR | Residual after SERD/ slope regression (see §5.4) |

**Pre-extraction steps:**
1. Mask fill values (`−3.4×10³⁸` in PDS4 mosaics).
2. Convert linear power to dB for SC/OC.
3. Clip CPR to physically plausible range [0, 3]; values > 2.5 flagged as `roughness_ambiguous`.

---

### 3.2 LRO Diviner

**Physical basis:** Multi-channel thermal radiometer (channels 3–8: 7.55–400 µm). Surface and subsurface temperature govern ice stability. Thermal inertia constrains regolith grain size and density, which modulates radar and neutron signatures.

**Source products:**
- Diviner RDR V3.1 (calibrated brightness temperature per channel)
- UCLA global/polar temperature mosaics (max, min, avg bolometric)
- Diviner Polar Resource Products (PRP): annual avg T, annual max T, permafrost depth
- Published cold-trap stability maps (Williams et al. 2019; Schorghofer & Williams 2020)

**Extracted features (per pixel):**

| Feature ID | Name | Derivation |
|------------|------|------------|
| `div_tbol_max` | Maximum bolometric temperature (K) | Seasonally aggregated RDR → mosaic |
| `div_tbol_min` | Minimum bolometric temperature (K) | Seasonally aggregated RDR → mosaic |
| `div_tbol_mean` | Mean bolometric temperature (K) | Diurnal mean |
| `div_tbol_std` | Temperature temporal std (K) | Captures diurnal/seasonal variability |
| `div_ch3_tb` … `div_ch8_tb` | Channel brightness temperatures | Per-channel mean over valid night passes |
| `div_emissivity_ch7` | Christiansen feature emissivity | Band-integrated emissivity |
| `div_thermal_inertia` | Thermal inertia (TIU) | Retrieved from diurnal amplitude (Hayne et al. 2017 method) |
| `div_cold_trap` | Binary cold-trap flag | T_max < volatile stability threshold for H₂O |
| `div_ice_stability` | Ice stability index [0,1] | Fraction of year below 110 K (configurable) |
| `div_permafrost_depth_m` | Depth to ice-stable permafrost (m) | From PRP V2.0 |
| `div_insolation` | Effective insolation (W/m²) | Ray-traced from LOLA DEM |

**Pre-extraction steps:**
1. Filter RDR by quality flags; exclude saturated detectors and daylight contamination in PSR-adjacent pixels.
2. Aggregate multiple orbital passes per grid cell: use **median** for temperature (robust to outliers).
3. Separate day-side and night-side observations before computing PSR-relevant temperatures.

---

### 3.3 LRO LOLA

**Physical basis:** 1064 nm laser altimetry. Provides the topographic foundation for shadow modeling, slope correction of radar, and terrain-aware spatial splits.

**Source products:**
- SLDEM2015 (LOLA + Kaguya SELENE fusion)
- GDR elevation maps (polar stereographic, 60–240 m)
- GDRPSR permanently shadowed maps (`LPSR_*`)
- RADR passive radiometry (optional reflectance at 1064 nm)
- Slope/azimuth derived from SLDEM2015

**Extracted features (per pixel):**

| Feature ID | Name | Derivation |
|------------|------|------------|
| `lola_elev_m` | Elevation (m, selenoid) | SLDEM2015 or GDR |
| `lola_slope_deg` | Surface slope (°) | Horn gradient on DEM |
| `lola_aspect_deg` | Aspect (°) | Horn gradient on DEM |
| `lola_roughness_m` | Surface roughness (m) | LOLA pulse-width-derived or DEM std in 3×3 window |
| `lola_psr_binary` | Permanently shadowed flag | GDRPSR ≥ 1 (any season) |
| `lola_psr_fraction` | PSR fraction | Fraction of year in shadow [0,1] |
| `lola_tpi` | Topographic position index | Elev − mean(elev in 500 m radius) |
| `lola_curvature` | Plan curvature | Second derivative of DEM |
| `lola_radiance_1064` | 1064 nm reflectance (optional) | RADR product |

**Pre-extraction steps:**
1. Fill small DEM voids (< 3 pixels) via bilinear interpolation; large voids remain masked.
2. Compute slope/aspect in polar stereographic projection (true slope, not lat/lon approximation).
3. Register GDRPSR to same DEM grid via nearest-neighbor (categorical).

---

### 3.4 LRO LAMP

**Physical basis:** Far-UV (57–196 nm) imaging spectroscopy. Detects water frost via on-band/off-band albedo contrast in PSRs where solar illumination is absent but Lyman-α sky glow provides excitation.

**Source products:**
- LAMP GDR brightness maps (on-band, off-band)
- LAMP GDR albedo maps
- H₂O absorption feature depth maps (`*_H2O_*`)
- Monthly time-series GDR (for temporal variability)

**Extracted features (per pixel):**

| Feature ID | Name | Derivation |
|------------|------|------------|
| `lamp_albedo_on` | On-band FUV albedo | GDR on-band albedo |
| `lamp_albedo_off` | Off-band FUV albedo | GDR off-band albedo |
| `lamp_albedo_ratio` | Off/on albedo ratio | Sensitive to H₂O frost |
| `lamp_h2o_depth` | H₂O absorption feature depth | Direct GDR product |
| `lamp_brightness` | Integrated brightness (photons/cm²/s) | Exposure-normalized |
| `lamp_exposure_time` | Effective exposure (s) | Quality weight for fusion |
| `lamp_temporal_std` | Albedo temporal std | From monthly GDR stack |

**Pre-extraction steps:**
1. Restrict to night-side and PSR observations only (per GDR processing metadata).
2. Normalize by Lyman-α illumination model (already applied in GDR albedo products).
3. Mask pixels with insufficient exposure (< configurable threshold, default 10 s equivalent).

---

### 3.5 LRO LEND

**Physical basis:** Neutron spectrometer measuring epithermal neutron flux suppression caused by hydrogen in the regolith. Provides the closest orbital proxy to bulk H content, sensitive to ~1 m depth (collimated channel shallower).

**Source products:**
- LEND averaged epithermal neutron counts map
- LEND collimated epithermal counts (higher resolution)
- Derived hydrogen-equivalent maps (wt% H) from science team processing

**Extracted features (per pixel):**

| Feature ID | Name | Derivation |
|------------|------|------------|
| `lend_epi_counts` | Epithermal neutron counts (c/s) | Averaged map product |
| `lend_epi_collimated` | Collimated epithermal counts | Higher-res channel |
| `lend_fast_counts` | Fast neutron counts | Averaged map |
| `lend_h_wt_pct` | Hydrogen equivalent (wt%) | Science team retrieval |
| `lend_neutron_suppression` | Suppression index | 1 − (counts / high-H reference) |
| `lend_h_gradient` | Spatial H gradient | Sobel on H map (edge detection at deposit boundaries) |

**Pre-extraction steps:**
1. Apply GCR and SPE corrections (use RDR-derived, not raw EDR).
2. Normalize counts to global reference highlands baseline per Feldman et al. methodology.
3. Acknowledge **inherent low resolution**: LEND features are upsampled to the fusion grid but retain a `lend_valid` confidence mask (see §7).

---

### 3.6 Chandrayaan-1 M3

**Physical basis:** Imaging spectrometer (0.43–3.0 µm). Detects OH and H₂O absorption features at 2.7–3.0 µm (surface hydration), with thermal emission contamination in polar regions requiring thermal correction.

**Source products:**
- M3 L2 reflectance (V3 calibrated)
- Li et al. (2023) thermal-corrected water mosaics (PDS4, 18 global tiles)
- Derived band depth at 2.8 µm and 1.25 µm

**Extracted features (per pixel):**

| Feature ID | Name | Derivation |
|------------|------|------------|
| `m3_r750` | Reflectance at 750 nm | Continuum reference |
| `m3_r1500` | Reflectance at 1500 nm | Mid-IR continuum |
| `m3_bd_1250` | Band depth at 1.25 µm | OH overtone |
| `m3_bd_2800` | Band depth at 2.8 µm | H₂O fundamental (surface) |
| `m3_slope_1um` | 1 µm absorption slope | Mafic mineralogy context |
| `m3_slope_2um` | 2 µm absorption slope | Pyroxene/plagioclase context |
| `m3_water_index` | Normalized water index | From Li et al. mosaic |
| `m3_sunlit` | Sunlit observation flag | M3 only valid on illuminated surface |

**Pre-extraction steps:**
1. Apply thermal correction model (Li & Milliken 2017 / Li et al. 2023) for polar tiles.
2. Photometric normalization to standard incidence/emergence geometry (i = 30°, e = 0°).
3. **Critical constraint:** M3 does not observe PSR interiors. Use M3 features only on sunlit pixels for mineralogy context and on PSR **rims** for boundary hydration. Do not treat M3 as ice label in shadow.

---

## 4. Final ML Feature Schema

### 4.1 Tensor Layout

Each training sample is a **spatial patch** (default 128 × 128 pixels at 240 m resolution ≈ 30.7 km footprint) represented as:

```
X ∈ ℝ^(C × H × W),  C = 59 channels,  H = W = 128
```

### 4.2 Channel Index (C = 59)

| Index | Feature ID | Group | Units / Range |
|-------|-----------|-------|---------------|
| 0 | `lola_elev_m` | Topography | m |
| 1 | `lola_slope_deg` | Topography | ° |
| 2 | `lola_aspect_deg_sin` | Topography | [−1, 1] |
| 3 | `lola_aspect_deg_cos` | Topography | [−1, 1] |
| 4 | `lola_roughness_m` | Topography | m |
| 5 | `lola_tpi` | Topography | m |
| 6 | `lola_curvature` | Topography | 1/m |
| 7 | `lola_psr_fraction` | Geometry | [0, 1] |
| 8 | `mrf_cpr` | Radar | [0, 3] |
| 9 | `mrf_cpr_rough_corrected` | Radar | [0, 3] |
| 10 | `mrf_sc_db` | Radar | dB |
| 11 | `mrf_oc_db` | Radar | dB |
| 12 | `mrf_s0` | Radar | linear |
| 13 | `mrf_s1_norm` | Radar | [−1, 1] |
| 14 | `mrf_s2_norm` | Radar | [−1, 1] |
| 15 | `mrf_s3_norm` | Radar | [−1, 1] |
| 16 | `mrf_mchi_odd` | Radar | [0, 1] |
| 17 | `mrf_mchi_vol` | Radar | [0, 1] |
| 18 | `div_tbol_max` | Thermal | K |
| 19 | `div_tbol_min` | Thermal | K |
| 20 | `div_tbol_mean` | Thermal | K |
| 21 | `div_tbol_std` | Thermal | K |
| 22–27 | `div_ch3_tb` … `div_ch8_tb` | Thermal | K |
| 28 | `div_emissivity_ch7` | Thermal | [0, 1] |
| 29 | `div_thermal_inertia` | Thermal | TIU |
| 30 | `div_ice_stability` | Thermal | [0, 1] |
| 31 | `div_permafrost_depth_m` | Thermal | m |
| 32 | `div_insolation` | Thermal | W/m² |
| 33 | `lamp_albedo_on` | UV | dimensionless |
| 34 | `lamp_albedo_off` | UV | dimensionless |
| 35 | `lamp_albedo_ratio` | UV | dimensionless |
| 36 | `lamp_h2o_depth` | UV | dimensionless |
| 37 | `lamp_brightness` | UV | photons/cm²/s |
| 38 | `lamp_temporal_std` | UV | dimensionless |
| 39 | `lend_epi_counts` | Neutron | c/s |
| 40 | `lend_epi_collimated` | Neutron | c/s |
| 41 | `lend_fast_counts` | Neutron | c/s |
| 42 | `lend_h_wt_pct` | Neutron | wt% |
| 43 | `lend_neutron_suppression` | Neutron | [0, 1] |
| 44 | `lend_h_gradient` | Neutron | wt%/km |
| 45 | `m3_r750` | Spectral | reflectance |
| 46 | `m3_r1500` | Spectral | reflectance |
| 47 | `m3_bd_1250` | Spectral | band depth |
| 48 | `m3_bd_2800` | Spectral | band depth |
| 49 | `m3_water_index` | Spectral | dimensionless |
| 50 | `m3_slope_1um` | Spectral | dimensionless |
| 51 | `m3_slope_2um` | Spectral | dimensionless |
| 52–58 | Physics-derived (see §4.3) | PINN | various |

### 4.3 Physics-Derived Channels (Indices 52–58)

These channels encode domain constraints for the physics-informed loss terms and serve as explicit ML inputs:

| Index | Feature ID | Formula / Source |
|-------|-----------|------------------|
| 52 | `phys_stefan_residual` | \|T_max⁴ − σ⁻¹·ε·(1−A)·I\| — thermodynamic consistency residual |
| 53 | `phys_ice_retention_prob` | Sigmoid(−(T_max − T_trap) / k) — trap temperature viability |
| 54 | `phys_radar_ice_likelihood` | P(ice \| CPR_corrected, TI, slope) — Bayesian lookup from literature priors |
| 55 | `phys_neutron_h_anomaly` | Z-score of `lend_h_wt_pct` within PSR mask |
| 56 | `phys_uv_frost_likelihood` | Normalized `lamp_h2o_depth` within PSR |
| 57 | `phys_multi_sensor_agreement` | Count of independent ice proxies above threshold / 5 |
| 58 | `phys_subsurface_accessibility` | PSR fraction × permafrost depth — favors buried ice hypothesis |

### 4.4 Label Schema

```
y ∈ {0, 1, 2}^H×W   (per-pixel classification)
```

| Value | Class | Meaning |
|-------|-------|---------|
| 0 | No ice | Insufficient evidence or thermally unstable |
| 1 | Surface/volatile ice | UV or spectral frost signature in cold trap |
| 2 | Subsurface ice | Neutron H anomaly + radar + thermal stability consensus |

**Soft labels (training):** `y_soft ∈ [0, 1]^(3×H×W)` — probability vector from multi-evidence label fusion (§9).

**Confidence mask:** `conf ∈ [0, 1]^(H×W)` — per-pixel label confidence for weighted loss.

---

## 5. Spatial Alignment & Multi-Sensor Fusion

### 5.1 Reference Coordinate System

| Parameter | Value |
|-----------|-------|
| Projection | Polar Stereographic (north: EPSG:104905 / IAU Moon north pole; south: EPSG:104906) |
| Base grid resolution | **240 m/pixel** (matches LAMP GDR and LOLA 240 m GDR) |
| Extent | 80°–90° latitude (full polar cap); optional 65°–90° for extended experiments |
| Vertical datum | LOLA selenoid (GLD100-compatible) |

**Rationale:** 240 m is the finest resolution achievable across all six sensors without excessive void interpolation. Mini-RF (100 m) is downsampled; LEND (~5 km) is upsampled with uncertainty propagation.

### 5.2 Alignment Pipeline

```
┌─────────────┐
│ LOLA SLDEM  │──► Reference grid (240 m, polar stereo)
│ + GDRPSR    │         │
└─────────────┘         ▼
                  ┌──────────────┐
                  │ DEM-derived  │
                  │ slope, PSR   │
                  └──────┬───────┘
                         │
    ┌────────────────────┼────────────────────┐
    ▼                    ▼                    ▼
┌────────┐        ┌──────────┐        ┌──────────┐
│Mini-RF │        │ Diviner  │        │  LAMP    │
│100→240m│        │ RDR→240m │        │ native   │
│bilinear│        │ median   │        │ 240m     │
└───┬────┘        └────┬─────┘        └────┬─────┘
    │                  │                    │
    └──────────────────┼────────────────────┘
                       ▼
              ┌─────────────────┐
              │  M3 (140m→240m) │  sunlit pixels only
              │  bilinear       │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ LEND (5km→240m) │
              │ area-weighted   │
              │ + conf mask     │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Fused tensor X  │
              │ 59 × H × W      │
              └─────────────────┘
```

### 5.3 Resampling Rules by Data Type

| Data Type | Interpolation | Examples |
|-----------|---------------|----------|
| Continuous scalar | Bilinear (downsample) / Bicubic (upsample ≤2×) | Temperature, elevation, CPR |
| Categorical / binary | Nearest neighbor (mode filter if downsampling) | PSR mask, cold-trap flag |
| Counting statistics | Area-weighted mean | LEND neutron counts |
| Directional | Vector-aware (aspect as sin/cos, then bilinear) | Aspect |
| Sparse point data (Diviner RDR) | Bin to grid → median per cell | RDR footprints |

### 5.4 Cross-Sensor Registration Validation

1. **LOLA-Mini-RF tie-point check:** Cross-correlate LOLA slope with Mini-RF SC backscatter at crater rims; acceptable offset < 1 pixel (240 m).
2. **Diviner-LAMP PSR boundary check:** PSR edges in LAMP albedo discontinuities must align with LOLA GDRPSR within 2 pixels.
3. **LEND smoothing acknowledgment:** Do not claim sub-km LEND spatial accuracy; store `lend_native_resolution_km` as metadata.

### 5.5 Temporal Harmonization

| Sensor | Temporal Strategy |
|--------|-------------------|
| Diviner | Aggregate full mission (~7+ years) → seasonal max/min/mean layers |
| Mini-RF | Use global mosaic (multi-orbit composite, 2009–2010 primary + extensions) |
| LAMP | Aggregate all night-side passes; optional monthly stack for temporal std |
| LOLA | Static (topography); PSR map version frozen to 2016 240 m product |
| LEND | Full mission average |
| M3 | Optical periods 1+2 only (2008–2009); no temporal update |

**Rule:** All products for a given patch must reference the same LOLA DEM version and the same Diviner aggregation window (document in patch metadata).

---

## 6. Preprocessing Pipeline (End-to-End)

### Stage 0 — Ingestion & Cataloging
- Download raw PDS products to `data/raw/{instrument}/`.
- Compute SHA-256 checksums; store PDS product IDs in a manifest (`data/processed/manifest.parquet`).
- Parse PDS labels (`.lbl` / PDS4 XML) for scaling factors, offsets, invalid values.

### Stage 1 — Instrument-Level Processing
- Apply PDS scaling equations per SIS documents.
- Apply instrument-specific quality masks.
- Export to Cloud-Optimized GeoTIFF (COG) in instrument-native projection.

### Stage 2 — Terrain Processing (LOLA)
- Mosaic SLDEM2015 polar tiles.
- Fill voids ≤ 3 pixels; mask larger voids.
- Compute slope, aspect (sin/cos), TPI, curvature, roughness.
- Rasterize GDRPSR to reference grid.

### Stage 3 — Derived Geophysical Products
- Diviner: aggregate RDR → bolometric temperature layers; compute thermal inertia.
- Mini-RF: compute CPR; apply roughness correction regression:
  `CPR_corrected = CPR − f(slope, roughness, SERD_proxy)` fitted on non-PSR highlands.
- LAMP: compute albedo ratio and temporal std from monthly GDR.
- LEND: compute suppression index and H gradient.
- M3: compute band depths; apply thermal correction; photometric normalization.

### Stage 4 — Reprojection & Fusion
- Warp all COGs to 240 m polar stereographic grid (§5.3 rules).
- Stack into multi-band GeoTIFF (59 bands) → `data/processed/fused/{pole}/tile_{row}_{col}.tif`.
- Generate per-band validity masks.

### Stage 5 — Patch Extraction
- Slide window: 128 × 128 pixels, stride 64 (50% overlap for training).
- Discard patches with > 30% invalid pixels (any band).
- Attach metadata: center lat/lon, crater ID (if overlapping), pole identifier.

### Stage 6 — Label Generation (§9)
- Run multi-evidence weak-label pipeline.
- Produce `y_hard`, `y_soft`, `conf` arrays per patch.

### Stage 7 — Normalization (§8)
- Fit scalers on **training craters only**.
- Apply to val/test.

### Stage 8 — Export
- Save as chunked Zarr or HDF5: `data/processed/patches/{split}/{patch_id}.zarr`.
- Schema version stamped in attributes.

---

## 7. Missing Value Handling

### 7.1 Missingness Categories

| Code | Cause | Affected Sensors |
|------|-------|------------------|
| `M0` | Instrument gap (no coverage) | Mini-RF (poles), M3 (PSR interior) |
| `M1` | Quality flag rejection | Diviner (saturated), LAMP (low exposure) |
| `M2` | DEM void | LOLA |
| `M3` | Resolution upsample uncertainty | LEND |
| `M4` | Thermal correction failure | M3 (polar) |

### 7.2 Handling Policy

| Category | Policy |
|----------|--------|
| `M0` Mini-RF polar gaps | Mark `mrf_valid = 0`; do **not** impute. Model receives validity channel. |
| `M0` M3 in PSR | Expected — set all `m3_*` to NaN; `m3_sunlit = 0`. Not missing at random. |
| `M1` Diviner | Interpolate from neighboring orbits within 30 days if ≥ 3 passes; else NaN. |
| `M2` DEM voids > 3 px | Patch rejected (Stage 5 threshold). |
| `M3` LEND | Area-weighted mean with `lend_conf = n_obs / n_expected`; conf < 0.3 → NaN. |
| `M4` M3 thermal | Exclude pixel from spectral features; retain thermal features from Diviner. |

### 7.3 Model-Facing Representation

For each band `b`, generate a companion validity mask:

```
valid_b = 1 if value is observed, 0 if missing
```

Missing continuous values are set to **0 after normalization** (per-group mean) with the validity mask signaling absence. The model must **not** receive raw NaN.

**Physics channels (52–58):** If any required input is missing, set physics channel to 0 and `phys_valid = 0`.

### 7.4 Imputation Prohibitions

- **Never** impute LEND from neighboring pixels for label generation (smearing H signal).
- **Never** impute M3 inside PSR (physically meaningless).
- **Never** impute CPR in rocky terrain without slope awareness.

---

## 8. Normalization Strategy

### 8.1 Group-Wise Feature Scaling

Features are normalized by **physically meaningful groups**, fit on training-set pixels only:

| Group | Features | Method | Notes |
|-------|----------|--------|-------|
| Topography | `lola_*` (0–7) | RobustScaler (median, IQR) | Per pole (N/S separate) |
| Radar | `mrf_*` (8–17) | StandardScaler (μ, σ) | Fit on non-PSR highlands + training PSR |
| Thermal | `div_*` (18–32) | MinMaxScaler to [0,1] | Per channel; T in [20, 400] K clip before |
| UV | `lamp_*` (33–38) | Log1p → StandardScaler | Albedo strictly positive |
| Neutron | `lend_*` (39–44) | StandardScaler | Per pole; collimated separate from uncollimated |
| Spectral | `m3_*` (45–51) | StandardScaler | Sunlit pixels only for fit |
| Physics | `phys_*` (52–58) | Already [0,1] or standardized | Fixed scaling from physical constants |

### 8.2 Slope & Aspect
- `lola_slope_deg` → clip to [0, 45°] then divide by 45.
- `lola_aspect_deg` → convert to `(sin(θ), cos(θ))` before any scaling (already in [−1, 1]).

### 8.3 Distribution Monitoring
- Store training-set histograms per channel in `data/processed/stats/normalization_v1.json`.
- Flag covariate shift if val/test KS-statistic > 0.1 for any channel.

### 8.4 No Leakage Rule
Scaler parameters (μ, σ, min, max, median, IQR) are computed **exclusively** from training crater polygons (§10). Validation and test craters are never seen during fitting.

---

## 9. Ground Truth Label Construction

### 9.1 Fundamental Constraint

**No global in-situ subsurface ice map exists.** All labels are **weak / proxy labels** derived from multi-sensor consensus and published geophysical thresholds. Reported model accuracy must be framed as agreement with geophysical proxy consensus, validated against held-out craters and LCROSS impact site chemistry.

### 9.2 Evidence Layers (Per Pixel)

Each layer produces a score ∈ [0, 1]:

| Layer | Source | Ice-Positive Criterion | Subsurface? |
|-------|--------|------------------------|-------------|
| E1 — Thermal viability | Diviner `div_ice_stability` | ≥ 0.8 | Prerequisite |
| E2 — Neutron hydrogen | LEND `lend_neutron_suppression` | ≥ 0.7 (top 10% within PSR) | Yes |
| E3 — Radar anomaly | Mini-RF `mrf_cpr_rough_corrected` | ≥ 1.2 AND slope < 10° | Yes (ambiguous alone) |
| E4 — UV frost | LAMP `lamp_h2o_depth` | ≥ 2σ above PSR mean | Surface |
| E5 — Spectral hydration | M3 `m3_bd_2800` | ≥ 2σ (sunlit pixels only) | Surface |
| E6 — PSR membership | LOLA `lola_psr_fraction` | ≥ 0.5 | Context |

### 9.3 Label Decision Rules

**Class 2 — Subsurface ice (y = 2):**
```
E1 ≥ 0.8  AND  E2 ≥ 0.7  AND  E3 ≥ 0.5  AND  E6 ≥ 0.5
```

**Class 1 — Surface/volatile ice (y = 1):**
```
E1 ≥ 0.8  AND  (E4 ≥ 0.6  OR  E5 ≥ 0.6)  AND  E6 ≥ 0.3
```

**Class 0 — No ice (y = 0):**
```
E1 < 0.5  OR  (E2 < 0.3 AND E3 < 0.3 AND E4 < 0.3)
```

**Unlabeled (excluded from loss):** All other pixels — ambiguous consensus.

### 9.4 Soft Label Vector

```
y_soft[0] = 1 − max(E2, E3, E4, E5)   # no ice
y_soft[1] = mean(E4, E5) × E1 × E6    # surface
y_soft[2] = min(E2, E3) × E1 × E6     # subsurface
# Renormalize to sum = 1
```

### 9.5 Confidence Score

```
conf = phys_multi_sensor_agreement / 5 × (1 − mrf_roughness_ambiguity)
```

Patches with mean `conf < 0.4` are excluded from training.

### 9.6 Anchor Validation Sites (Not Used for Training Labels)

| Site | Role | Reference |
|------|------|-----------|
| Cabeus crater (LCROSS impact) | Independent validation — confirmed H₂O ice | Colaprete et al. 2010 |
| Shackleton rim | High-interest ambiguity (roughness vs ice) | Thomson et al. 2012 |
| Faustini, Haworth, Shoemaker | CH-2 DFSAR ice candidates (external validation) | ISRO SAC 2024 |

### 9.7 Label Versioning

- `label_v1`: Threshold rules above.
- Future `label_v2`: Incorporate Chandrayaan-2 DFSAR SERD-decoupled CPR (external to this 6-dataset spec).
- All experiments must log `label_version` in metadata.

---

## 10. Train / Validation / Test Split Strategy

### 10.1 Spatial Holdout by Crater (Mandatory)

**Never split at random pixels.** Spatial autocorrelation in lunar terrain and ice proxies would inflate accuracy by 15–30 percentage points.

**Unit of split:** Named crater polygons (or PSR contiguous regions > 5 km²) from LOLA GDRPSR connected-component analysis.

### 10.2 Partition Allocation

| Split | Fraction | Selection Criteria |
|-------|----------|-------------------|
| Train | 60% | Random crater assignment within pole |
| Validation | 20% | Stratified by crater area and PSR fraction |
| Test | 20% | Includes at least 2 "anchor" craters held out entirely |

### 10.3 Held-Out Test Craters (Fixed, Never in Train/Val)

**South Pole:**
- Cabeus (LCROSS validation)
- Shackleton
- Haworth

**North Pole:**
- Peary
- Rozhdestvenskiy W

### 10.4 Cross-Validation

- **5-fold spatial block CV** on training craters for hyperparameter tuning.
- Blocks defined by 5° longitude sectors within polar cap.
- Report mean ± std across folds.

### 10.5 North / South Pole Strategy

**Option A (recommended for v1):** Train separate models per pole (different illumination, different PSR statistics).

**Option B:** Pooled model with `pole_id` as categorical metadata channel (not in the 59 features — provided as patch attribute).

### 10.6 Leakage Checks

- [ ] No overlapping 128 px patches between splits (use crater polygon buffer of 2 km).
- [ ] Normalization scalers fit on train only.
- [ ] CPR roughness regression fit on non-PSR highlands excluding test craters.
- [ ] Label thresholds frozen before any model training.

### 10.7 Expected Performance Interpretation

| Metric | Target | Caveat |
|--------|--------|--------|
| Pixel accuracy vs weak labels | > 90% on test craters | Only meaningful with spatial holdout |
| F1 (subsurface ice, class 2) | > 0.75 | Rare class — use macro-F1 |
| LCROSS Cabeus hit rate | Recall > 0.8 in impact region | Independent check |
| Calibration (ECE) | < 0.05 | Required for mission decisions |

> **Critical:** > 90% accuracy is achievable only when (a) spatial holdout is enforced, (b) class 0 dominates and the model is not evaluated on random pixels, and (c) weak labels are internally consistent. Report per-class metrics and Cabeus validation separately.

---

## 11. Quality Assurance Checklist

- [ ] All PDS product IDs logged in manifest
- [ ] Residual misregistration < 240 m at 10 random tie-points per pole
- [ ] No NaN in exported model tensors (validity masks used)
- [ ] Training patch count ≥ 5,000 per pole
- [ ] Class 2 (subsurface) ≥ 5% of labeled pixels (adjust thresholds if needed)
- [ ] Normalization stats saved and versioned
- [ ] Label version and threshold config committed to `configs/labeling_v1.yaml`
- [ ] Reproducible: fixed random seed for crater split (seed = 42)

---

## 12. Output Artifacts

| Artifact | Path | Format |
|----------|------|--------|
| Raw ingest manifest | `data/processed/manifest.parquet` | Parquet |
| Per-instrument COGs | `data/processed/{instrument}/` | GeoTIFF (COG) |
| Fused stack | `data/processed/fused/{pole}/` | GeoTIFF (59 bands) |
| Training patches | `data/processed/patches/{split}/` | Zarr |
| Normalization params | `data/processed/stats/normalization_v1.json` | JSON |
| Crater split map | `data/processed/splits/crater_split_v1.geojson` | GeoJSON |
| Label rasters | `data/processed/labels/{pole}/` | GeoTIFF |
| QA report | `logs/preprocessing_qa_v1.md` | Markdown |

---

## 13. References

1. Paige et al. (2010) — Diviner lunar radiometer; Science 330, 479.
2. Spudis et al. (2010) — Mini-SAR ice detection; GRL 37.
3. Feldman et al. (2010) — LEND hydrogen mapping; Science 330.
4. Gladstone et al. (2012) — LAMP far-UV albedo; JGR 117.
5. Li & Milliken (2017) — M3 thermal correction; JGR Planets 122.
6. Li et al. (2023) — M3 water mosaics; PDS4 archive, USGS Astropedia.
7. Colaprete et al. (2010) — LCROSS water detection; Science 330.
8. Williams et al. (2019) — Diviner seasonal polar temperatures; JGR 124.
9. Schorghofer & Williams (2020) — Cold trap mapping; PSJ 1, 238.
10. Mazarico et al. (2011) — LOLA PSR maps; Icarus 213.

---

*Document maintained by the LUNAR OS data engineering team. Next revision: integrate Chandrayaan-2 DFSAR as optional seventh sensor stream.*
