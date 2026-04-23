# Multi-Modal Physiological Analysis — Code Guide
https://claude.ai/share/b0e81db8-145f-4838-8b1d-60dc539a6758
## Files Required (place all in same folder as .py scripts)
| File | Type | Description |
|------|------|-------------|
| `check_thermal.mp4` | Video | 640×480 thermal camera, HOT colormap, 25fps, ~23 min |
| `Radar_Heart_Output_1.csv` | CSV | Radar-based heart rate + HRV (87 rows, every 15s) |
| `Radar_Breath_Output_1.csv` | CSV | Radar-based breathing rate + HRV (87 rows) |
| `posture_analysis_full.csv` | CSV | 2D posture scores per minute (24 rows) |
| `3D_posture_analysis_full.csv` | CSV | 3D posture scores per minute (24 rows) |
| `cd6_write.csv` | CSV | CD6 continuous signal (28k rows, ~20 Hz) |
| `IR_01678.IS3 / IR_01679.IS3` | Binary | FLIR raw IR frames (Matroska container) |

---

## Install Dependencies
```bash
pip install opencv-python-headless numpy pandas matplotlib scipy
```

---

## Run Order

### Step 1 — Extract thermal nose tracking
```bash
python 1_thermal_nose_tracker.py
```
**Output:** `nose_tracking.csv`  
Detects face every second, extracts 9 nose-region landmarks, records R/G/B + thermal intensity.

**Key parameter to tune:**
```python
SAMPLE_EVERY_N_FRAMES = 25   # 25 = 1 sample/sec. Reduce to 5 for finer resolution
PATCH_RADIUS = 2             # pixel neighbourhood around each landmark
```

---

### Step 2 — Plot thermal time-series
```bash
python 2_thermal_timeseries_plots.py
```
**Outputs:** `nose_thermal_timeseries.png`, `nose_thermal_heatmap.png`, `nose_thermal_correlation.png`, `nose_thermal_delta.png`

---

### Step 3 — Radar vitals analysis
```bash
python 3_radar_vitals_analysis.py
```
**Outputs:** `radar_vitals_overview.png`, `radar_hrv_metrics.png`, `radar_breathing_metrics.png`, `radar_freq_bands.png`

---

### Step 4 — Posture analysis
```bash
python 4_posture_analysis.py
```
**Outputs:** `posture_score_timeline.png`, `posture_angle_vs_kpdist.png`, `posture_2d_vs_3d.png`

---

### Step 5 — CD6 signal analysis
```bash
python 5_cd6_signal_analysis.py
```
**Outputs:** `cd6_raw_signal.png`, `cd6_spectrogram.png`, `cd6_fft.png`, `cd6_rolling_stats.png`

---

### Step 6 — Full multi-modal fusion dashboard
```bash
python 6_multimodal_fusion.py
```
**Outputs:** `multimodal_dashboard.png`, `multimodal_correlation.png`, `fused_multimodal.csv`  
Aligns all streams to a common 15-second grid and plots them together.

---

## Thermal Intensity Interpretation
The video uses the **HOT colormap** (black → red → orange → yellow → white).  
Thermal Intensity (TI) is computed as:

```
TI = R/2 + G/4 + B/8
```

This is monotonically proportional to temperature. **Higher TI = warmer region.**

### 9 Nose Landmark Positions
| Point | Face-bbox ratio (x, y) | Notes |
|-------|------------------------|-------|
| nose_tip | (0.50, 0.65) | Coldest nasal point (exhaled air) |
| nose_bridge | (0.50, 0.47) | Bone-backed, relatively stable |
| left_nostril | (0.35, 0.68) | Airflow-sensitive (breath detection) |
| right_nostril | (0.65, 0.68) | Airflow-sensitive |
| nose_left_ala | (0.30, 0.60) | Cartilage + skin |
| nose_right_ala | (0.70, 0.60) | Cartilage + skin |
| philtrum | (0.50, 0.74) | Upper lip area |
| nose_base_left | (0.38, 0.72) | Near nostril floor |
| nose_base_right | (0.62, 0.72) | Near nostril floor |

### Thermal–Physiology Connections
- **Nostril TI oscillation** → respiratory rate (cooler on exhale, warmer on inhale)
- **Nose tip TI** → skin blood flow / vasodilation proxy
- **Nose bridge TI** → stable reference (minimal airflow effect)
- **Philtrum TI** → correlates with facial blood perfusion

---

## IS3 Files (IR_01678.IS3 / IR_01679.IS3)
These are **FLIR raw thermal frames** stored in Matroska container format.  
To extract raw temperature data (in Kelvin/Celsius), use:
```bash
# Install FLIR tools or use ffmpeg to extract frames:
ffmpeg -i IR_01678.IS3 -vf "fps=1" frame_%04d.png
```
Or use the **FLIR Science File SDK** / `irbis3` Python bindings for true temperature values.

---

## Possible Analyses
| Analysis | Files Needed | Code |
|----------|-------------|------|
| Nose thermal tracking | check_thermal.mp4 | 1, 2 |
| Respiratory rate from thermal | check_thermal.mp4 | 1, 2 |
| Heart rate variability | Radar_Heart | 3 |
| Breathing HRV | Radar_Breath | 3 |
| Posture scoring | posture CSVs | 4 |
| CD6 signal frequency | cd6_write.csv | 5 |
| Cross-modal correlations | All | 6 |
| Thermal ↔ HR correlation | All | 6 |
| Stress index (LF/HF) | Radar_Heart | 3 |

---

## Nostril Keypoint Detection (Scripts 7–12)

A deep learning sub-pipeline to detect nostril positions in thermal frames:

```bash
python 7_check_nose_points.py        # Debug nose landmark overlays on frames
python 8_semi_unsupervised_analysis.py  # Cluster thermal features (unsupervised)
python 9_thermal_blob_detector.py    # Detect nostril blobs via thresholding
python 10_validate_nostril_labels.py # Validate CSV label file for errors
python 11_label_nostrils.py          # Interactively annotate nostril positions
python 12_train_nostril_model.py     # Train CNN keypoint model (run on Colab)
```

**Training on Google Colab:**
1. Run `prepare_colab_data.py` locally to zip frames + labels
2. Upload `nostril_colab_data.zip` to Google Drive
3. Open Colab, mount Drive, unzip, then run `12_train_nostril_model.py`
4. Outputs: `nostril_model.pt`, `nostril_predictions.csv`, `training_curves.png`

**Inference:**
```bash
# On Colab, after training:
python nostril_inference_colab.py
```

---

## Utility Scripts

| Script | Description |
|--------|-------------|
| `extract_face_frames.py` | Crops face regions from video frames |
| `extract_thermal_frames.py` | Saves individual frames from thermal video |
| `extract_nostril.py` | Crops nostril patches for labeling/training |
| `prepare_colab_data.py` | Zips frames + labels for Colab upload |
| `scan_frontal.py` | Filters for forward-facing frames |
| `plot_nostril_labels.py` | Visualises ground-truth label annotations |
| `plot_inference_predictions.py` | Overlays model predictions on frames |
| `emognition_hf.py` | HF band emotion/arousal signal processing |
