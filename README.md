# BEV 2D Occupancy — Uncertainty-Aware Prototype
### Hackathon Submission · Problem Statement 3

---

## 🔗 Live Demo  
👉 https://meddoesdev.github.io/uncertainty-aware-bev/

Note: Make sure to wait for 40-60s(sometimes visualisations may take upto 2 mins to load) after clicking on the link.


## What This Does

Transforms surround-camera images from the nuScenes dataset into a **Bird's-Eye-View (BEV) occupancy grid** with **epistemic uncertainty quantification** via Monte Carlo Dropout.

### Novel Contribution
Most BEV systems output a single occupancy map. This prototype outputs **three co-registered maps simultaneously**:

1. **Mean Occupancy** — where obstacles likely are, derived from vertical column scan features
2. **Epistemic Uncertainty** — *how confident* the model is (via MC Dropout variance across 10 passes)
3. **Fused Uncertainty-Overlay** — occupancy coloured by uncertainty (red bleed = uncertain region)

This lets a planner know not just *where* obstacles are, but *how trustworthy* each grid cell's prediction is — critical for safe L4 autonomy. The **higher uncertainty at camera seam boundaries** is a physically valid signal: those are exactly the regions where multi-camera agreement is lowest.

---

## Architecture

```
6x Camera Images (nuScenes surround rig)
            |
            v
Calibrated IPM Projection
(Ego pose quaternion + camera intrinsics K + extrinsics R,t -> world grid coords)
            |
            v
Vertical Column Scan  <- KEY IMPROVEMENT
(12 samples, 80px above each ground-plane projection point)
            |
            v
6-Channel Feature Extraction per BEV cell
[ground_R, ground_G, ground_B, col_var, col_dev, col_sat]
            |
            v
Multi-Camera Feature Fusion
(Weighted average across all cameras that see each cell -- 99%+ coverage)
            |
            v
MC Dropout x 10 passes
(Stochastic dropout p=0.3 -> distribution of occupancy predictions)
            |
            v
Mean + Variance  ->  Occupancy Map + Epistemic Uncertainty Map
            |
            v
Spatial Post-Processing
(7x7 Median filter -> Dual Gaussian blur -> Unsharp mask edge restore)
            |
            v
Self-contained HTML Dashboard (no server needed)
```

---

## How It Works

### Vertical Column Obstacle Detection

The core accuracy improvement over standard IPM: instead of sampling only the single ground-plane pixel for each BEV cell, the pipeline **scans a vertical column of 12 pixels above that point** in the camera image.

**Why this works:** A standing obstacle (car, truck, construction barrier) occupies vertical image space *above* its ground footprint. The camera projection of a BEV cell lands on the road surface — but any object sitting on that surface will appear directly above it in the image. A bare road surface produces a smooth, uniform column (sky blending into road). An obstacle produces a sharp luminance and colour discontinuity in that column.

Three column-derived features are extracted:

| Feature | What it detects |
|---|---|
| `col_var` — column luminance variance | Vertical discontinuities from obstacle edges |
| `col_dev` — max colour deviation from ground | Objects that differ in colour from the road |
| `col_sat` — upper-column saturation | Coloured objects vs grey sky in the upper portion |

These three signals contribute **77% of the final occupancy score**. A **road suppressor** multiplier `(1 - ground_sat) x (1 - col_var)` zeroes out cells that score flat on both ground appearance and column variation — eliminating false positives on plain tarmac.

### Monte Carlo Dropout Uncertainty

10 stochastic forward passes with feature dropout (p=0.3) approximate a Bayesian posterior over occupancy. The variance across passes is the **epistemic uncertainty** per BEV cell. Camera boundary seams show highest uncertainty because adjacent cameras see slightly different perspectives of the same ground cell, causing genuine prediction disagreement.

### Spatial Post-Processing

Three-stage smoothing pipeline:
1. **7x7 Median filter** — kills isolated salt-and-pepper outliers
2. **Dual Gaussian blur** (sigma=3 then sigma=1.5) — merges nearby signals into coherent regions
3. **Unsharp mask** — restores structural detail, preventing over-smoothing

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download nuScenes mini
- Go to https://www.nuscenes.org/nuscenes#download
- Download the **"Mini"** split (~4 GB)
- Extract so your folder looks like:
```
nuscenes/
  v1.0-mini/
    sample.json
    sample_data.json
    calibrated_sensor.json
    ego_pose.json
    ...
  samples/
    CAM_FRONT/
    CAM_FRONT_LEFT/
    CAM_FRONT_RIGHT/
    CAM_BACK/
    CAM_BACK_LEFT/
    CAM_BACK_RIGHT/
  sweeps/
  maps/
```

> **Note:** The pipeline auto-detects your dataroot layout — pass the parent folder of `v1.0-mini/` regardless of how the zip extracted.

### 3. Run the pipeline
```bash
python run_pipeline.py --dataroot /path/to/nuscenes --output ./output --max_samples 5
```

### 4. Open the dashboard
```bash
open output/bev_dashboard.html   # macOS
# Windows: double-click the file
# No server, no dependencies -- pure HTML/JS
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `BEV_GRID_SIZE` | 200x200 | BEV output resolution in pixels |
| `BEV_REAL_RANGE` | 50 m | Area covered (+-50m each axis = 100mx100m total) |
| `GRID_RES` | 0.5 m/px | Physical resolution per pixel |
| `MC_SAMPLES` | 10 | MC Dropout passes (higher = more accurate uncertainty, slower) |
| `DROPOUT_P` | 0.3 | Feature dropout probability during stochastic inference |
| `VERT_SCAN_STEPS` | 12 | Pixel samples in vertical column above each ground point |
| `VERT_SCAN_PX` | 80 | Total pixel height scanned above ground projection |
| `--max_samples` | 5 | Number of nuScenes keyframe samples to process |

---

## Metrics

| Metric | Description | Typical Value |
|---|---|---|
| **Coverage** | % of BEV cells visible to at least one camera | ~99.1% |
| **Mean Confidence** | Average predicted occupancy probability across covered cells | ~50-65% |
| **Epistemic Uncertainty** | Mean MC variance — lower = more certain | shown x10^3 |
| **Occupancy Ratio** | % of cells classified as occupied (threshold 0.55) | ~40-50% |
| **Dist-Weighted Error** | Per-cell error weighted by exp(-2d/R) — near-ego errors penalised more | ~0.20-0.40 |

---

## Dashboard Visualisations

The HTML dashboard (offline, no server) shows four co-registered BEV maps and the raw camera strip:

| Panel | Colormap | What it shows |
|---|---|---|
| **Occupancy Map** | Turbo | Blue = free space, Red/Yellow = likely obstacle |
| **Epistemic Uncertainty** | Viridis | Purple = certain, Yellow = uncertain |
| **Uncertainty-Overlay** | Fused | Red bleed = occupied but uncertain |
| **Coverage Mask** | Turbo | Which BEV cells are seen by at least one camera |
| **6-Camera Strip** | Raw | Input images from all surround cameras |

Each panel has a colour-coded legend and an insight annotation explaining the physical meaning.

---

## No External ML Framework Required

The pipeline uses **only NumPy + Pillow** — no PyTorch, no TensorFlow, no CUDA.
- Runs entirely on CPU
- ~20 seconds per sample on a MacBook
- 5-sample batch completes in under 2 minutes
- BEV grid memory footprint: 160 KB (200x200 float32)
- Dashboard output: single self-contained .html file

---

## Potential Extensions (Round 2)

- Replace column-scan heuristic with a trained ResNet encoder for learned feature extraction
- Evaluate Occupancy IoU against nuScenes LiDAR ground truth
- Add temporal fusion across consecutive frames
- Implement BEVFormer-style cross-attention spatial transformer
- Benchmark on embedded hardware (Jetson Nano / Raspberry Pi)
