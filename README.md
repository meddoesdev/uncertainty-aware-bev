# BEV 2D Occupancy — Uncertainty-Aware Prototype
### Hackathon Submission · Problem Statement 3

---

## What This Does

Transforms surround-camera images from the nuScenes dataset into a **Bird's-Eye-View (BEV)
occupancy grid** with **epistemic uncertainty quantification** via Monte Carlo Dropout.

### Novel Contribution
Most BEV systems output only a single occupancy map.  
This prototype outputs **three co-registered maps simultaneously**:
1. **Mean Occupancy** — where obstacles likely are
2. **Epistemic Uncertainty** — *how confident* the model is (via MC Dropout variance)
3. **Fused Overlay** — occupancy coloured by uncertainty (red = uncertain region)

This lets a planner know not just *where* obstacles are, but *how trustworthy* each
grid cell's prediction is — critical for safe L4 autonomy.

---

## Architecture

```
6× Camera Images (nuScenes)
        │
        ▼
Calibrated IPM Projection
(Camera intrinsics + extrinsics → world grid coords)
        │
        ▼
Bilinear Feature Sampling
(RGB features at each BEV grid cell from all visible cameras)
        │
        ▼
Multi-Camera Feature Fusion
(Weighted average across cameras that see each cell)
        │
        ▼
MC Dropout × 10 passes
(Stochastic dropout → distribution of predictions)
        │
        ▼
Mean + Variance  →  Occupancy Map + Uncertainty Map
        │
        ▼
Distance-Weighted Evaluation
(Near-ego errors penalised more heavily)
        │
        ▼
Self-contained HTML Dashboard (no server needed)
```

---

## Setup (5 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download nuScenes mini
- Go to https://www.nuscenes.org/nuscenes#download
- Download **"Mini"** split (~4 GB)
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
    ...
  sweeps/
  maps/
```

### 3. Run the pipeline
```bash
python run_pipeline.py --dataroot /path/to/nuscenes --output ./output --max_samples 5
python run_pipeline.py --dataroot /Users/medhanshnair/Downloads/v1.0-mini --output ./output --max_samples 5
```

### 4. View the dashboard
```bash
open output/bev_dashboard.html   # macOS
# or just double-click the file
```

---

## Key Parameters (in run_pipeline.py)

| Parameter      | Default | Description                          |
|----------------|---------|--------------------------------------|
| `BEV_GRID_SIZE`| 200×200 | BEV output resolution in pixels      |
| `BEV_REAL_RANGE`| 50 m   | Area covered (±50m each axis)        |
| `MC_SAMPLES`   | 10      | MC Dropout passes (↑ = more accurate uncertainty) |
| `DROPOUT_P`    | 0.3     | Dropout probability during inference |
| `--max_samples`| 5       | Number of nuScenes samples to process|

---

## Metrics

| Metric               | Description                                        |
|----------------------|----------------------------------------------------|
| **Coverage**         | % of BEV cells visible to at least one camera     |
| **Mean Confidence**  | Average predicted occupancy probability            |
| **Epistemic Uncert** | Mean MC variance — lower = more certain model      |
| **Occupancy Ratio**  | % of cells classified as occupied (threshold 0.45)|
| **Dist-Weighted Err**| Error penalising near-ego mistakes more heavily    |

---

## Visualisation

The HTML dashboard (no server, opens offline) shows:
- **BEV Occupancy** — Turbo colormap, high = more likely occupied
- **Uncertainty Map** — Viridis colormap, high = model is uncertain  
- **Fused Overlay** — Red bleed = uncertain occupied regions
- **Coverage Mask** — Which BEV cells have camera coverage
- **6-Camera Strip** — The raw camera inputs used

---

## No External ML Framework Required
The pipeline uses **only NumPy + Pillow** — no PyTorch, no TensorFlow.  
MC Dropout is simulated analytically (valid for heuristic-based occupancy functions).  
This keeps CPU runtime to **< 2 min for 5 samples** on a MacBook.

---

## Potential Extensions (Round 2)
- Replace heuristic with a trained ResNet encoder (PyTorch)
- Add temporal fusion across consecutive frames
- Implement proper IoU evaluation against LiDAR ground truth
- BEVFormer-style cross-attention spatial transformer
