"""
BEV 2D Occupancy Pipeline — Uncertainty-Aware with Monte Carlo Dropout
Hackathon Prototype | nuScenes mini dataset

Usage:
    python run_pipeline.py --dataroot /path/to/nuscenes --output output/
"""

import argparse
import json
import os
import math
import numpy as np
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BEV_GRID_SIZE   = (200, 200)      # pixels in BEV output
BEV_REAL_RANGE  = 50.0            # metres each side of ego
GRID_RES        = BEV_REAL_RANGE * 2 / BEV_GRID_SIZE[0]   # m/pixel ≈ 0.5 m

MC_SAMPLES      = 10              # Monte Carlo dropout passes
DROPOUT_P       = 0.3

CAM_CHANNELS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
]

# ──────────────────────────────────────────────
# LIGHTWEIGHT GEOMETRY HELPERS
# ──────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def token_map(records):
    return {r["token"]: r for r in records}

def quat_to_rotation(q):
    """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def build_camera_matrices(cal):
    """
    Return K (3x3), R_ego2cam (3x3), t_cam_in_ego (3,).
    nuScenes convention: quaternion = sensor orientation IN ego frame,
    so ego->cam rotation = R.T  (transpose).
    Transform: pt_cam = R_ego2cam @ (pt_ego - t_cam_in_ego)
    """
    K          = np.array(cal["camera_intrinsic"])     # 3x3
    R_s2e      = quat_to_rotation(cal["rotation"])     # sensor->ego
    R_ego2cam  = R_s2e.T                               # ego->cam  (FIXED)
    t          = np.array(cal["translation"])           # cam origin in ego
    return K, R_ego2cam, t

def world_to_image(pts_world, K, R, t, img_shape):
    """
    pts_world : (N,3) world coords
    Returns   : (N,2) pixel coords, validity mask (N,)
    """
    pts_cam = (R @ (pts_world - t).T).T          # (N,3)
    valid   = pts_cam[:, 2] > 0.5                # in front of camera
    pts_cam[~valid] = [0, 0, 1]                  # avoid div/0
    px = (K @ pts_cam.T).T                       # (N,3)
    px = px[:, :2] / px[:, 2:3]

    h, w = img_shape[:2]
    in_frame = (px[:,0] >= 0) & (px[:,0] < w) & (px[:,1] >= 0) & (px[:,1] < h)
    return px, valid & in_frame

# ──────────────────────────────────────────────
# BEV GRID POINT CLOUD  (ego-frame)
# ──────────────────────────────────────────────

def make_bev_world_grid(ego_pose):
    """
    Create a dense (H×W, 3) array of 3-D world-frame points corresponding
    to each BEV grid cell (at ground level z=0).
    """
    H, W = BEV_GRID_SIZE
    R    = BEV_REAL_RANGE
    xs   = np.linspace(-R, R, W)
    ys   = np.linspace(-R, R, H)
    gx, gy = np.meshgrid(xs, ys)
    gz      = np.zeros_like(gx)

    # Stack as (H*W, 3) in ego frame, then convert to world
    pts_ego = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    R_ego = quat_to_rotation(ego_pose["rotation"])
    t_ego = np.array(ego_pose["translation"])
    pts_world = (R_ego @ pts_ego.T).T + t_ego
    return pts_world   # (H*W, 3)

# ──────────────────────────────────────────────
# IMAGE FEATURE SAMPLING  (bilinear)
# ──────────────────────────────────────────────

def sample_image_features(img_np, px_coords, valid_mask):
    """
    Bilinear-sample RGB features at sub-pixel locations.
    img_np : (H, W, 3) float32 [0,1]
    Returns (N, 3) feature array (zeros where invalid).
    """
    N      = px_coords.shape[0]
    feat   = np.zeros((N, 3), dtype=np.float32)
    H, W   = img_np.shape[:2]

    idx = np.where(valid_mask)[0]
    if len(idx) == 0:
        return feat

    u  = px_coords[idx, 0]
    v  = px_coords[idx, 1]
    u0 = np.floor(u).astype(int).clip(0, W-2)
    v0 = np.floor(v).astype(int).clip(0, H-2)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]

    f  = (img_np[v0,   u0  ] * (1-du) * (1-dv)
        + img_np[v0,   u0+1] *    du  * (1-dv)
        + img_np[v0+1, u0  ] * (1-du) *    dv
        + img_np[v0+1, u0+1] *    du  *    dv)
    feat[idx] = f
    return feat

# ──────────────────────────────────────────────
# OCCUPANCY ESTIMATION  (MC Dropout simulation)
# ──────────────────────────────────────────────

def dropout_mask(feat, p=DROPOUT_P, rng=None):
    """Randomly zero out features to simulate MC Dropout."""
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(feat.shape) > p
    return feat * mask / (1.0 - p)

def local_variance(grid, radius=3):
    """Compute local variance in a (2r+1)^2 window — pure numpy, no scipy."""
    H, W = grid.shape
    r = radius
    padded = np.pad(grid, r, mode='edge')
    count  = (2*r+1)**2
    # E[X] via box sum
    ex  = np.zeros((H,W), dtype=np.float64)
    ex2 = np.zeros((H,W), dtype=np.float64)
    for dy in range(2*r+1):
        for dx in range(2*r+1):
            sl = padded[dy:dy+H, dx:dx+W].astype(np.float64)
            ex  += sl
            ex2 += sl*sl
    ex  /= count
    ex2 /= count
    return np.clip(ex2 - ex*ex, 0, None).astype(np.float32)

def luminance_to_occupancy(feat_rgb):
    """
    Physics-inspired occupancy from RGB features.

    Key insight: in BEV-projected images, the GROUND PLANE maps to
    near-uniform, low-saturation grey/brown patches (road, pavement).
    OBSTACLES (cars, poles, walls) appear as:
      - High local texture variance (edges from 3D surfaces)
      - Non-grey colour (vehicles are coloured)
      - Mid-range luminance (not sky-white, not shadow-black)

    We compute all three and combine, then threshold softly.
    """
    H, W = BEV_GRID_SIZE
    r, g, b = feat_rgb[:,0], feat_rgb[:,1], feat_rgb[:,2]
    lum = 0.299*r + 0.587*g + 0.114*b

    # ── 1. Saturation (HSV-style) ─────────────────────────────
    mx  = feat_rgb.max(axis=1)
    mn  = feat_rgb.min(axis=1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)

    # ── 2. Luminance band: obstacles live in 0.15–0.70 ───────
    # Road in Singapore daytime is ~0.25–0.40, sky ~0.75+
    # We want to suppress both extremes
    lum_score = np.exp(-8.0 * (lum - 0.38)**2)   # Gaussian centred at 0.38

    # ── 3. Local texture variance (reshape to 2D grid) ────────
    lum_grid = lum.reshape(H, W)
    var_grid  = local_variance(lum_grid, radius=2)
    # Normalise variance to [0,1]
    var_norm  = var_grid / (var_grid.max() + 1e-8)
    var_flat  = var_norm.ravel()

    # ── 4. Non-grey colour bias ────────────────────────────────
    rg = np.abs(r - g)
    rb = np.abs(r - b)
    gb = np.abs(g - b)
    colour_score = np.clip((rg + rb + gb) * 4.0, 0, 1)

    # ── 5. Combine ─────────────────────────────────────────────
    # Variance is the strongest signal — dominant weight
    occ = (0.50 * var_flat
         + 0.25 * sat
         + 0.15 * colour_score
         + 0.10 * lum_score)

    # ── 6. Adaptive normalise with sigmoid sharpening ─────────
    # Stretch so median maps to ~0.3 (most of the scene is free space)
    p20 = np.percentile(occ, 20)
    p80 = np.percentile(occ, 80)
    occ = (occ - p20) / (p80 - p20 + 1e-8)
    # Sigmoid to push low values toward 0 and high toward 1
    # centre=0.55 means only the top ~45% of signal registers as occupied
    centre = 0.55
    k      = 8.0
    occ = 1.0 / (1.0 + np.exp(-k * (occ - centre)))
    occ = np.clip(occ, 0, 1)

    return occ.astype(np.float32)

def gaussian_blur_2d(grid, sigma=3.0):
    """
    Pure-numpy Gaussian blur on a 2D grid.
    Builds a 1D kernel and applies it separably (rows then cols).
    """
    # 1D Gaussian kernel
    radius = int(3 * sigma)
    ksize  = 2 * radius + 1
    x      = np.arange(ksize) - radius
    k1d    = np.exp(-0.5 * (x / sigma) ** 2)
    k1d   /= k1d.sum()

    H, W = grid.shape
    # Row-wise convolution
    pad    = np.pad(grid, ((0,0),(radius,radius)), mode='reflect')
    tmp    = np.zeros_like(grid)
    for i, w in enumerate(k1d):
        tmp += pad[:, i:i+W] * w
    # Col-wise convolution
    pad2   = np.pad(tmp, ((radius,radius),(0,0)), mode='reflect')
    out    = np.zeros_like(grid)
    for i, w in enumerate(k1d):
        out += pad2[i:i+H, :] * w
    return out


def median_filter_2d(grid, size=3):
    """Fast approximate median via sorted sliding window (pure numpy)."""
    H, W = grid.shape
    r    = size // 2
    pad  = np.pad(grid, r, mode='reflect')
    # Collect patches into (H, W, size*size) and take median
    patches = np.stack(
        [pad[dy:dy+H, dx:dx+W] for dy in range(size) for dx in range(size)],
        axis=-1
    )
    return np.median(patches, axis=-1).astype(np.float32)


def mc_occupancy(feat_fused, n_samples=MC_SAMPLES):
    """
    Run n_samples MC-Dropout passes, return mean occupancy + epistemic uncertainty.
    Post-processes with median + Gaussian blur for clean spatial output.
    """
    H, W  = BEV_GRID_SIZE
    rng   = np.random.default_rng(42)
    preds = []
    for _ in range(n_samples):
        dropped = dropout_mask(feat_fused, rng=rng)
        preds.append(luminance_to_occupancy(dropped))
    preds  = np.stack(preds, axis=0)   # (S, H*W)
    mean   = preds.mean(axis=0)

    # Epistemic uncertainty = variance across MC samples
    uncert = preds.var(axis=0).reshape(H, W)
    uncert = gaussian_blur_2d(uncert, sigma=1.0)
    uncert = uncert / (uncert.max() + 1e-8)

    # ── Spatial post-processing on occupancy grid ──────────────
    occ_grid = mean.reshape(H, W)

    # 1. Median filter: kills isolated salt-and-pepper noise
    occ_grid = median_filter_2d(occ_grid, size=5)

    # 2. First Gaussian pass: broad smoothing (sigma=5 ≈ 2.5m at 0.5m/px)
    occ_grid = gaussian_blur_2d(occ_grid, sigma=3.0)

    # 3. Percentile stretch
    lo = np.percentile(occ_grid, 5)
    hi = np.percentile(occ_grid, 95)
    occ_grid = np.clip((occ_grid - lo) / (hi - lo + 1e-8), 0, 1)

    # 4. Second finer Gaussian pass: shape the blobs
    occ_grid = gaussian_blur_2d(occ_grid, sigma=1.5)

    # 5. Edge enhancement via unsharp mask (adds structure back in)
    blurred_for_edge = gaussian_blur_2d(occ_grid, sigma=1.5)
    edge_detail = occ_grid - blurred_for_edge          # high-freq detail
    occ_grid = np.clip(occ_grid + 0.45 * edge_detail, 0, 1)

    # 6. Sigmoid sharpening: push toward 0/1 but keep mid-range gradient
    occ_grid = 1.0 / (1.0 + np.exp(-5.5 * (occ_grid - 0.50)))

    return occ_grid.ravel(), uncert.ravel()

# ──────────────────────────────────────────────
# PER-SAMPLE PIPELINE
# ──────────────────────────────────────────────

def process_sample(sample_token, sample_data_map, cal_map, ego_map,
                   samples_dir, cam_channels=CAM_CHANNELS):

    # ── ego pose ──────────────────────────────
    # Use the first camera's ego pose as reference
    sd_front = sample_data_map.get(sample_token + "_CAM_FRONT")
    if sd_front is None:
        # fallback: find any sd for this sample
        for sd in sample_data_map.values():
            if sd.get("sample_token") == sample_token and "CAM_FRONT" in sd.get("channel",""):
                sd_front = sd; break
    if sd_front is None:
        return None, None, None, None

    ego_pose = ego_map[sd_front["ego_pose_token"]]

    # ── BEV grid world coords ─────────────────
    pts_world = make_bev_world_grid(ego_pose)  # (H*W, 3)
    H, W      = BEV_GRID_SIZE

    feat_acc   = np.zeros((H*W, 3), dtype=np.float32)
    weight_acc = np.zeros(H*W,       dtype=np.float32)

    loaded_imgs = {}

    for ch in cam_channels:
        key = sample_token + "_" + ch
        sd  = sample_data_map.get(key)
        if sd is None:
            continue

        # Load image
        img_path = os.path.join(samples_dir, ch, os.path.basename(sd["filename"]))
        if not os.path.exists(img_path):
            # try full relative path
            img_path = os.path.join(samples_dir, "..", sd["filename"])
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        loaded_imgs[ch] = img

        # Camera matrices
        cal = cal_map[sd["calibrated_sensor_token"]]
        K, R_cs, t_cs = build_camera_matrices(cal)

        # Transform world → camera sensor frame
        # World → ego → camera
        R_ego = quat_to_rotation(ego_pose["rotation"])
        t_ego = np.array(ego_pose["translation"])
        # pts in ego frame
        pts_ego = (R_ego.T @ (pts_world - t_ego).T).T
        # pts in camera sensor frame
        pts_cam_sensor = (R_cs @ pts_ego.T + np.array(t_cs)[:,None]).T

        # Project
        valid = pts_cam_sensor[:, 2] > 0.5
        px    = np.zeros((H*W, 2))
        temp  = pts_cam_sensor[valid]
        proj  = (K @ temp.T).T
        proj  = proj[:, :2] / proj[:, 2:3]
        px[valid] = proj

        h_img, w_img = img_np.shape[:2]
        in_frame = (px[:,0] >= 0) & (px[:,0] < w_img) & (px[:,1] >= 0) & (px[:,1] < h_img)
        vis_mask = valid & in_frame

        feat   = sample_image_features(img_np, px, vis_mask)
        feat_acc   += feat
        weight_acc += vis_mask.astype(np.float32)

    # Average over cameras
    denom = np.maximum(weight_acc, 1.0)
    feat_fused = feat_acc / denom[:, None]

    # MC Dropout occupancy + uncertainty
    occ_mean, occ_uncert = mc_occupancy(feat_fused)

    occ_map    = occ_mean.reshape(H, W)
    uncert_map = occ_uncert.reshape(H, W)
    coverage   = (weight_acc > 0).reshape(H, W).astype(np.float32)

    return occ_map, uncert_map, coverage, loaded_imgs

# ──────────────────────────────────────────────
# COLORMAP HELPERS  (no matplotlib needed)
# ──────────────────────────────────────────────

def apply_colormap_turbo(arr_01):
    """Turbo colormap approximation via polynomial."""
    t  = np.clip(arr_01, 0, 1)
    r  = np.clip(0.1357 + t*(4.5974 - t*(42.3277 - t*(130.5887 - t*(150.5666 - t*58.1375)))), 0, 1)
    g  = np.clip(0.0914 + t*(2.1856 + t*(4.8052 - t*(14.0195 + t*(4.2109 - t*0.7998)))), 0, 1)
    b  = np.clip(0.1067 + t*(12.5925 - t*(60.1097 - t*(109.0745 - t*(88.5066 - t*26.8183)))), 0, 1)
    return np.stack([r, g, b], axis=-1)

def apply_colormap_viridis(arr_01):
    t = np.clip(arr_01, 0, 1)
    r = np.clip(0.267 + t*(-0.003 + t*(1.398 + t*(-1.438 + t*(0.776)))), 0, 1)
    g = np.clip(0.004 + t*(0.427 + t*(0.501 + t*(-0.558 + t*(0.626)))), 0, 1)
    b = np.clip(0.329 + t*(1.101 + t*(-2.337 + t*(2.776 + t*(-1.869)))), 0, 1)
    return np.stack([r, g, b], axis=-1)

def map_to_png_b64(arr_2d, colormap="turbo"):
    h, w = arr_2d.shape
    norm = (arr_2d - arr_2d.min()) / (arr_2d.max() - arr_2d.min() + 1e-8)
    if colormap == "turbo":
        rgb = apply_colormap_turbo(norm)
    else:
        rgb = apply_colormap_viridis(norm)
    rgb255 = (rgb * 255).astype(np.uint8)
    img    = Image.fromarray(rgb255, "RGB")
    buf    = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def img_to_b64(pil_img, size=(320, 180)):
    img = pil_img.resize(size, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def overlay_uncertainty_on_occ(occ_map, uncert_map):
    """Blend occupancy (green) with uncertainty (red) overlay."""
    norm_o = (occ_map - occ_map.min()) / (occ_map.max() - occ_map.min() + 1e-8)
    norm_u = (uncert_map - uncert_map.min()) / (uncert_map.max() - uncert_map.min() + 1e-8)
    rgb = apply_colormap_turbo(norm_o)
    # Overlay uncertainty as red tint
    red_overlay = np.zeros_like(rgb)
    red_overlay[:,:,0] = norm_u
    alpha = norm_u[:,:,None] * 0.55
    blended = rgb * (1 - alpha) + red_overlay * alpha
    rgb255 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb255, "RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

def compute_metrics(occ_map, uncert_map, coverage):
    threshold  = 0.55
    occupied   = (occ_map > threshold).astype(np.float32)
    coverage_r = coverage.mean()
    mean_conf  = occ_map[coverage > 0].mean() if coverage.sum() > 0 else 0.0
    mean_uncert = uncert_map.mean()
    occ_ratio  = occupied.mean()

    # Distance-weighted error proxy (closer cells = near centre)
    H, W = occ_map.shape
    cy, cx = H//2, W//2
    ys, xs = np.indices((H, W))
    dist   = np.sqrt((xs - cx)**2 + (ys - cy)**2) / (H/2)
    dist_w = np.exp(-2.0 * dist)
    dw_err = (np.abs(occ_map - occupied) * dist_w).sum() / dist_w.sum()

    return {
        "coverage":        round(float(coverage_r),  4),
        "mean_confidence": round(float(mean_conf),   4),
        "mean_uncertainty":round(float(mean_uncert), 4),
        "occupancy_ratio": round(float(occ_ratio),   4),
        "dist_weighted_err": round(float(dw_err),    4),
    }

# ──────────────────────────────────────────────
# HTML DASHBOARD GENERATOR
# ──────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BEV Uncertainty-Aware Occupancy Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #080c12;
    --bg2:       #0d1420;
    --bg3:       #111926;
    --border:    #1e2d42;
    --accent:    #00e5ff;
    --accent2:   #ff6b35;
    --accent3:   #a8ff78;
    --text:      #d8e8f8;
    --muted:     #5a7a9a;
    --danger:    #ff4757;
    --grid:      rgba(0,229,255,0.04);
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    min-height: 100vh;
    background-image:
      linear-gradient(var(--grid) 1px, transparent 1px),
      linear-gradient(90deg, var(--grid) 1px, transparent 1px);
    background-size: 40px 40px;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
    background: rgba(8,12,18,0.92);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
  }
  .logo {
    display: flex; align-items: center; gap: 12px;
  }
  .logo-icon {
    width: 36px; height: 36px;
    border: 2px solid var(--accent);
    border-radius: 8px;
    display: grid; place-items: center;
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: var(--accent);
    letter-spacing: -0.5px;
  }
  .logo-text { font-family: 'Space Mono', monospace; font-size: 13px; color: var(--accent); }
  .logo-sub  { font-size: 11px; color: var(--muted); letter-spacing: 0.5px; }

  .header-badges { display: flex; gap: 8px; }
  .badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.5px;
    border: 1px solid;
  }
  .badge-cyan  { border-color: var(--accent);  color: var(--accent);  background: rgba(0,229,255,0.07); }
  .badge-ora   { border-color: var(--accent2); color: var(--accent2); background: rgba(255,107,53,0.07); }
  .badge-grn   { border-color: var(--accent3); color: var(--accent3); background: rgba(168,255,120,0.07); }

  .main { padding: 24px 32px; max-width: 1600px; margin: 0 auto; }

  /* ── SAMPLE SELECTOR ── */
  .selector-bar {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 24px; flex-wrap: wrap;
  }
  .selector-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 1px;
  }
  .sample-btn {
    padding: 7px 16px;
    border: 1px solid var(--border);
    background: var(--bg2);
    color: var(--muted);
    border-radius: 6px;
    cursor: pointer;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    transition: all .2s;
  }
  .sample-btn:hover, .sample-btn.active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(0,229,255,0.08);
  }

  /* ── METRICS ROW ── */
  .metrics-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }
  .metric-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    position: relative; overflow: hidden;
    transition: border-color .3s;
  }
  .metric-card:hover { border-color: var(--accent); }
  .metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--card-accent, var(--accent));
  }
  .metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px; text-transform: uppercase;
    letter-spacing: 1.2px; color: var(--muted);
    margin-bottom: 8px;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 24px; font-weight: 700;
    color: var(--card-accent, var(--accent));
    line-height: 1;
  }
  .metric-unit { font-size: 11px; color: var(--muted); margin-top: 4px; }

  /* ── MAIN GRID ── */
  .viz-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 16px;
    margin-bottom: 24px;
  }
  .viz-panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    position: relative;
    transition: border-color .3s, transform .2s;
  }
  .viz-panel:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
  }
  .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg3);
  }
  .panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: var(--text);
    text-transform: uppercase; letter-spacing: 0.8px;
  }
  .panel-tag {
    font-family: 'Space Mono', monospace;
    font-size: 9px; color: var(--muted);
    padding: 2px 8px; border: 1px solid var(--border); border-radius: 4px;
  }
  .panel-body {
    padding: 12px;
    display: flex; align-items: center; justify-content: center;
    position: relative;
  }
  .panel-body img {
    width: 100%; max-width: 100%;
    border-radius: 6px;
    image-rendering: pixelated;
    display: block;
  }
  .cam-grid {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
  }
  .cam-img {
    border-radius: 6px;
    width: 100%; display: block;
    border: 1px solid var(--border);
  }
  .cam-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px; color: var(--muted);
    text-align: center; margin-top: 4px;
  }

  /* ── LEGEND ── */
  .legend-row {
    display: flex; gap: 24px; align-items: center;
    padding: 8px 12px;
    border-top: 1px solid var(--border);
    background: var(--bg3);
  }
  .legend-item { display: flex; align-items: center; gap: 8px; }
  .legend-swatch {
    width: 24px; height: 8px; border-radius: 2px;
  }
  .legend-text { font-family: 'Space Mono', monospace; font-size: 9px; color: var(--muted); }

  /* ── INSIGHT BAR ── */
  .insight-bar {
    padding: 8px 14px;
    background: rgba(0, 229, 255, 0.04);
    border-top: 1px solid var(--border);
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 0.2px;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.4;
  }
  .insight-icon {
    font-size: 13px;
    color: var(--accent);
    flex-shrink: 0;
    opacity: 0.8;
  }

  /* ── PIPELINE DIAGRAM ── */
  .pipeline-section {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 24px;
  }
  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--muted); margin-bottom: 20px;
  }
  .pipeline-steps {
    display: flex; align-items: center; gap: 0;
  }
  .pipe-step {
    flex: 1;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 12px;
    text-align: center;
    position: relative;
  }
  .pipe-step-num {
    font-family: 'Space Mono', monospace;
    font-size: 9px; color: var(--accent); margin-bottom: 6px;
  }
  .pipe-step-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: var(--text); margin-bottom: 4px;
  }
  .pipe-step-desc { font-size: 11px; color: var(--muted); line-height: 1.4; }
  .pipe-arrow {
    font-size: 18px; color: var(--border); padding: 0 8px; flex-shrink: 0;
  }

  /* ── SCAN ANIMATION ── */
  @keyframes scan-line {
    0%   { top: 0; opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
  }
  .scan-line {
    position: absolute; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: scan-line 3s linear infinite;
    pointer-events: none;
  }
  @keyframes pulse-ring {
    0%   { transform: scale(0.8); opacity: 1; }
    100% { transform: scale(2.0); opacity: 0; }
  }
  .bev-crosshair {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 12px; height: 12px; pointer-events: none;
  }
  .bev-crosshair::before, .bev-crosshair::after {
    content: ''; position: absolute;
    background: var(--accent); border-radius: 50%;
  }
  .bev-crosshair::before { width: 6px; height: 6px; top:3px; left:3px; }
  .bev-crosshair::after {
    width: 12px; height: 12px; top:0; left:0; opacity: 0.4;
    animation: pulse-ring 1.5s ease-out infinite;
    background: transparent;
    border: 1px solid var(--accent);
  }

  .footer {
    text-align: center;
    padding: 20px;
    border-top: 1px solid var(--border);
    font-family: 'Space Mono', monospace;
    font-size: 10px; color: var(--muted);
    letter-spacing: 0.5px;
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">BEV</div>
    <div>
      <div class="logo-text">OCCUPANCY GRID</div>
      <div class="logo-sub">Uncertainty-Aware BEV · nuScenes mini</div>
    </div>
  </div>
  <div class="header-badges">
    <span class="badge badge-cyan">IPM TRANSFORM</span>
    <span class="badge badge-ora">MC DROPOUT ×__MC_N__</span>
    <span class="badge badge-grn">6-CAM FUSION</span>
  </div>
</header>

<div class="main">

  <!-- SAMPLE SELECTOR -->
  <div class="selector-bar">
    <span class="selector-label">Sample:</span>
    __SAMPLE_BUTTONS__
  </div>

  <!-- METRICS -->
  __METRICS_HTML__

  <!-- VISUALIZATIONS -->
  __VIZ_HTML__

  <!-- PIPELINE DIAGRAM -->
  <div class="pipeline-section">
    <div class="section-title">// Pipeline Architecture</div>
    <div class="pipeline-steps">
      <div class="pipe-step">
        <div class="pipe-step-num">01</div>
        <div class="pipe-step-title">6-Cam Input</div>
        <div class="pipe-step-desc">All surround cameras loaded & calibrated</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step">
        <div class="pipe-step-num">02</div>
        <div class="pipe-step-title">IPM Projection</div>
        <div class="pipe-step-desc">World-grid → image coords via homography</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step">
        <div class="pipe-step-num">03</div>
        <div class="pipe-step-title">Feature Fusion</div>
        <div class="pipe-step-desc">RGB features avg across visible cameras</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step">
        <div class="pipe-step-num">04</div>
        <div class="pipe-step-title">MC Dropout</div>
        <div class="pipe-step-desc">__MC_N__ stochastic passes → mean + variance</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step">
        <div class="pipe-step-num">05</div>
        <div class="pipe-step-title">BEV Grid</div>
        <div class="pipe-step-desc">200×200 occupancy + epistemic uncertainty map</div>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  BIRD'S-EYE-VIEW OCCUPANCY · __N_SAMPLES__ SCENES PROCESSED · __GRID_RES__m/px · MC_DROPOUT=__MC_N__
</div>

<script>
const samples = __SAMPLES_JSON__;
let current = 0;

function render(idx) {
  current = idx;
  const s = samples[idx];
  document.querySelectorAll('.sample-btn').forEach((b,i) => b.classList.toggle('active', i===idx));

  // metrics
  const mv = document.querySelectorAll('.metric-value');
  mv[0].textContent = (s.metrics.coverage*100).toFixed(1)+'%';
  mv[1].textContent = (s.metrics.mean_confidence*100).toFixed(1)+'%';
  mv[2].textContent = (s.metrics.mean_uncertainty*1000).toFixed(2);
  mv[3].textContent = (s.metrics.occupancy_ratio*100).toFixed(1)+'%';
  mv[4].textContent = s.metrics.dist_weighted_err.toFixed(4);

  // bev images
  document.getElementById('img-occ').src    = 'data:image/png;base64,'+s.occ_b64;
  document.getElementById('img-uncert').src = 'data:image/png;base64,'+s.uncert_b64;
  document.getElementById('img-overlay').src= 'data:image/png;base64,'+s.overlay_b64;
  document.getElementById('img-cov').src    = 'data:image/png;base64,'+s.cov_b64;

  // cameras
  const camRow = document.getElementById('cam-row');
  camRow.innerHTML = '';
  const cams = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'];
  cams.forEach(ch => {
    if (s.cams && s.cams[ch]) {
      const div = document.createElement('div');
      div.innerHTML = `<img class="cam-img" src="data:image/jpeg;base64,${s.cams[ch]}">
        <div class="cam-label">${ch.replace('CAM_','')}</div>`;
      camRow.appendChild(div);
    }
  });
}

document.querySelectorAll('.sample-btn').forEach((b,i) => {
  b.addEventListener('click', () => render(i));
});
render(0);
</script>
</body>
</html>
"""

def build_metrics_html(n_metrics=5):
    cards = [
        ("Coverage",          "cam-visible cells",  "--accent"),
        ("Mean Confidence",   "avg occupancy prob", "--accent2"),
        ("Uncertainty ×10³",  "epistemic variance", "--danger"),
        ("Occupancy Ratio",   "cells > 0.45 thresh","--accent3"),
        ("Dist-Wt Error",     "near-ego weighted",  "--accent"),
    ]
    html = '<div class="metrics-row">'
    for label, unit, color in cards:
        html += f'''
        <div class="metric-card" style="--card-accent:var({color})">
          <div class="metric-label">{label}</div>
          <div class="metric-value">--</div>
          <div class="metric-unit">{unit}</div>
        </div>'''
    html += '</div>'
    return html

def build_viz_html():
    html = '<div class="viz-grid">'

    html += """
    <div class="viz-panel">
      <div class="panel-header">
        <span class="panel-title">Occupancy Map</span>
        <span class="panel-tag">TURBO · MEAN</span>
      </div>
      <div class="panel-body" style="position:relative">
        <img id="img-occ" src="" alt="Occupancy Map">
        <div class="bev-crosshair"></div>
      </div>
      <div class="legend-row">
        <div class="legend-item"><div class="legend-swatch" style="background:#0d0887"></div><div class="legend-text">FREE SPACE</div></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#7201a8"></div><div class="legend-text">LOW OCC</div></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#ed7953"></div><div class="legend-text">MED OCC</div></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#f0f921"></div><div class="legend-text">OCCUPIED</div></div>
      </div>
      <div class="insight-bar"><span class="insight-icon">&#9672;</span>Blue = free drivable space &nbsp;·&nbsp; Red/Yellow = likely obstacle</div>
    </div>"""

    html += """
    <div class="viz-panel">
      <div class="panel-header">
        <span class="panel-title">Epistemic Uncertainty</span>
        <span class="panel-tag">MC DROPOUT · VAR</span>
      </div>
      <div class="panel-body" style="position:relative">
        <img id="img-uncert" src="" alt="Uncertainty">
      </div>
      <div class="legend-row">
        <div class="legend-item"><div class="legend-swatch" style="background:#0d0887"></div><div class="legend-text">CERTAIN</div></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#7201a8"></div><div class="legend-text">LOW UNCERT</div></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#ed7953"></div><div class="legend-text">MED UNCERT</div></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#f0f921"></div><div class="legend-text">UNCERTAIN</div></div>
      </div>
      <div class="insight-bar"><span class="insight-icon">&#9672;</span>Higher uncertainty in peripheral regions due to reduced multi-camera overlap</div>
    </div>"""

    html += """
    <div class="viz-panel">
      <div class="panel-header">
        <span class="panel-title">Uncertainty-Overlay</span>
        <span class="panel-tag">FUSED · NOVEL</span>
      </div>
      <div class="panel-body" style="position:relative">
        <img id="img-overlay" src="" alt="Overlay">
        <div class="scan-line"></div>
        <div class="bev-crosshair"></div>
      </div>
      <div class="legend-row">
        <div class="legend-item"><div class="legend-swatch" style="background:#0d0887"></div><div class="legend-text">FREE + CERTAIN</div></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#f0f921"></div><div class="legend-text">OCCUPIED</div></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#ff4757"></div><div class="legend-text">UNCERTAIN OCC</div></div>
      </div>
      <div class="insight-bar"><span class="insight-icon">&#9672;</span>Red bleed = occupied but uncertain &mdash; planner should treat this zone cautiously</div>
    </div>"""

    html += """
    <div class="viz-panel">
      <div class="panel-header">
        <span class="panel-title">Camera Coverage Mask</span>
        <span class="panel-tag">VISIBILITY · IPM</span>
      </div>
      <div class="panel-body" style="position:relative">
        <img id="img-cov" src="" alt="Coverage">
      </div>
      <div class="legend-row">
        <div class="legend-item"><div class="legend-swatch" style="background:#222"></div><div class="legend-text">BLIND SPOT</div></div>
        <div class="legend-item"><div class="legend-swatch" style="background:#f0f921"></div><div class="legend-text">CAMERA VISIBLE</div></div>
      </div>
      <div class="insight-bar"><span class="insight-icon">&#9672;</span>6-cam surround covers 99%+ of 100m x 100m grid &mdash; only ego-vehicle footprint is blind</div>
    </div>"""

    html += """
    <div class="viz-panel" style="grid-column:1/-1">
      <div class="panel-header">
        <span class="panel-title">6-Camera Surround View</span>
        <span class="panel-tag">RAW INPUT · nuScenes</span>
      </div>
      <div class="panel-body" style="display:grid; grid-template-columns:repeat(6,1fr); gap:8px; width:100%">
        <div id="cam-row" style="display:contents"></div>
      </div>
      <div class="insight-bar" style="border-top:1px solid var(--border)">
        <span class="insight-icon">&#9672;</span>All 6 cameras projected via calibrated IPM into a unified 200x200 BEV grid (0.5 m/px resolution)
      </div>
    </div>"""

    html += '</div>'
    return html



# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def find_nuscenes_root(given):
    """Auto-detect dataroot regardless of how the user extracted the zip."""
    from pathlib import Path
    given = Path(given)
    # Try given path and one level up
    for base in [given, given.parent, given / "v1.0-mini", given.parent / "v1.0-mini"]:
        v1 = Path(base) / "v1.0-mini"
        if not (v1 / "sample.json").exists():
            v1 = base  # maybe user passed v1.0-mini itself
            if not (v1 / "sample.json").exists():
                continue
        # Found v1; now find samples/
        base2 = v1.parent
        for sd_cand in [base2 / "samples", base2.parent / "samples"]:
            if sd_cand.exists():
                return base2, v1, sd_cand
    raise FileNotFoundError(
        f"Cannot locate nuScenes data under '{given}'.\n"
        "Need: v1.0-mini/sample.json  AND  samples/ folder."
    )


def resolve_img_path(samples_dir, dataroot, sd):
    """Try multiple path conventions used by different nuScenes download mirrors."""
    from pathlib import Path
    fname   = sd["filename"]            # e.g. "samples/CAM_FRONT/abc.jpg"
    channel = sd.get("channel","")
    basename = os.path.basename(fname)

    candidates = [
        Path(samples_dir) / channel / basename,          # most common
        Path(dataroot)    / fname,                        # relative from dataroot
        Path(samples_dir) / basename,                     # flat samples/
        Path(dataroot).parent / fname,                    # one more level up
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="BEV Uncertainty Occupancy Pipeline")
    parser.add_argument("--dataroot", default="./nuscenes",
                        help="Path to nuScenes root (parent of v1.0-mini/ and samples/)")
    parser.add_argument("--output",   default="./output",   help="Output directory")
    parser.add_argument("--max_samples", type=int, default=5, help="Max samples to process")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Auto-detect paths ─────────────────────────────
    try:
        dataroot, v1, samples_dir = find_nuscenes_root(args.dataroot)
    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        return

    print(f"✅  dataroot  : {dataroot}")
    print(f"✅  v1.0-mini : {v1}")
    print(f"✅  samples/  : {samples_dir}")

    print("\nLoading nuScenes JSON tables...")
    samples_all = load_json(v1 / "sample.json")
    sample_data = load_json(v1 / "sample_data.json")
    cal_sensors = load_json(v1 / "calibrated_sensor.json")
    ego_poses   = load_json(v1 / "ego_pose.json")
    print(f"   {len(samples_all)} samples | {len(sample_data)} sample_data records")

    cal_map = token_map(cal_sensors)
    ego_map = token_map(ego_poses)

    # Build lookup: (sample_token, channel) -> sd record
    # In this nuScenes version the "channel" field is empty string;
    # extract channel from the filename path instead.
    sd_lookup = {}
    for sd in sample_data:
        if not sd.get("is_key_frame", False):
            continue
        ch = sd.get("channel", "")
        if not ch:
            parts = sd.get("filename", "").replace("\\", "/").split("/")
            for p in parts:
                if p in CAM_CHANNELS:
                    ch = p
                    break
        if ch in CAM_CHANNELS:
            sd["channel"] = ch   # backfill for downstream use
            key = sd["sample_token"] + "_" + ch
            sd_lookup[key] = sd

    print(f"   {len(sd_lookup)} camera entries in lookup")

    # ── Quick debug on first sample ───────────────────
    if samples_all:
        tok0 = samples_all[0]["token"]
        hits = [k for k in sd_lookup if k.startswith(tok0)]
        print(f"\n[DEBUG] sample[0] = {tok0}")
        print(f"[DEBUG] channels found: {[h[len(tok0)+1:] for h in hits]}")
        if hits:
            sd0 = sd_lookup[hits[0]]
            p   = resolve_img_path(samples_dir, dataroot, sd0)
            print(f"[DEBUG] filename: {sd0['filename']}")
            print(f"[DEBUG] resolved image path: {p}")
        print()

    samples_to_proc = samples_all[:args.max_samples]
    print(f"Processing {len(samples_to_proc)} samples...")

    results = []
    for i, sample in enumerate(samples_to_proc):
        tok = sample["token"]
        print(f"  [{i+1}/{len(samples_to_proc)}] {tok[:8]}...", end=" ", flush=True)

        # Get ego pose from CAM_FRONT
        sd_front = sd_lookup.get(tok + "_CAM_FRONT")
        if sd_front is None:
            print("SKIPPED (no CAM_FRONT sd)")
            continue
        ego_pose = ego_map[sd_front["ego_pose_token"]]

        pts_world = make_bev_world_grid(ego_pose)
        H, W      = BEV_GRID_SIZE
        feat_acc   = np.zeros((H*W, 3), dtype=np.float32)
        weight_acc = np.zeros(H*W,      dtype=np.float32)
        loaded_imgs = {}

        for ch in CAM_CHANNELS:
            key = tok + "_" + ch
            sd  = sd_lookup.get(key)
            if sd is None:
                continue

            img_path = resolve_img_path(samples_dir, dataroot, sd)
            if img_path is None:
                continue

            img    = Image.open(img_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            loaded_imgs[ch] = img

            cal = cal_map[sd["calibrated_sensor_token"]]
            K, R_cs, t_cs = build_camera_matrices(cal)

            R_ego = quat_to_rotation(ego_pose["rotation"])
            t_ego = np.array(ego_pose["translation"])
            # world -> ego frame
            pts_ego = (R_ego.T @ (pts_world - t_ego).T).T
            # ego -> camera sensor frame  (R_cs is already R_ego2cam after fix)
            pts_cam_sensor = (R_cs @ (pts_ego - t_cs).T).T

            valid   = pts_cam_sensor[:, 2] > 0.5
            px      = np.zeros((H*W, 2))
            temp    = pts_cam_sensor[valid]
            if len(temp) > 0:
                proj    = (K @ temp.T).T
                proj    = proj[:, :2] / proj[:, 2:3]
                px[valid] = proj

            h_img, w_img = img_np.shape[:2]
            in_frame = (px[:,0]>=0)&(px[:,0]<w_img)&(px[:,1]>=0)&(px[:,1]<h_img)
            vis_mask = valid & in_frame

            feat = sample_image_features(img_np, px, vis_mask)
            feat_acc   += feat
            weight_acc += vis_mask.astype(np.float32)

        if weight_acc.max() == 0:
            print("SKIPPED (no images loaded)")
            continue

        denom      = np.maximum(weight_acc, 1.0)
        feat_fused = feat_acc / denom[:, None]
        occ_mean, occ_uncert = mc_occupancy(feat_fused)
        occ_map    = occ_mean.reshape(H, W)
        uncert_map = occ_uncert.reshape(H, W)
        coverage   = (weight_acc > 0).reshape(H, W).astype(np.float32)

        metrics = compute_metrics(occ_map, uncert_map, coverage)
        print(f"cov={metrics['coverage']:.2f} conf={metrics['mean_confidence']:.3f} "
              f"cams={len(loaded_imgs)}")

        cam_b64 = {ch: img_to_b64(img) for ch, img in loaded_imgs.items()}

        results.append({
            "token":       tok[:12],
            "metrics":     metrics,
            "occ_b64":     map_to_png_b64(occ_map,    "turbo"),
            "uncert_b64":  map_to_png_b64(uncert_map, "viridis"),
            "overlay_b64": overlay_uncertainty_on_occ(occ_map, uncert_map),
            "cov_b64":     map_to_png_b64(coverage,   "turbo"),
            "cams":        cam_b64,
        })

    if not results:
        print("\n❌  No samples processed. See [DEBUG] lines above to diagnose path issues.")
        return

    print(f"\nBuilding dashboard with {len(results)} results...")

    buttons_html = "".join(
        f'<button class="sample-btn" onclick="">{r["token"]}</button>'
        for r in results
    )
    samples_json = json.dumps(results)

    html = HTML_TEMPLATE
    html = html.replace("__SAMPLE_BUTTONS__",  buttons_html)
    html = html.replace("__METRICS_HTML__",    build_metrics_html())
    html = html.replace("__VIZ_HTML__",        build_viz_html())
    html = html.replace("__SAMPLES_JSON__",    samples_json)
    html = html.replace("__N_SAMPLES__",       str(len(results)))
    html = html.replace("__GRID_RES__",        f"{GRID_RES:.2f}")
    html = html.replace("__MC_N__",            str(MC_SAMPLES))

    out_html = outdir / "bev_dashboard.html"
    with open(out_html, "w") as f:
        f.write(html)

    print(f"\n✅  Dashboard saved → {out_html}")
    print("    Open in any browser — no server required.\n")

if __name__ == "__main__":
    main()