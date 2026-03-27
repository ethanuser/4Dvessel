#!/usr/bin/env python3
"""
Generate and visualize a 'Binary Mixture Map' between a reference and a condition.
Rigidly aligns the condition to the reference coordinate space, resamples stress,
and creates an RGB mixture (Reference=Red, Condition=Blue).
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import after path setup
import utils.vessel_utils as vu

# ============================================================================
# CONFIGURATION
# ============================================================================
# Path to the reference .npy file (defines Geometry and Red channel stress)
REFERENCE_FILE = r"./data/processed/a14/cluster_mesh_export/vertex_stress_timeseries_a14.npy"

# Path to the condition .npy file (mapped to Reference; defines Blue channel stress)
CONDITION_FILE = r"./data/processed/a1/cluster_mesh_export/vertex_stress_timeseries_a1.npy"

# Threshold distance for point mapping (nearest-neighbor distance limit)
MAX_MAP_DIST = 0.5

# "max_over_time" or "mean_over_time" - Summarization method across frames
SUMMARY_MODE = "mean_over_time"

# If True, shows a per-frame map with a slider
PER_FRAME_VIEW = False

# --- Normalization ---
# Percentage of max stress to saturated (e.g. 98.0)
# Helps with visibility if there are high-stress outliers
V_MAX_PERCENTILE = 98.0
V_MAX_FIXED = None # If set, overrides percentile-based scaling

# Visibility Constants
POINT_SIZE = 15
EPS = 1e-12
BACKGROUND_BRIGHTNESS = 0.25 # Brightness of zero-stress points

# ============================================================================

def best_fit_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch algorithm for rigid transform alignment."""
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am.T @ Bm
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B.reshape(3, 1) - R @ centroid_A.reshape(3, 1)
    return R, t


def icp_registration(source: np.ndarray, target: np.ndarray, max_iterations: int = 50, tolerance: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Iterative Closest Point registration."""
    src = source[np.all(np.isfinite(source), axis=1)].copy()
    dst = target[np.all(np.isfinite(target), axis=1)].copy()
    if len(src) == 0 or len(dst) == 0:
        return np.eye(3), np.zeros((3, 1))
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))
    prev_error = 0
    tree = KDTree(dst)
    for i in range(max_iterations):
        distances, indices = tree.query(src)
        R, t = best_fit_transform(src, dst[indices])
        src = (R @ src.T + t).T
        R_total = R @ R_total
        t_total = R @ t_total + t
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    return R_total, t_total


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid transform."""
    shape = points.shape
    flat_pts = points.reshape(-1, 3)
    transformed = (R @ flat_pts.T + t).T
    return transformed.reshape(shape)


def validate_keys(data: dict, name: str):
    """Ensure data has required keys."""
    required = ["vertices", "vertex_stress", "times", "experiment_name"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"\nCRITICAL ERROR: File for {name} is missing keys: {missing}")
        sys.exit(1)


def visualize_binary_map(ref_data: dict, cond_data: dict):
    """Visualize using PyVista with Binary RGB mixture (Red vs Blue)."""
    plotter = pv.Plotter(title="Vessel Stress Binary Mixture Map")
    plotter.set_background('white')

    ref_verts = ref_data["vertices"]  # (T, N, 3)
    ref_stress = ref_data["vertex_stress"] # (T, N)
    cond_verts = cond_data["vertices"] # (T, N_c, 3)
    cond_stress_raw = cond_data["vertex_stress"] # (T, N_c)

    num_frames = ref_verts.shape[0]
    num_verts = ref_verts.shape[1]

    # 1. Align Condition to Reference using ICP on Frame 0
    print(f"\nAligning {cond_data['experiment_name']} to {ref_data['experiment_name']} (Frame 0)...")
    R, t = icp_registration(cond_verts[0], ref_verts[0])
    cond_verts_aligned = apply_transform(cond_verts, R, t)

    # 2. Resample Condition stress onto Reference vertices using Frame 0 neighborhood
    print(f"Establishing point correspondences...")
    tree_0 = KDTree(cond_verts_aligned[0])
    dists_0, indices_0 = tree_0.query(ref_verts[0])
    valid_mask = dists_0 <= MAX_MAP_DIST
    pct_lost = (np.sum(~valid_mask) / num_verts) * 100
    print(f"  Mapping Coverage: {100-pct_lost:.1f}% of vertices within MAX_MAP_DIST.")

    # 3. Prepare Stress Arrays (2, T, N)
    # [0] = Reference (Red), [1] = Condition (Blue)
    all_stress = np.zeros((2, num_frames, num_verts))
    all_stress[0] = ref_stress
    
    # Map condition stress through indices
    cond_mapped = np.full((num_frames, num_verts), np.nan)
    for t_idx in range(num_frames):
        cond_mapped[t_idx, valid_mask] = cond_stress_raw[t_idx, indices_0[valid_mask]]
    all_stress[1] = cond_mapped
    # Calculate normalization factor (V_MAX)
    if V_MAX_FIXED is not None:
        v_max = V_MAX_FIXED
    else:
        # Use percentile to find a robust maximum (ignoring NaNs)
        valid_stress = all_stress[~np.isnan(all_stress)]
        if len(valid_stress) > 0:
            v_max = np.percentile(valid_stress, V_MAX_PERCENTILE)
        else:
            v_max = 1.0
    
    if v_max <= 0: v_max = 1.0
    print(f"Visualization Scale (V_MAX): {v_max:.2e} (based on {V_MAX_PERCENTILE}th percentile)")

    # Summarize if needed
    if not PER_FRAME_VIEW:
        print(f"Summarizing stress (mode: {SUMMARY_MODE})...")
        with np.errstate(all='ignore'):
            if SUMMARY_MODE == "max_over_time":
                final_stress = np.nanmax(all_stress, axis=1) # (2, N)
            else:
                final_stress = np.nanmean(all_stress, axis=1) # (2, N)
    else:
        final_stress = all_stress # (2, T, N)

    def get_mesh_for_frame(t_idx):
        poly = pv.PolyData(ref_verts[t_idx])
        
        if PER_FRAME_VIEW:
            s_frame = final_stress[:, t_idx, :] # (2, N)
        else:
            s_frame = final_stress # (2, N)

        # Mapping: Ref -> Red, Cond -> Blue
        rgb = np.zeros((num_verts, 3))
        s_norm = np.maximum(s_frame, 0) / v_max
        
        rgb[:, 0] = s_norm[0] # R (Ref)
        rgb[:, 2] = s_norm[1] # B (Cond)
        
        rgb = np.clip(rgb, 0, 1)
        
        # Faint neutral background
        total = np.nansum(s_norm, axis=0)
        rgb[total < EPS] = BACKGROUND_BRIGHTNESS
        
        poly.point_data["RGB"] = rgb
        return poly

    # Interaction
    current_frame_idx = 0
    frame_indices = list(range(num_frames))
    current_actor = [None]
    current_text_actor = [None]

    def update_frame_display(idx):
        nonlocal current_frame_idx
        if idx < 0 or idx >= num_frames: return
        current_frame_idx = idx
        mesh = get_mesh_for_frame(idx)
        current_actor[0] = plotter.add_mesh(mesh, scalars="RGB", rgb=True, render_points_as_spheres=True, point_size=POINT_SIZE, lighting=False, name="binary_mesh")
        if current_text_actor[0]: plotter.remove_actor(current_text_actor[0])
        txt = f"Frame: {idx}/{num_frames-1} | Red: {ref_data['experiment_name']} | Blue: {cond_data['experiment_name']}"
        current_text_actor[0] = plotter.add_text(txt, font_size=11, position='upper_left', color='black')
        plotter.render()

    if not PER_FRAME_VIEW:
        update_frame_display(0)
    else:
        plotter.add_slider_widget(lambda v: update_frame_display(int(v)), (0, num_frames-1), 0, title="Frame", pointa=(0.1, 0.02), pointb=(0.9, 0.02), style='modern')
        plotter.add_key_event('Right', lambda: update_frame_display(min(current_frame_idx + 1, num_frames-1)) or plotter.slider_widgets[0].GetRepresentation().SetValue(current_frame_idx))
        plotter.add_key_event('Left', lambda: update_frame_display(max(current_frame_idx - 1, 0)) or plotter.slider_widgets[0].GetRepresentation().SetValue(current_frame_idx))
        update_frame_display(0)

    legend_text = (
        f"Binary Map (RGB Mixture):\n"
        f"  Red  = {ref_data['experiment_name']} (Reference)\n"
        f"  Blue = {cond_data['experiment_name']} (Condition)\n"
        f"  Purple = Mixture"
    )
    plotter.add_text(legend_text, font_size=10, position='upper_right', color='black')
    plotter.camera_position = 'xy'
    plotter.reset_camera()
    plotter.show()


def main():
    print(f"=== BINARY STRESS MIXTURE MAP ===")
    
    # Load Reference
    ref_path = Path(REFERENCE_FILE)
    if not ref_path.exists():
        print(f"Error: Reference path not found: {ref_path}")
        return
    ref_data = np.load(str(ref_path), allow_pickle=True).item()
    validate_keys(ref_data, "Reference")

    # Load Condition
    cond_path = Path(CONDITION_FILE)
    if not cond_path.exists():
        print(f"Error: Condition path not found: {cond_path}")
        return
    cond_data = np.load(str(cond_path), allow_pickle=True).item()
    validate_keys(cond_data, "Condition")

    visualize_binary_map(ref_data, cond_data)

if __name__ == "__main__":
    main()
