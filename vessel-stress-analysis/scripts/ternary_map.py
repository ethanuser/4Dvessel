#!/usr/bin/env python3
"""
Generate and visualize a 'Ternary Mixture Map' across three experimental conditions.
Rigidly aligns conditions to a reference coordinate space, resamples stress data,
and creates an RGB mixture where each channel represents the relative stress contribution.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

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
# The reference experiment defines the common coordinate space (geometry)
REFERENCE_EXPERIMENT_FILE = r"./data/processed/a13/cluster_mesh_export/vertex_stress_timeseries_a13.npy"

# Dictionary mapping condition names to their exported vertex stress files
# These should be the files produced by export_vertex_stress_timeseries.py
CONDITION_FILES = {
    "cervical": r"./data/processed/a1/cluster_mesh_export/vertex_stress_timeseries_a1.npy",
    "cavernous": r"./data/processed/a7/cluster_mesh_export/vertex_stress_timeseries_a7.npy",
    "terminal": r"./data/processed/a14/cluster_mesh_export/vertex_stress_timeseries_a14.npy",
}

# Threshold distance for point mapping (nearest-neighbor distance limit)
MAX_MAP_DIST = 0.8

# "max_over_time" or "mean_over_time" - Summarization method across frames
SUMMARY_MODE = "max_over_time"

# If True, shows a per-frame winner map with a slider
# If False, shows a single summary map based on SUMMARY_MODE
PER_FRAME_VIEW = True

# --- Ternary Constants ---
EPS = 1e-12
INTENSITY_BASE = 0.3
INTENSITY_SCALE = 0.7
INTENSITY_GAMMA = 1.0

# Visualization Constants
POINT_SIZE = 15

# ============================================================================

def best_fit_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch algorithm for rigid transform alignment."""
    # 1. Centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 2. Center the points
    Am = A - centroid_A
    Bm = B - centroid_B

    # 3. Covariance matrix
    H = Am.T @ Bm

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)

    # 5. Rotation
    R = Vt.T @ U.T

    # 6. Reflection check
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 7. Translation
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
    """Ensure data has required keys, or fail loudly."""
    required = ["vertices", "vertex_stress", "times", "experiment_name"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"\nCRITICAL ERROR: File for {name} is missing keys: {missing}")
        print(f"Available keys: {list(data.keys())}")
        sys.exit(1)


def process_condition(name: str, path: Path, ref_pts_0: np.ndarray) -> Dict[str, Any]:
    """Load, align, and prepare a condition's data."""
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    data = np.load(str(path), allow_pickle=True).item()
    validate_keys(data, name)
    
    print(f"\nProcessing Condition: {name} ({data['experiment_name']})")
    
    cond_verts = data["vertices"]    # (T, N, 3)
    cond_stress = data["vertex_stress"] # (T, N)
    
    # 1. Align using frame 0
    print(f"  Aligning to reference using ICP (Frame 0)...")
    tree_ref = KDTree(ref_pts_0)
    d_init, _ = tree_ref.query(cond_verts[0])
    err_init = np.mean(d_init)
    
    R, t = icp_registration(cond_verts[0], ref_pts_0)
    
    aligned_0 = apply_transform(cond_verts[0], R, t)
    d_final, _ = tree_ref.query(aligned_0)
    err_final = np.mean(d_final)
    
    print(f"  ICP Result: Mean NN Error {err_init:.6f} -> {err_final:.6f}")
    
    # 2. Apply alignment to all frames
    print(f"  Applying transform to all {cond_verts.shape[0]} frames...")
    cond_verts_aligned = apply_transform(cond_verts, R, t)
    
    return {
        "vertices": cond_verts_aligned,
        "stress": cond_stress,
        "name": name,
        "err_final": err_final
    }


def compute_ternary_data(ref_verts: np.ndarray, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Resample condition stresses and compute ternary R/G/B weights and dominance.
    """
    is_timeseries = len(ref_verts.shape) == 3
    num_frames = ref_verts.shape[0] if is_timeseries else 1
    num_verts = ref_verts.shape[1] if is_timeseries else ref_verts.shape[0]
    num_cond = len(conditions)
    
    # Storage for resampled stress on reference: (num_cond, T, N)
    all_stress_resampled = np.full((num_cond, num_frames, num_verts), np.nan)
    nan_counts = [0] * num_cond

    print(f"\nEstablishing point correspondences at Frame 0...")
    for c_idx, cond in enumerate(conditions):
        c_verts = cond["vertices"] # (T, N_c, 3)
        c_stress = cond["stress"]   # (T, N_c)
        
        # 1. Find nearest neighbors at Frame 0 only
        ref_0 = ref_verts[0] if is_timeseries else ref_verts
        cond_0 = c_verts[0] if len(c_verts.shape) == 3 else c_verts
        
        tree = KDTree(cond_0)
        dists_0, indices_0 = tree.query(ref_0)
        
        # Determine valid mask based on Frame 0 distance
        valid_mask = dists_0 <= MAX_MAP_DIST
        nan_counts[c_idx] = np.sum(~valid_mask)
        
        # 2. Track those indices across all frames
        # This assumes point indices in c_stress are temporally consistent
        for t in range(num_frames):
            all_stress_resampled[c_idx, t, valid_mask] = c_stress[t, indices_0[valid_mask]]

    for c_idx, cond in enumerate(conditions):
        pct = (nan_counts[c_idx] / (num_frames * num_verts)) * 100
        print(f"  {cond['name']}: {pct:.1f}% vertices beyond MAX_MAP_DIST threshold.")

    # Summarize over time
    if not PER_FRAME_VIEW:
        print(f"\nSummarizing stress across time (mode: {SUMMARY_MODE})...")
        with np.errstate(all='ignore'):
            if SUMMARY_MODE == "max_over_time":
                S_ref = np.nanmax(all_stress_resampled, axis=1) # (num_cond, N)
            else:
                S_ref = np.nanmean(all_stress_resampled, axis=1) # (num_cond, N)
        
        # S_ref: (3, N)
        S_clamped = np.maximum(S_ref, 0)
        total_stress = np.nansum(S_clamped, axis=0) # (N,)
        
        # Weights: (3, N)
        weights = S_clamped / (total_stress + EPS)
        
        # Intensity: normalized distance from uniform (1/3, 1/3, 1/3)
        # d = sqrt((w0-1/3)**2 + (w1-1/3)**2 + (w2-1/3)**2)
        diff = weights - (1.0/3.0)
        d = np.sqrt(np.sum(diff**2, axis=0))
        d_max = np.sqrt((1.0 - 1.0/3.0)**2 + 2.0 * (0.0 - 1.0/3.0)**2)
        intensity = d / d_max
        
        # Handle all NaN vertices
        all_nan = np.all(np.isnan(S_ref), axis=0)
        weights[:, all_nan] = 0.333
        intensity[all_nan] = 0
        total_stress[all_nan] = 0
        
        # Print stats
        for i, cond in enumerate(conditions):
            valid = S_ref[i][~np.isnan(S_ref[i])]
            smin = np.min(valid) if len(valid)>0 else 0
            smax = np.max(valid) if len(valid)>0 else 0
            print(f"  {cond['name']} Stress: min={smin:.2e}, max={smax:.2e}")
        
        print(f"  Total Stress: min={np.min(total_stress):.2e}, max={np.max(total_stress):.2e}")
        print(f"  Intensity: min={np.min(intensity):.4f}, max={np.max(intensity):.4f}")

        return {
            "weights": weights,  # (3, N)
            "intensity": intensity, # (N,)
            "total_stress": total_stress # (N,)
        }
    else:
        # Per-frame view
        print(f"\nProcessing ternary data per frame...")
        S_clamped = np.maximum(all_stress_resampled, 0) # (3, T, N)
        total_stress = np.nansum(S_clamped, axis=0) # (T, N)
        
        weights = S_clamped / (total_stress[np.newaxis, :, :] + EPS) # (3, T, N)
        
        diff = weights - (1.0/3.0)
        d = np.sqrt(np.sum(diff**2, axis=0))
        d_max = np.sqrt((1.0 - 1.0/3.0)**2 + 2.0 * (0.0 - 1.0/3.0)**2)
        intensity = d / d_max
        
        all_nan_T = np.all(np.isnan(all_stress_resampled), axis=0)
        # Fix NaNs for visualization
        intensity[all_nan_T] = 0
        total_stress[all_nan_T] = 0
        
        return {
            "weights": weights,   # (3, T, N)
            "intensity": intensity, # (T, N)
            "total_stress": total_stress # (T, N)
        }


def visualize_ternary_map(ref_verts: np.ndarray, ternary_data: dict, cond_names: List[str]):
    """Visualize using PyVista with Ternary RGB mixture."""
    plotter = pv.Plotter(title="Vessel Stress Ternary Mixture Map")
    plotter.set_background('white')

    is_timeseries = len(ref_verts.shape) == 3
    
    def get_mesh_for_frame(t_idx):
        pts = ref_verts[t_idx] if is_timeseries else ref_verts
        poly = pv.PolyData(pts)
        
        if PER_FRAME_VIEW:
            # weights is (3, T, N)
            w = ternary_data["weights"][:, t_idx, :].T # (N, 3)
            intensity = ternary_data["intensity"][t_idx]     # (N,)
            total = ternary_data["total_stress"][t_idx] # (N,)
        else:
            # weights is (3, N)
            w = ternary_data["weights"].T # (N, 3)
            intensity = ternary_data["intensity"] # (N,)
            total = ternary_data["total_stress"] # (N,)

        # RGB directly from weights
        rgb = w.copy()
        
        # Modulate brightness by intensity
        # I_vis = BASE + SCALE * intensity^GAMMA
        modulation = INTENSITY_BASE + INTENSITY_SCALE * (intensity ** INTENSITY_GAMMA)
        
        # Handle neutral balanced mixture for non-mapped / low-stress points
        # Instead of solid gray, we use the uniform mixture at base intensity
        low_stress_mask = total < EPS
        rgb[low_stress_mask] = 0.333333
        modulation[low_stress_mask] = INTENSITY_BASE
        
        modulated_rgb = rgb * modulation[:, np.newaxis]
        modulated_rgb = np.clip(modulated_rgb, 0, 1)

        poly.point_data["RGB"] = modulated_rgb
        poly.point_data["Intensity"] = intensity
        poly.point_data["TotalStress"] = total
        
        return poly

    # Support for interactive sliding
    current_frame_idx = 0
    num_frames = ref_verts.shape[0] if is_timeseries else 1
    frame_indices = list(range(num_frames))
    current_actor = [None]
    current_text_actor = [None]

    def update_frame_display(frame_list_idx):
        """Update the display to show a specific frame"""
        nonlocal current_frame_idx
        
        if frame_list_idx < 0 or frame_list_idx >= len(frame_indices):
            return
            
        current_frame_idx = frame_list_idx
        frame_idx = frame_indices[frame_list_idx]
        mesh = get_mesh_for_frame(frame_idx)

        # Store old actors for removal
        old_actor = current_actor[0]

        # Add new mesh
        current_actor[0] = plotter.add_mesh(
            mesh, scalars="RGB", rgb=True,
            render_points_as_spheres=True, point_size=POINT_SIZE,
            lighting=False, name="ternary_mesh"
        )

        # Update text if needed (currently using static legend, but could add frame info)
        if current_text_actor[0] is not None:
             plotter.remove_actor(current_text_actor[0])
        
        # Add frame info text
        frame_text = f"Frame: {frame_idx}/{frame_indices[-1]}"
        current_text_actor[0] = plotter.add_text(frame_text, font_size=11, position='upper_left', color='black')

        # Render
        plotter.render()

        # No need to manually remove if name is consistent in add_mesh, 
        # but follow the pattern from reference script for robustness
        if old_actor is not None and old_actor != current_actor[0]:
            try: plotter.remove_actor(old_actor)
            except: pass

    def slider_callback(value):
        update_frame_display(int(value))

    def on_next_frame():
        nonlocal current_frame_idx
        if current_frame_idx < len(frame_indices) - 1:
            new_idx = current_frame_idx + 1
            update_frame_display(new_idx)
            if len(plotter.slider_widgets) > 0:
                plotter.slider_widgets[0].GetRepresentation().SetValue(new_idx)

    def on_prev_frame():
        nonlocal current_frame_idx
        if current_frame_idx > 0:
            new_idx = current_frame_idx - 1
            update_frame_display(new_idx)
            if len(plotter.slider_widgets) > 0:
                plotter.slider_widgets[0].GetRepresentation().SetValue(new_idx)

    if not PER_FRAME_VIEW:
        mesh = get_mesh_for_frame(0)
        plotter.add_mesh(
            mesh, scalars="RGB", rgb=True, 
            render_points_as_spheres=True, point_size=POINT_SIZE,
            lighting=False
        )
    else:
        # Add slider widget
        plotter.add_slider_widget(
            callback=slider_callback,
            rng=(0, len(frame_indices) - 1),
            value=0,
            title="Frame",
            pointa=(0.1, 0.02),
            pointb=(0.9, 0.02),
            style='modern',
        )

        # Register keys
        plotter.add_key_event('Right', on_next_frame)
        plotter.add_key_event('Left', on_prev_frame)
        plotter.add_key_event('d', on_next_frame)
        plotter.add_key_event('a', on_prev_frame)

        # Initialize
        update_frame_display(0)

    # Legend text
    legend_text = (
        f"Mixture Map (RGB):\n"
        f"  Red   = {cond_names[0]}\n"
        f"  Green = {cond_names[1]}\n"
        f"  Blue  = {cond_names[2]}\n"
        f"\n"
        f"Secondary Colors:\n"
        f"  Yellow  = R+G mixture\n"
        f"  Cyan    = G+B mixture\n"
        f"  Magenta = R+B mixture\n"
        f"\n"
        f"Intensity modulated by 'Distance from Uniform':\n"
        f"  distance to (1/3, 1/3, 1/3)\n"
        f"  Balanced points are darkened ({INTENSITY_BASE:.1f} base)\n"
        f"  Strong winners are brightened"
    )
    plotter.add_text(legend_text, font_size=10, position='upper_right', color='black')
    
    plotter.camera_position = 'xy'
    plotter.reset_camera()
    plotter.show()


def main():
    print(f"=== TERNARY MIXTURE MAP GENERATION ===")
    
    # 1. Load Reference
    ref_path = Path(REFERENCE_EXPERIMENT_FILE)
    if not ref_path.exists():
        print(f"Error: Reference path does not exist: {ref_path}")
        return
    
    ref_data = np.load(str(ref_path), allow_pickle=True).item()
    validate_keys(ref_data, "Reference")
    ref_verts = ref_data["vertices"] 
    ref_pts_0 = ref_verts[0]
    
    # 2. Process Conditions
    conditions = []
    cond_names_ordered = list(CONDITION_FILES.keys())
    for name in cond_names_ordered:
        path = Path(CONDITION_FILES[name])
        cond_info = process_condition(name, path, ref_pts_0)
        conditions.append(cond_info)
    
    # 3. Compute Ternary Data
    ternary_data = compute_ternary_data(ref_verts, conditions)
    
    # 4. Visualization
    visualize_ternary_map(ref_verts, ternary_data, cond_names_ordered)

if __name__ == "__main__":
    main()
