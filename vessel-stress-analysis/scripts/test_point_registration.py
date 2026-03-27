#!/usr/bin/env python3
"""
Test point cloud registration between two datasets.
Rigidly aligns the condition point cloud (orange) to the reference (blue)
using ICP (Iterative Closest Point) on the first frame, and applies that 
transform to the entire sequence.
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
# Default paths
REFERENCE_CLUSTER_NPY_FILE = r"./data/processed/a13/cluster_mesh_export/cluster_mesh_timeseries_a13.npy"
CONDITION_CLUSTER_NPY_FILE = r"./data/processed/a14/cluster_mesh_export/cluster_mesh_timeseries_a14.npy"

# Visualization constants
CLUSTER_POINT_SIZE = 20
REFERENCE_CLUSTER_COLOR = "blue"  # Reference clusters
CONDITION_CLUSTER_COLOR = "orange"  # Condition clusters


def _resolve_path(file_path: str, project_root: Path) -> Path:
    """
    Resolve a relative or absolute path against the project root and CWD.
    """
    if not os.path.isabs(file_path):
        # Try relative to project root first
        resolved = project_root / file_path
        if not resolved.exists():
            # Fallback: relative to current working directory
            resolved = Path(file_path).resolve()
    else:
        resolved = Path(file_path)
    return resolved


def best_fit_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the least-squares best-fit transform that maps 
    corresponding points A to B in 3D (Rigid / Procrustes).
    
    B = R @ A.T + t
    
    Args:
        A: (N, 3) moving points
        B: (N, 3) fixed points (corresponding)
    Returns:
        R: (3, 3) rotation matrix
        t: (3, 1) translation vector
    """
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
    """
    Iterative Closest Point registration.
    Aligns source to target.
    
    Args:
        source: Moving point cloud (N, 3)
        target: Fixed point cloud (M, 3)
        max_iterations: Max iterations
        tolerance: Convergence tolerance on change in MSE
        
    Returns:
        R (3, 3) rotation matrix, t (3, 1) translation vector
    """
    # Filter points
    src = source[np.all(np.isfinite(source), axis=1)].copy()
    dst = target[np.all(np.isfinite(target), axis=1)].copy()
    
    if len(src) == 0 or len(dst) == 0:
        return np.eye(3), np.zeros((3, 1))

    # Initialize transform
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))
    
    prev_error = 0

    # Build KDTree for target
    tree = KDTree(dst)

    for i in range(max_iterations):
        # 1. Find nearest neighbors
        distances, indices = tree.query(src)
        
        # 2. Compute best fit transform for neighbors
        R, t = best_fit_transform(src, dst[indices])

        # 3. Update source and total transform
        src = (R @ src.T + t).T
        
        # Accumulate: New_Point = R_new * (R_total * P + t_total) + t_new
        #                   = (R_new * R_total) * P + (R_new * t_total + t_new)
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Check convergence
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            print(f"  ICP converged at iteration {i} (Error: {mean_error:.6f})")
            break
        prev_error = mean_error
        
        if i == max_iterations - 1:
            print(f"  ICP reached maximum iterations (Error: {mean_error:.6f})")

    return R_total, t_total


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply Rigid Transform (R, t) to points.
    """
    # points: (..., 3)
    shape = points.shape
    flat_pts = points.reshape(-1, 3)
    
    # Filter non-finite for transform apply, but keep array structure
    # Actually, matrix multiplication handles NaNs fine (they stay NaNs)
    transformed = (R @ flat_pts.T + t).T
    return transformed.reshape(shape)


def create_interactive_comparison(
    ref_positions: np.ndarray,
    cond_positions: np.ndarray,
    ref_times: np.ndarray,
    cond_times: np.ndarray,
    experiment_names: Tuple[str, str],
) -> None:
    """
    Create an interactive 3D visualization comparing clusters.
    """
    num_frames = ref_positions.shape[0]
    num_frames_cond = cond_positions.shape[0]
    frame_indices = list(range(num_frames))

    print(f"\nCreating interactive visualization for {num_frames} frames...")

    # Precompute global bounds
    all_positions = []
    for frame_idx in frame_indices:
        all_positions.append(ref_positions[frame_idx])
        if frame_idx < num_frames_cond:
            all_positions.append(cond_positions[frame_idx])

    if len(all_positions) > 0:
        all_positions_flat = np.vstack(all_positions)
        bounds = [
            float(np.nanmin(all_positions_flat[:, 0])),
            float(np.nanmax(all_positions_flat[:, 0])),
            float(np.nanmin(all_positions_flat[:, 1])),
            float(np.nanmax(all_positions_flat[:, 1])),
            float(np.nanmin(all_positions_flat[:, 2])),
            float(np.nanmax(all_positions_flat[:, 2])),
        ]
    else:
        bounds = [-1, 1, -1, 1, -1, 1]

    # Pre-create PolyData objects
    frame_mesh_data = {}
    for frame_idx in frame_indices:
        ref_poly = pv.PolyData(ref_positions[frame_idx])
        cond_poly = pv.PolyData(cond_positions[frame_idx]) if frame_idx < num_frames_cond else None
        frame_mesh_data[frame_idx] = {"ref": ref_poly, "cond": cond_poly}

    plotter = pv.Plotter(title="Point Cloud ICP Registration")
    plotter.set_background('white')

    current_ref_actor = None
    current_cond_actor = None
    current_text_actor = None
    current_frame_idx = 0

    def update_frame_display(frame_list_idx):
        nonlocal current_ref_actor, current_cond_actor, current_text_actor, current_frame_idx
        
        if frame_list_idx < 0 or frame_list_idx >= len(frame_indices):
            return
            
        current_frame_idx = frame_list_idx
        frame_idx = frame_indices[frame_list_idx]
        mesh_data = frame_mesh_data[frame_idx]

        old_ref = current_ref_actor
        old_cond = current_cond_actor

        if mesh_data["ref"] is not None:
            current_ref_actor = plotter.add_mesh(
                mesh_data["ref"], render_points_as_spheres=True,
                point_size=CLUSTER_POINT_SIZE, color=REFERENCE_CLUSTER_COLOR,
                name=f"ref_{frame_idx}"
            )
        if mesh_data["cond"] is not None:
            current_cond_actor = plotter.add_mesh(
                mesh_data["cond"], render_points_as_spheres=True,
                point_size=CLUSTER_POINT_SIZE, color=CONDITION_CLUSTER_COLOR,
                name=f"cond_{frame_idx}"
            )

        if current_text_actor:
            plotter.remove_actor(current_text_actor)
        
        time_val = float(ref_times[frame_idx]) if ref_times is not None and len(ref_times) > frame_idx else float(frame_idx)
        text = f"Frame: {frame_idx}/{frame_indices[-1]} | Time: {time_val:.4f} | Blue: {experiment_names[0]} | Orange: {experiment_names[1]} (ICP ALIGNED)"
        current_text_actor = plotter.add_text(text, font_size=11, position='upper_left', color='black')

        plotter.render()
        if old_ref:
            try: plotter.remove_actor(old_ref)
            except: pass
        if old_cond:
            try: plotter.remove_actor(old_cond)
            except: pass

    plotter.add_slider_widget(lambda v: update_frame_display(int(v)), (0, len(frame_indices)-1), 0, title="Frame", pointa=(0.1, 0.02), pointb=(0.9, 0.02), style='modern')
    plotter.add_key_event('Right', lambda: update_frame_display(min(current_frame_idx + 1, len(frame_indices)-1)))
    plotter.add_key_event('Left', lambda: update_frame_display(max(current_frame_idx - 1, 0)))
    plotter.add_axes()
    
    update_frame_display(0)
    plotter.camera_position = 'xy'
    plotter.reset_camera(bounds=bounds)
    plotter.show()


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    ref_path_str = sys.argv[1] if len(sys.argv) >= 2 else REFERENCE_CLUSTER_NPY_FILE
    cond_path_str = sys.argv[2] if len(sys.argv) >= 3 else CONDITION_CLUSTER_NPY_FILE

    resolved_ref_path = _resolve_path(ref_path_str, project_root)
    resolved_cond_path = _resolve_path(cond_path_str, project_root)

    print(f"\n=== POINT CLOUD ICP REGISTRATION TEST ===")
    
    # Load data
    ref_data = np.load(str(resolved_ref_path), allow_pickle=True).item()
    cond_data = np.load(str(resolved_cond_path), allow_pickle=True).item()
    
    ref_positions = ref_data.get("cluster_positions")
    cond_positions = cond_data.get("cluster_positions")
    ref_times = ref_data.get("times")
    cond_times = cond_data.get("times")
    ref_exp = ref_data.get("experiment_name", "Ref")
    cond_exp = cond_data.get("experiment_name", "Cond")

    print(f"Loaded Reference: {ref_exp} ({ref_positions.shape[0]} frames, {ref_positions.shape[1]} points)")
    print(f"Loaded Condition: {cond_exp} ({cond_positions.shape[0]} frames, {cond_positions.shape[1]} points)")

    # 1. Estimate transform using ICP on Frame 0
    print(f"\nRegistering {cond_exp} to {ref_exp} using ICP (Frame 0)...")
    
    # Initial error (Nearest Neighbor based)
    tree_ref = KDTree(ref_positions[0])
    dists_init, _ = tree_ref.query(cond_positions[0])
    err_initial = np.mean(dists_init)
    
    R, t = icp_registration(cond_positions[0], ref_positions[0])
    
    print("\nRecovered Rigid Transform (ICP):")
    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (t):\n{t.flatten()}")
    
    # Final error
    aligned_0 = apply_transform(cond_positions[0], R, t)
    dists_final, _ = tree_ref.query(aligned_0)
    err_final = np.mean(dists_final)
    
    print(f"Mean Nearest Neighbor Error (Initial): {err_initial:.6f}")
    print(f"Mean Nearest Neighbor Error (Aligned): {err_final:.6f}")

    # 2. Apply transform to ALL frames
    print(f"\nApplying alignment to all {cond_positions.shape[0]} frames...")
    cond_positions_aligned = np.zeros_like(cond_positions)
    for i in range(cond_positions.shape[0]):
        cond_positions_aligned[i] = apply_transform(cond_positions[i], R, t)

    # 3. Visualize
    create_interactive_comparison(
        ref_positions=ref_positions,
        cond_positions=cond_positions_aligned,
        ref_times=ref_times,
        cond_times=cond_times,
        experiment_names=(ref_exp, cond_exp)
    )

if __name__ == "__main__":
    main()
