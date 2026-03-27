#!/usr/bin/env python3
"""
Compare two point cloud cluster datasets by calculating chamfer distance
between cluster data from two sources.

This script:
1. Loads cluster positions from two cluster_mesh_export numpy files
2. Calculates chamfer distance between the two cluster sets
3. Creates an interactive visualization showing:
   - Blue points: Reference clusters
   - Orange points: Condition clusters

Usage:
    python compare_point_clouds_interactive.py [reference_npy_path] [condition_npy_path]

Arguments:
    reference_npy_path: Path to the reference .npy file (e.g. baseline)
    condition_npy_path: Path to the condition .npy file (e.g. comparison)
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple
from scipy.spatial.distance import cdist

import numpy as np
import pyvista as pv

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
# If SKIP_DISTANCE_ANALYSIS is True, skips computing chamfer distances (faster loading)
SKIP_DISTANCE_ANALYSIS = True  # Set to True to skip distance calculations

# Default paths
REFERENCE_CLUSTER_NPY_FILE = r"./data/processed/a1/cluster_mesh_export/cluster_mesh_timeseries_a1.npy"
CONDITION_CLUSTER_NPY_FILE = r"./data/processed/a2/cluster_mesh_export/cluster_mesh_timeseries_a2.npy"

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


def compute_chamfer_distance(
    points_a: np.ndarray, points_b: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute chamfer distance between two point clouds.

    Args:
        points_a: First point cloud (N, 3)
        points_b: Second point cloud (M, 3)

    Returns:
        Tuple of (chamfer_distance, a_to_b_mean, b_to_a_mean)
        - chamfer_distance: Symmetric chamfer distance
        - a_to_b_mean: Mean distance from each point in A to nearest in B
        - b_to_a_mean: Mean distance from each point in B to nearest in A
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf"), float("inf"), float("inf")

    # Compute pairwise distances
    distances = cdist(points_a, points_b)

    # For each point in A, find nearest point in B
    a_to_b_distances = np.min(distances, axis=1)
    a_to_b_mean = float(np.mean(a_to_b_distances))

    # For each point in B, find nearest point in A
    b_to_a_distances = np.min(distances, axis=0)
    b_to_a_mean = float(np.mean(b_to_a_distances))

    # Symmetric chamfer distance
    chamfer_distance = (a_to_b_mean + b_to_a_mean) / 2.0

    return chamfer_distance, a_to_b_mean, b_to_a_mean


def create_interactive_comparison(
    ref_positions: np.ndarray,
    cond_positions: np.ndarray,
    ref_times: np.ndarray,
    cond_times: np.ndarray,
    chamfer_distances: Dict[int, Tuple[float, float, float]],
    skip_distance_analysis: bool = False,
) -> None:
    """
    Create an interactive 3D visualization comparing clusters from both sources.

    Args:
        ref_positions: Array of shape (T, K, 3) for reference
        cond_positions: Array of shape (T, K, 3) for condition
        ref_times: Array of times for reference
        cond_times: Array of times for condition
        chamfer_distances: Dict mapping frame indices to chamfer distance tuples
    """
    num_frames_ref = ref_positions.shape[0]
    num_frames_cond = cond_positions.shape[0]
    
    # We use reference as the primary sequence
    num_frames = num_frames_ref
    frame_indices = list(range(num_frames))

    print(f"\nCreating interactive visualization for {num_frames} frames...")

    # Pre-create PolyData objects for each frame
    frame_mesh_data = {}

    # Precompute global bounds across all frames
    all_positions = []
    for frame_idx in frame_indices:
        all_positions.append(ref_positions[frame_idx])
        if frame_idx < num_frames_cond:
            all_positions.append(cond_positions[frame_idx])

    if len(all_positions) > 0:
        all_positions_flat = np.vstack(all_positions)
        bounds = [
            float(all_positions_flat[:, 0].min()),
            float(all_positions_flat[:, 0].max()),
            float(all_positions_flat[:, 1].min()),
            float(all_positions_flat[:, 1].max()),
            float(all_positions_flat[:, 2].min()),
            float(all_positions_flat[:, 2].max()),
        ]
    else:
        bounds = [-1, 1, -1, 1, -1, 1]

    for frame_idx in frame_indices:
        # Reference positions (blue)
        ref_poly = pv.PolyData(ref_positions[frame_idx])

        # Condition positions (orange)
        if frame_idx < num_frames_cond:
            cond_poly = pv.PolyData(cond_positions[frame_idx])
        else:
            cond_poly = None

        frame_mesh_data[frame_idx] = {
            "ref_clusters": ref_poly,
            "cond_clusters": cond_poly,
        }

    # Create PyVista plotter
    plotter = pv.Plotter(title="Point Cloud Comparison")
    plotter.set_background('white')

    # Store the current actors
    current_ref_actor = None
    current_cond_actor = None
    current_text_actor = None
    current_frame_idx = 0

    def update_frame_display(frame_list_idx):
        """Update the display to show a specific frame"""
        nonlocal current_ref_actor, current_cond_actor, current_text_actor, current_frame_idx
        
        if frame_list_idx < 0 or frame_list_idx >= len(frame_indices):
            return
            
        current_frame_idx = frame_list_idx
        frame_idx = frame_indices[frame_list_idx]
        mesh_data = frame_mesh_data[frame_idx]

        # Store old actors for removal
        old_ref_actor = current_ref_actor
        old_cond_actor = current_cond_actor

        # Add reference clusters (blue)
        if mesh_data["ref_clusters"] is not None:
            current_ref_actor = plotter.add_mesh(
                mesh_data["ref_clusters"],
                render_points_as_spheres=True,
                point_size=CLUSTER_POINT_SIZE,
                color=REFERENCE_CLUSTER_COLOR,
                name=f"ref_clusters_{frame_idx}",
            )
        else:
            current_ref_actor = None

        # Add condition clusters (orange)
        if mesh_data["cond_clusters"] is not None:
            current_cond_actor = plotter.add_mesh(
                mesh_data["cond_clusters"],
                render_points_as_spheres=True,
                point_size=CLUSTER_POINT_SIZE,
                color=CONDITION_CLUSTER_COLOR,
                name=f"cond_clusters_{frame_idx}",
            )
        else:
            current_cond_actor = None

        # Update text
        if current_text_actor is not None:
            plotter.remove_actor(current_text_actor)
            
        time_val = float(ref_times[frame_idx]) if ref_times is not None and len(ref_times) > frame_idx else float(frame_idx)
        ref_count = len(ref_positions[frame_idx])
        cond_count = len(cond_positions[frame_idx]) if frame_idx < num_frames_cond else 0
        
        text = f"Frame: {frame_idx}/{frame_indices[-1]} | Time: {time_val:.4f} | Blue (Ref): {ref_count} | Orange (Cond): {cond_count}"
        
        if not skip_distance_analysis:
            if frame_idx in chamfer_distances:
                chamfer_dist, _, _ = chamfer_distances[frame_idx]
                text += f" | Chamfer: {chamfer_dist:.4f}"
                
        current_text_actor = plotter.add_text(text, font_size=11, position='upper_left', color='black')

        # Render
        plotter.render()

        # Remove old actors
        if old_ref_actor is not None:
            try: plotter.remove_actor(old_ref_actor)
            except: pass
        if old_cond_actor is not None:
            try: plotter.remove_actor(old_cond_actor)
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

    # Add axes
    plotter.add_axes()
    
    # Set instructions
    instructions = (
        "Controls:\n"
        "  Slider: Navigate frames\n"
        "  Arrows/AD: Step frames\n"
        "  Mouse: View controls\n"
        "\n"
        "Visuals:\n"
        "  Blue: Reference clusters\n"
        "  Orange: Condition clusters"
    )
    plotter.add_text(instructions, font_size=8, position='lower_left', color='black')

    # Initialize
    update_frame_display(0)
    
    # Set camera
    plotter.camera_position = 'xy'
    plotter.reset_camera(bounds=bounds)
    
    print("Use the slider or arrow keys (Left/Right) to navigate frames\n")
    plotter.show()


def main():
    """
    Main entry point for comparing two cluster numpy files.
    """
    # Parse command-line arguments
    ref_path_str = sys.argv[1] if len(sys.argv) >= 2 else REFERENCE_CLUSTER_NPY_FILE
    cond_path_str = sys.argv[2] if len(sys.argv) >= 3 else CONDITION_CLUSTER_NPY_FILE

    resolved_ref_path = _resolve_path(ref_path_str, project_root)
    resolved_cond_path = _resolve_path(cond_path_str, project_root)

    # Validate files exist
    if not resolved_ref_path.exists():
        print(f"⚠️  ERROR: Reference file not found: {resolved_ref_path}")
        return
    if not resolved_cond_path.exists():
        print(f"⚠️  ERROR: Condition file not found: {resolved_cond_path}")
        return

    try:
        print(f"\n=== POINT CLOUD CLUSTER COMPARISON ===")
        print(f"Loading Reference: {resolved_ref_path}")
        print(f"Loading Condition: {resolved_cond_path}")

        # Load reference data
        ref_data = np.load(str(resolved_ref_path), allow_pickle=True).item()
        ref_positions = ref_data.get("cluster_positions")
        ref_times = ref_data.get("times")
        ref_exp = ref_data.get("experiment_name", "Ref")

        if ref_positions is None:
            print("ERROR: Reference file missing 'cluster_positions'")
            return

        # Load condition data
        cond_data = np.load(str(resolved_cond_path), allow_pickle=True).item()
        cond_positions = cond_data.get("cluster_positions")
        cond_times = cond_data.get("times")
        cond_exp = cond_data.get("experiment_name", "Cond")

        if cond_positions is None:
            print("ERROR: Condition file missing 'cluster_positions'")
            return

        print(f"Reference: {ref_positions.shape[0]} frames, {ref_positions.shape[1]} clusters")
        print(f"Condition: {cond_positions.shape[0]} frames, {cond_positions.shape[1]} clusters")

        # Compute chamfer distances for each frame (if not skipped)
        chamfer_distances = {}
        if not SKIP_DISTANCE_ANALYSIS:
            print("\nComputing chamfer distances for each frame...")
            num_frames = min(ref_positions.shape[0], cond_positions.shape[0])
            for i in range(num_frames):
                dist, a_to_b, b_to_a = compute_chamfer_distance(ref_positions[i], cond_positions[i])
                chamfer_distances[i] = (dist, a_to_b, b_to_a)
                
                if (i + 1) % 50 == 0 or i == num_frames - 1:
                    print(f"  Frame {i}/{num_frames-1}: {dist:.4f}")

            if len(chamfer_distances) > 0:
                all_dist = [d[0] for d in chamfer_distances.values()]
                print(f"\nChamfer Stats: Mean={np.mean(all_dist):.4f}, Max={np.max(all_dist):.4f}")
        else:
            print("\nSkipping distance analysis (SKIP_DISTANCE_ANALYSIS = True)")

        # Create interactive visualization
        create_interactive_comparison(
            ref_positions=ref_positions,
            cond_positions=cond_positions,
            ref_times=ref_times,
            cond_times=cond_times,
            chamfer_distances=chamfer_distances,
            skip_distance_analysis=SKIP_DISTANCE_ANALYSIS,
        )

        print(f"\n✓ Completed comparison: {ref_exp} vs {cond_exp}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n⚠️  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
