#!/usr/bin/env python3
"""
Compare ground truth and ours meshes by calculating stress-based edge matching distance.

This script:
1. Loads cluster positions and meshes from GT and ours numpy files
2. Computes stress for each edge in both meshes
3. Matches each edge in one mesh to the nearest edge in the other mesh
4. Computes the stress difference between matched edges
5. Calculates a symmetric distance metric (like chamfer distance) by averaging:
   - Mean stress difference from GT edges to ours edges
   - Mean stress difference from ours edges to GT edges
6. Creates an interactive visualization showing both meshes with stress-colored edges

The metric is called "Edge Stress Difference Distance" (ESDD) - it measures how well
the stress distributions match between two meshes by comparing corresponding edges.

Usage:
    python compare_GT_ours_edges_interactive.py [gt_npy_path] [ours_npy_path]

Arguments:
    gt_npy_path: Path to the GT .npy file with cluster positions and edges
    ours_npy_path: Path to the ours .npy file with cluster positions and edges
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple
from numpy._core.numeric import False_
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

import numpy as np
import pyvista as pv

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import vessel_utils as vu
from utils.interactive_utils import (
    create_interactive_visualization as setup_interactive_visualization,
    create_default_text_generator,
    create_default_instructions,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Default paths
GT_CLUSTER_NPY_FILE = r"cluster_data/cluster_mesh_timeseries_GT_bend_twist.npy"
OURS_CLUSTER_NPY_FILE = r"cluster_data/cluster_mesh_timeseries_bend_twist.npy"

# GT_CLUSTER_NPY_FILE = r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_25.npy"
# OURS_CLUSTER_NPY_FILE = r"cluster_data/cluster_mesh_timeseries_lattice_strength_0_25.npy"

# Computation flags
COMPUTE_ESDD = False  # Set to False to skip ESDD computation (faster, visualization only)

# Physical constants
YOUNG_MODULUS_SILICON = vu.YOUNG_MODULUS_SILICON

# Visualization constants
EDGE_LINE_WIDTH = 5
EDGE_OPACITY = 0.7
MAX_STRESS_PA = vu.MAX_STRESS_PA  # Convert to Pa
MAX_STRESS_MPA = MAX_STRESS_PA / 1e6

# Removed GT_COLORS and OURS_COLORS, now using viridis via vu.get_colors


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


def load_clustered_numpy_and_convert(clustered_npy_path: Path):
    """
    Load clustered numpy file and extract mesh data.
    
    Args:
        clustered_npy_path: Path to the exported cluster mesh numpy file
        
    Returns:
        Tuple of (cluster_positions, edges, initial_lengths, times, experiment_name)
    """
    if not clustered_npy_path.exists():
        raise FileNotFoundError(f"Clustered numpy file not found: {clustered_npy_path}")
    
    print(f"Loading clustered numpy file: {clustered_npy_path}")
    data = np.load(str(clustered_npy_path), allow_pickle=True).item()
    
    cluster_positions = data.get("cluster_positions")  # (T, K, 3)
    times = data.get("times")  # (T,)
    initial_cluster_means = data.get("initial_cluster_means")  # (K, 3)
    edges = data.get("edges")  # (E, 2)
    initial_lengths = data.get("initial_lengths")  # (E,)
    experiment_name = data.get("experiment_name", "UnknownExperiment")
    
    if cluster_positions is None:
        raise ValueError("Clustered numpy file missing 'cluster_positions' key")
    
    if initial_cluster_means is None:
        initial_cluster_means = cluster_positions[0]
    
    if edges is None:
        edges = np.array([]).reshape(0, 2)
    
    if initial_lengths is None:
        if len(edges) > 0:
            initial_lengths = np.linalg.norm(
                initial_cluster_means[edges[:, 0]] - initial_cluster_means[edges[:, 1]], axis=1
            )
        else:
            initial_lengths = np.array([])
    
    num_frames, num_clusters, _ = cluster_positions.shape
    print(f"  Loaded {num_frames} frames, {num_clusters} clusters, {len(edges)} edges")
    print(f"  Experiment: {experiment_name}")
    
    return cluster_positions, edges, initial_lengths, times, experiment_name

def compute_edge_centers(cluster_positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Compute the center point of each edge.
    
    Args:
        cluster_positions: Cluster positions (K, 3)
        edges: Edge indices (E, 2)
        
    Returns:
        Array of edge center points (E, 3)
    """
    if len(edges) == 0:
        return np.array([]).reshape(0, 3)
    
    edge_centers = (cluster_positions[edges[:, 0]] + cluster_positions[edges[:, 1]]) / 2.0
    return edge_centers


def compute_edge_stress_difference_distance(
    gt_stresses: np.ndarray,
    ours_stresses: np.ndarray,
    gt_edge_centers: np.ndarray,
    ours_edge_centers: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute Edge Stress Difference Distance (ESDD) between two meshes.
    
    For each edge in GT mesh, find the nearest edge in ours mesh and compute
    the absolute stress difference. Then do the same in reverse. The final metric
    is the average of the two mean differences (symmetric, like chamfer distance).
    
    Uses KDTree for fast nearest neighbor search instead of computing all pairwise
    distances, which is much faster for large meshes.
    
    Args:
        gt_stresses: Stress values for GT edges (E_gt,)
        ours_stresses: Stress values for ours edges (E_ours,)
        gt_edge_centers: Center points of GT edges (E_gt, 3)
        ours_edge_centers: Center points of ours edges (E_ours, 3)
        
    Returns:
        Tuple of (esdd, gt_to_ours_mean, ours_to_gt_mean)
        - esdd: Edge Stress Difference Distance (symmetric)
        - gt_to_ours_mean: Mean stress difference from GT to ours
        - ours_to_gt_mean: Mean stress difference from ours to GT
    """
    if len(gt_stresses) == 0 or len(ours_stresses) == 0:
        return float("inf"), float("inf"), float("inf")
    
    if len(gt_edge_centers) == 0 or len(ours_edge_centers) == 0:
        return float("inf"), float("inf"), float("inf")
    
    # Use KDTree for fast nearest neighbor search (much faster than cdist for large meshes)
    # Build KDTree for ours edges
    ours_tree = cKDTree(ours_edge_centers)
    
    # For each GT edge, find nearest ours edge and compute stress difference
    gt_to_ours_distances, gt_to_ours_nearest_indices = ours_tree.query(gt_edge_centers, k=1)
    gt_to_ours_stress_diffs = np.abs(
        gt_stresses - ours_stresses[gt_to_ours_nearest_indices]
    )
    gt_to_ours_mean = float(np.mean(gt_to_ours_stress_diffs))
    
    # Build KDTree for GT edges
    gt_tree = cKDTree(gt_edge_centers)
    
    # For each ours edge, find nearest GT edge and compute stress difference
    ours_to_gt_distances, ours_to_gt_nearest_indices = gt_tree.query(ours_edge_centers, k=1)
    ours_to_gt_stress_diffs = np.abs(
        ours_stresses - gt_stresses[ours_to_gt_nearest_indices]
    )
    ours_to_gt_mean = float(np.mean(ours_to_gt_stress_diffs))
    
    # Symmetric ESDD (like chamfer distance)
    esdd = (gt_to_ours_mean + ours_to_gt_mean) / 2.0
    
    return esdd, gt_to_ours_mean, ours_to_gt_mean


# Removed local generate_stress_colormap function, now using vu.get_colors


def create_interactive_comparison(
    gt_cluster_positions: np.ndarray,
    ours_cluster_positions: np.ndarray,
    gt_edges: np.ndarray,
    ours_edges: np.ndarray,
    gt_initial_lengths: np.ndarray,
    ours_initial_lengths: np.ndarray,
    gt_times: np.ndarray,
    ours_times: np.ndarray,
    esdd_distances: Dict[int, Tuple[float, float, float]],
) -> None:
    """
    Create an interactive 3D visualization comparing GT and ours meshes.
    
    Args:
        gt_cluster_positions: GT cluster positions (T, K, 3)
        ours_cluster_positions: Ours cluster positions (T, K, 3)
        gt_edges: GT edge indices (E_gt, 2)
        ours_edges: Ours edge indices (E_ours, 2)
        gt_initial_lengths: GT initial edge lengths (E_gt,)
        ours_initial_lengths: Ours initial edge lengths (E_ours,)
        gt_times: GT time array (T,)
        ours_times: Ours time array (T,)
        esdd_distances: Dict mapping frame indices to ESDD tuples
    """
    num_frames_gt = gt_cluster_positions.shape[0]
    num_frames_ours = ours_cluster_positions.shape[0]
    num_frames = min(num_frames_gt, num_frames_ours)
    
    frame_indices = list(range(num_frames))
    
    print(f"\nCreating interactive visualization for {num_frames} frames...")
    
    # Pre-compute mesh data for all frames
    frame_mesh_data = {}
    all_positions = []
    
    cmap = vu.get_colormap()
    
    # Offset for Ours vessel to distinguish it from GT
    ours_offset = np.array([0, -0.3, 0])  # Shift Ours down slightly in Y axis
    if gt_cluster_positions.size > 0:
        # Better heuristic: shift by 1.2x the bounding box height
        bbox_min = gt_cluster_positions[0].min(axis=0)
        bbox_max = gt_cluster_positions[0].max(axis=0)
        height = bbox_max[1] - bbox_min[1]
        ours_offset = np.array([0, -height * 1.2, 0])
    for frame_idx in frame_indices:
        # GT mesh
        gt_positions = gt_cluster_positions[frame_idx]
        gt_stresses = vu.compute_edge_stress(gt_positions, gt_edges, gt_initial_lengths, YOUNG_MODULUS_SILICON)
        gt_edge_centers = compute_edge_centers(gt_positions, gt_edges)
        gt_colors = vu.get_colors(gt_stresses, cmap, vmin=0, vmax=MAX_STRESS_PA, use_abs=True)
        
        # Ours mesh
        ours_positions = ours_cluster_positions[frame_idx]
        ours_positions_offset = ours_positions + ours_offset
        ours_stresses = vu.compute_edge_stress(ours_positions, ours_edges, ours_initial_lengths, YOUNG_MODULUS_SILICON)
        ours_edge_centers = compute_edge_centers(ours_positions_offset, ours_edges)
        ours_colors = vu.get_colors(ours_stresses, cmap, vmin=0, vmax=MAX_STRESS_PA, use_abs=True)
        
        # Create PolyData for edges
        if len(gt_edges) > 0 and len(gt_positions) > 0:
            gt_lines = np.hstack([np.full((gt_edges.shape[0], 1), 2), gt_edges])
            gt_polydata = pv.PolyData(gt_positions, lines=gt_lines)
        else:
            gt_polydata = None
        
        if len(ours_edges) > 0 and len(ours_positions_offset) > 0:
            ours_lines = np.hstack([np.full((ours_edges.shape[0], 1), 2), ours_edges])
            ours_polydata = pv.PolyData(ours_positions_offset, lines=ours_lines)
        else:
            ours_polydata = None
        
        frame_mesh_data[frame_idx] = {
            "gt_polydata": gt_polydata,
            "ours_polydata": ours_polydata,
            "gt_colors": gt_colors,
            "ours_colors": ours_colors,
            "gt_stresses": gt_stresses,
            "ours_stresses": ours_stresses,
        }
        
        all_positions.append(gt_positions)
        all_positions.append(ours_positions_offset)
    
    # Compute global bounds
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
    
    # Create frames_data structure for interactive utils
    frames_data = {}
    for frame_idx in frame_indices:
        gt_time = float(gt_times[frame_idx]) if gt_times is not None and len(gt_times) > frame_idx else float(frame_idx)
        ours_time = float(ours_times[frame_idx]) if ours_times is not None and len(ours_times) > frame_idx else float(frame_idx)
        mesh_data = frame_mesh_data[frame_idx]
        
        frames_data[frame_idx] = {
            "gt_time": gt_time,
            "ours_time": ours_time,
            "gt_edge_count": len(mesh_data["gt_stresses"]),
            "ours_edge_count": len(mesh_data["ours_stresses"]),
            "gt_stress_max": np.max(mesh_data["gt_stresses"]) / 1e6 if len(mesh_data["gt_stresses"]) > 0 else 0.0,
            "ours_stress_max": np.max(mesh_data["ours_stresses"]) / 1e6 if len(mesh_data["ours_stresses"]) > 0 else 0.0,
        }
    
    # Create PyVista plotter
    plotter = pv.Plotter(title="GT vs Ours Edge Stress Comparison")
    
    # Store the current actors
    current_gt_actor = None
    current_ours_actor = None
    
    def update_callback(frame_list_idx, frame_idx, plotter):
        """Update callback for frame changes - handles mesh updates."""
        nonlocal current_gt_actor, current_ours_actor
        
        mesh_data = frame_mesh_data[frame_idx]
        
        # Store old actors for removal after adding new ones (double-buffering to avoid flash)
        old_gt_actor = current_gt_actor
        old_ours_actor = current_ours_actor
        
        # Add GT mesh (yellow -> orange -> red) FIRST (before removing old ones to avoid flash)
        if mesh_data["gt_polydata"] is not None:
            current_gt_actor = plotter.add_mesh(
                mesh_data["gt_polydata"],
                scalars=mesh_data["gt_colors"],
                rgb=True,
                opacity=EDGE_OPACITY,
                line_width=EDGE_LINE_WIDTH,
                name=f"gt_edges_{frame_idx}",
            )
        else:
            current_gt_actor = None
        
        # Add ours mesh (purple -> blue -> green) FIRST (before removing old ones to avoid flash)
        if mesh_data["ours_polydata"] is not None:
            current_ours_actor = plotter.add_mesh(
                mesh_data["ours_polydata"],
                scalars=mesh_data["ours_colors"],
                rgb=True,
                opacity=EDGE_OPACITY,
                line_width=EDGE_LINE_WIDTH,
                name=f"ours_edges_{frame_idx}",
            )
        else:
            current_ours_actor = None
        
        # Render after adding new actors to ensure they're visible before removing old ones
        plotter.render()
        
        # NOW remove old actors (after new ones are already added and rendered)
        if old_gt_actor is not None:
            try:
                plotter.remove_actor(old_gt_actor)
            except Exception:
                pass
        
        if old_ours_actor is not None:
            try:
                plotter.remove_actor(old_ours_actor)
            except Exception:
                pass
    
    def text_generator(frame_idx, frame_data):
        """Generate text for frame display."""
        # Calculate padding width based on max frame number
        max_frame = frame_indices[-1]
        frame_width = len(str(max_frame))
        frame_str = f"{frame_idx:0{frame_width}d}"
        max_frame_str = f"{max_frame:0{frame_width}d}"
        
        text_parts = [
            f"Frame: {frame_str}/{max_frame_str}",
            f"Time: GT={frame_data['gt_time']:.4f}, Ours={frame_data['ours_time']:.4f}",
            f"GT Edges: {frame_data['gt_edge_count']} (max stress: {frame_data['gt_stress_max']:.2f} MPa)",
            f"Ours Edges: {frame_data['ours_edge_count']} (max stress: {frame_data['ours_stress_max']:.2f} MPa)",
        ]
        
        # Add ESDD info only if ESDD was computed
        if frame_idx in esdd_distances:
            esdd, gt_to_ours, ours_to_gt = esdd_distances[frame_idx]
            esdd_mpa = esdd / 1e6  # Convert to MPa
            text_parts.append(f"ESDD: {esdd_mpa:.4f} MPa")
        
        return " | ".join(text_parts)
    
    # Prepare instructions
    esdd_info = ""
    if COMPUTE_ESDD:
        esdd_info = "\n  ESDD: Edge Stress Difference Distance (symmetric metric)"
    
    instructions = create_default_instructions(
        visualization_description="Top: GT Edges, Bottom: Ours Edges",
        additional_info=[
            "Purple/Dark = Zero absolute stress, Yellow/Light = High absolute stress",
            f"Global colormap scale: 0 to {MAX_STRESS_MPA:.1f} MPa (absolute)",
        ] + ([esdd_info] if esdd_info else []),
    )
    
    # Use the standardized interactive visualization
    setup_interactive_visualization(
        plotter=plotter,
        frame_indices=frame_indices,
        frames_data=frames_data,
        update_callback=update_callback,
        text_generator=text_generator,
        instructions=instructions,
        bounds=bounds,
        title="GT vs Ours Edge Stress Comparison",
        font='courier',
        slider_pointa=(0.1, 0.02),
        slider_pointb=(0.9, 0.02),
    )
    
    print("Use the slider or arrow keys (Left/Right) to navigate frames\n")
    plotter.show()


def main():
    """Main entry point for GT vs Ours edge comparison."""
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    # Parse command-line arguments
    if len(sys.argv) >= 2:
        gt_npy_path_str = sys.argv[1]
    else:
        gt_npy_path_str = GT_CLUSTER_NPY_FILE
    
    if len(sys.argv) >= 3:
        ours_npy_path_str = sys.argv[2]
    else:
        ours_npy_path_str = OURS_CLUSTER_NPY_FILE
    
    if not gt_npy_path_str or not ours_npy_path_str:
        print("=" * 60)
        print("⚠️  Missing file paths.")
        print("Edit GT_CLUSTER_NPY_FILE and OURS_CLUSTER_NPY_FILE")
        print("or pass paths as command-line arguments:")
        print("  python compare_GT_ours_edges_interactive.py <gt_npy_path> <ours_npy_path>")
        print("=" * 60)
        return
    
    resolved_gt_path = _resolve_path(gt_npy_path_str, project_root)
    resolved_ours_path = _resolve_path(ours_npy_path_str, project_root)
    
    # Validate files exist
    if not resolved_gt_path.exists():
        print(f"⚠️  ERROR: GT file not found: {gt_npy_path_str}")
        print(f"   Resolved path: {resolved_gt_path}")
        return
    
    if not resolved_ours_path.exists():
        print(f"⚠️  ERROR: Ours file not found: {ours_npy_path_str}")
        print(f"   Resolved path: {resolved_ours_path}")
        return
    
    try:
        print(f"\n=== GT vs OURS EDGE COMPARISON ===")
        print(f"Loading GT data: {resolved_gt_path}")
        print(f"Loading ours data: {resolved_ours_path}")
        
        # Load GT data
        gt_cluster_positions, gt_edges, gt_initial_lengths, gt_times, gt_experiment = load_clustered_numpy_and_convert(resolved_gt_path)
        
        # Load ours data
        ours_cluster_positions, ours_edges, ours_initial_lengths, ours_times, ours_experiment = load_clustered_numpy_and_convert(resolved_ours_path)
        
        num_frames_gt = gt_cluster_positions.shape[0]
        num_frames_ours = ours_cluster_positions.shape[0]
        num_frames = min(num_frames_gt, num_frames_ours)
        
        # Compute ESDD for each frame (if enabled)
        esdd_distances = {}
        if COMPUTE_ESDD:
            print(f"\nComputing Edge Stress Difference Distance (ESDD) for {num_frames} frames...")
            frame_indices = list(range(num_frames))
            
            for frame_idx in frame_indices:
                # Get cluster positions for this frame
                gt_positions = gt_cluster_positions[frame_idx]
                ours_positions = ours_cluster_positions[frame_idx]
                
                # Compute stresses
                gt_stresses = vu.compute_edge_stress(gt_positions, gt_edges, gt_initial_lengths, YOUNG_MODULUS_SILICON)
                ours_stresses = vu.compute_edge_stress(ours_positions, ours_edges, ours_initial_lengths, YOUNG_MODULUS_SILICON)
                
                # Compute edge centers
                gt_edge_centers = compute_edge_centers(gt_positions, gt_edges)
                ours_edge_centers = compute_edge_centers(ours_positions, ours_edges)
                
                # Compute ESDD
                esdd, gt_to_ours, ours_to_gt = compute_edge_stress_difference_distance(
                    gt_stresses, ours_stresses,
                    gt_edge_centers, ours_edge_centers
                )
                
                esdd_distances[frame_idx] = (esdd, gt_to_ours, ours_to_gt)
                
                if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                    esdd_mpa = esdd / 1e6
                    print(f"  Frame {frame_idx}/{num_frames - 1}: ESDD = {esdd_mpa:.4f} MPa")
            
            # Print summary statistics
            if len(esdd_distances) > 0:
                all_esdd = [ed[0] for ed in esdd_distances.values()]
                all_esdd_mpa = np.array(all_esdd) / 1e6  # Convert to MPa
                print(f"\nEdge Stress Difference Distance (ESDD) Statistics:")
                print(f"  Frames analyzed: {len(esdd_distances)}")
                print(f"  Min ESDD: {np.min(all_esdd_mpa):.4f} MPa")
                print(f"  Max ESDD: {np.max(all_esdd_mpa):.4f} MPa")
                print(f"  Mean ESDD: {np.mean(all_esdd_mpa):.4f} MPa")
                print(f"  Median ESDD: {np.median(all_esdd_mpa):.4f} MPa")
                print(f"  Std dev: {np.std(all_esdd_mpa):.4f} MPa")
        else:
            print(f"\nSkipping ESDD computation (COMPUTE_ESDD = False)")
            print(f"Visualization will proceed without distance metrics.")
        
        # Create interactive visualization
        create_interactive_comparison(
            gt_cluster_positions=gt_cluster_positions,
            ours_cluster_positions=ours_cluster_positions,
            gt_edges=gt_edges,
            ours_edges=ours_edges,
            gt_initial_lengths=gt_initial_lengths,
            ours_initial_lengths=ours_initial_lengths,
            gt_times=gt_times,
            ours_times=ours_times,
            esdd_distances=esdd_distances,
        )
        
        print(f"\n✓ Completed GT vs Ours edge comparison")
        print(f"  GT: {gt_experiment}")
        print(f"  Ours: {ours_experiment}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n⚠️  ERROR during comparison:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
