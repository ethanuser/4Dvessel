#!/usr/bin/env python3
"""
Export cluster positions and mesh connections as a function of time.

This script reads the same numpy data as numpy_to_displacements_interactive.py
and exports cluster positions and mesh connectivity to a single NumPy file.

The export format matches cluster_mesh_export.py:
  - cluster_positions: (T, K, 3) array of cluster centroids over time
  - edges: (E, 2) array of edge indices from Delaunay triangulation
  - initial_lengths: (E,) initial edge lengths at t=0
  - times: (T,) time values for each frame
  - experiment_name: Name of the experiment
  - initial_cluster_means: (K, 3) initial cluster positions

Usage:
    python export_cluster_mesh.py [npz_path] [output_path] [experiment_name]

Arguments:
    npz_path: Path to the .npz file with mesh data (default: from numpy_to_displacements_interactive.py)
    output_path: Path to save the exported .npy file (default: cluster_data/cluster_mesh_timeseries_<experiment_name>.npy)
    experiment_name: Name of the experiment (default: inferred from npz_path or "UnknownExperiment")
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.cluster import DBSCAN
import vessel_utils as vu

# ============================================================================
# CONFIGURATION
# ============================================================================
# DBSCAN parameters (matching numpy_to_displacements_interactive.py)
DBSCAN_EPS = 0.045
DBSCAN_MIN_SAMPLES = 2

# Delaunay edge filtering threshold (matching numpy_to_stress_interactive.py)
OUTLIER_EDGE_THRESHOLD = 2.0  # Multiplier of median edge length

# ============================================================================
# BATCH EXPORT CONFIGURATION
# ============================================================================
# If NPZ_FILES_TO_EXPORT is not empty, exports all files in the list.
# If empty, requires a command-line argument to specify the npz file.
# Each entry can be a string (npz path) or a dict with 'npz_path' and optional 'experiment_name' and 'output_path'
NPZ_FILES_TO_EXPORT = [
    # Example format 1: Just the path (experiment name and output path will be auto-generated)
    # str(Path.home() / "Documents/Blender/Vessel-Renders/s_pulls/lattice_strength_0_25/blender-data/mesh_data_single.npz"),
    
    # Example format 2: Dict with custom experiment name and output path
    # {
    #     "npz_path": str(Path.home() / "Documents/Blender/Vessel-Renders/s_pulls/lattice_strength_0_25/blender-data/mesh_data_single.npz"),
    #     "experiment_name": "lattice_strength_0_25",
    #     "output_path": r"cluster_data/cluster_mesh_timeseries_lattice_strength_0_25.npy"
    # },


    str(Path.home() / "Documents/Blender/Vessel-Renders/bend_twist/old_deforms/blender-data/mesh_data_single.npz"),
    # str(Path.home() / "Documents/Blender/Vessel-Renders/s_pulls/lattice_strength_0_25/blender-data/mesh_data_single.npz"),
    # str(Path.home() / "Documents/Blender/Vessel-Renders/s_pulls/lattice_strength_0_50/blender-data/mesh_data_single.npz"),
    # str(Path.home() / "Documents/Blender/Vessel-Renders/s_pulls/lattice_strength_0_75/blender-data/mesh_data_single.npz"),
    # str(Path.home() / "Documents/Blender/Vessel-Renders/s_pulls/lattice_strength_1_00/blender-data/mesh_data_single.npz"),
    # str(Path.home() / "Documents/Blender/Vessel-Renders/s_pulls/lattice_strength_1_25/blender-data/mesh_data_single.npz"),
]

# Default output directory
DEFAULT_OUTPUT_DIR = r"cluster_data"


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


def _compute_cluster_trajectories(
    labels: np.ndarray,
    initial_cluster_means: np.ndarray,
    frames_data: Dict[int, Any],
    base_labels: np.ndarray,
    unique_labels: list,
) -> Dict[str, Any]:
    """
    Compute cluster centroids over time using fixed clustering labels.

    This follows the pattern used in numpy_to_displacements_interactive.py:
    - labels are fixed, defining which points belong to each cluster index k
    - for each frame, we recompute the centroid for each k; if a cluster has
      no points in the current frame, we keep the previous position.

    Args:
        labels: Cluster labels from first frame (N,)
        initial_cluster_means: Initial cluster means (K, 3)
        frames_data: Dictionary mapping frame indices to frame data
        base_labels: Base labels from first frame (for reference)

    Returns:
        Dictionary with:
            - cluster_positions: (T, K, 3) array of centroids over time
            - edges: (E, 2) array of edge indices
            - initial_lengths: (E,) initial edge lengths
    """
    frame_indices = sorted(frames_data.keys())
    num_frames = len(frame_indices)
    num_clusters = len(initial_cluster_means)

    # Create Delaunay edges from initial cluster means
    if len(initial_cluster_means) >= 4:
        vertices, raw_edges, _ = vu.create_delaunay_edges(initial_cluster_means)
        # Remove outlier edges
        edges, removed_count = vu.remove_outlier_edges(
            vertices, raw_edges, OUTLIER_EDGE_THRESHOLD
        )
        print(f"  Created {len(edges)} edges (removed {removed_count} outliers)")
    else:
        edges = np.array([]).reshape(0, 2)
        removed_count = 0
        print(f"  Warning: Not enough clusters ({num_clusters}) for Delaunay triangulation")

    # Compute initial edge lengths
    if len(edges) > 0:
        initial_lengths = np.linalg.norm(
            initial_cluster_means[edges[:, 0]] - initial_cluster_means[edges[:, 1]], axis=1
        )
    else:
        initial_lengths = np.array([])

    # Allocate output array: (T, K, 3)
    cluster_positions = np.zeros((num_frames, num_clusters, 3), dtype=np.float64)

    # Store t=0 positions
    cluster_positions[0] = initial_cluster_means.copy()

    # Track previous frame's cluster means for continuity
    prev_cluster_means = initial_cluster_means.copy()

    print("\nCalculating cluster trajectories over time...")
    total_frames = num_frames - 1

    for j, frame_idx in enumerate(frame_indices[1:], start=1):
        # Progress bar
        progress = int((j / total_frames) * 50)
        bar = "█" * progress + "░" * (50 - progress)
        print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)

        frame_data = frames_data[frame_idx]
        coords = frame_data["coords"]

        if coords.shape[0] == 0 or len(initial_cluster_means) == 0:
            # No points or no clusters - keep previous positions
            cluster_positions[j] = prev_cluster_means.copy()
            continue

        # CRITICAL: maintain same cluster index structure across frames
        # Map DBSCAN labels to cluster indices (0, 1, 2, ...)
        new_cluster_means = np.zeros_like(initial_cluster_means)

        # If the point count matches, we can reuse labels directly
        if len(coords) == len(base_labels):
            # Iterate over cluster indices (0, 1, 2, ...) which correspond to unique_labels
            for cluster_idx, dbscan_label in enumerate(unique_labels):
                cluster_mask = base_labels == dbscan_label
                if np.any(cluster_mask):
                    new_cluster_means[cluster_idx] = np.mean(coords[cluster_mask], axis=0)
                else:
                    # If a cluster has no points in this frame, keep previous position
                    new_cluster_means[cluster_idx] = prev_cluster_means[cluster_idx]
        else:
            # Different number of points - assign based on proximity to previous cluster positions
            cluster_point_counts = np.zeros(len(initial_cluster_means), dtype=int)

            for point in coords:
                distances = np.linalg.norm(prev_cluster_means - point, axis=1)
                closest_cluster_idx = np.argmin(distances)
                new_cluster_means[closest_cluster_idx] += point
                cluster_point_counts[closest_cluster_idx] += 1

            # Calculate mean for each cluster
            for k in range(len(initial_cluster_means)):
                if cluster_point_counts[k] > 0:
                    new_cluster_means[k] /= cluster_point_counts[k]
                else:
                    new_cluster_means[k] = prev_cluster_means[k]

        prev_cluster_means = new_cluster_means.copy()
        cluster_positions[j] = new_cluster_means

    print()  # newline after progress bar

    return {
        "cluster_positions": cluster_positions,
        "edges": edges,
        "initial_lengths": initial_lengths,
    }


def export_cluster_mesh(
    npz_path: str,
    output_path: str = None,
    experiment_name: str = None,
) -> None:
    """
    Export cluster positions and mesh connections as a function of time.

    Args:
        npz_path: Path to the .npz file with mesh data
        output_path: Path to save the exported .npy file (optional)
        experiment_name: Name of the experiment (optional)
    """
    # Determine project root (parent of export/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Resolve input path
    resolved_npz_path = _resolve_path(npz_path, project_root)

    if not resolved_npz_path.exists():
        print(f"ERROR: NPZ file not found: {npz_path}")
        print(f"   Resolved path: {resolved_npz_path}")
        return

    # Determine experiment name
    if experiment_name is None:
        # Derive experiment name from the directory structure of the NPZ path.
        # For example:
        #   /Users/YOUR_USERNAME/Documents/Blender/Vessel-Renders/synthetic_pull_160_frames/blender-data/mesh_data_single.npz
        # -> experiment_name = synthetic_pull_160_frames
        try:
            # Parent of the NPZ file is typically 'blender-data', so we take its parent
            experiment_dir = resolved_npz_path.parents[1]
            experiment_name = experiment_dir.name
        except Exception:
            # Fallback if the path structure is unexpected
            experiment_name = "UnknownExperiment"

    # Determine output path
    if output_path is None:
        output_dir = project_root / DEFAULT_OUTPUT_DIR
        output_dir.mkdir(exist_ok=True)
        output_filename = f"cluster_mesh_timeseries_{experiment_name}.npy"
        output_path = output_dir / output_filename
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== CLUSTER MESH EXPORT ===")
    print(f"Input: {resolved_npz_path}")
    print(f"Output: {output_path}")
    print(f"Experiment: {experiment_name}")

    # Load npz data
    print(f"\nLoading npz file...")
    npz_data = np.load(str(resolved_npz_path))
    coords = npz_data["coords"]
    frame_numbers = npz_data["frame_numbers"]
    times_npz_array = npz_data["times"]

    print(f"  Total points: {len(coords):,}")
    print(f"  Unique frames: {len(np.unique(frame_numbers))}")

    # Organize data by frames
    frames_data = vu.get_all_frames_data(coords, frame_numbers, times_npz_array)

    frame_indices = sorted(frames_data.keys())
    num_frames = len(frame_indices)

    print(f"  Frames: {num_frames}")

    # Cluster the first frame to establish base structure
    first_frame_idx = frame_indices[0]
    first_frame_data = frames_data[first_frame_idx]
    first_coords = first_frame_data["coords"]

    if first_coords.shape[0] == 0:
        print("ERROR: First frame has no coordinates!")
        return

    # Apply DBSCAN clustering to first frame
    print(f"\nApplying DBSCAN clustering to first frame...")
    print(f"  Parameters: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}")
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(first_coords)
    base_labels = clustering.labels_
    initial_cluster_means, _ = vu.calculate_cluster_means(first_coords, base_labels)

    if len(initial_cluster_means) == 0:
        print("ERROR: No clusters found in first frame!")
        return

    print(f"  Found {len(initial_cluster_means)} clusters")

    # Get unique labels (excluding noise label -1), sorted to maintain order
    # This matches the logic in numpy_to_displacements_interactive.py
    unique_labels = sorted([k for k in np.unique(base_labels) if k != -1])

    # Compute trajectories over time
    # Note: We use base_labels directly, and map DBSCAN labels to cluster indices
    traj_data = _compute_cluster_trajectories(
        labels=base_labels,
        initial_cluster_means=initial_cluster_means,
        frames_data=frames_data,
        base_labels=base_labels,
        unique_labels=unique_labels,
    )

    cluster_positions = traj_data["cluster_positions"]
    edges_used = traj_data["edges"]
    initial_lengths = traj_data["initial_lengths"]

    # Build time axis from frames_data
    times = np.array([frames_data[idx]["time"] for idx in frame_indices], dtype=np.float64)

    # Prepare export data
    export_data = {
        "experiment_name": experiment_name,
        "times": times,  # (T,)
        "cluster_positions": cluster_positions,  # (T, K, 3)
        "edges": edges_used,  # (E, 2)
        "initial_cluster_means": initial_cluster_means,  # (K, 3)
        "initial_lengths": initial_lengths,  # (E,)
    }

    # Save to file
    np.save(str(output_path), export_data, allow_pickle=True)

    print("\n✓ Export complete.")
    print(f"  File: {output_path}")
    print(f"  Frames: {cluster_positions.shape[0]}")
    print(f"  Clusters: {cluster_positions.shape[1]}")
    print(f"  Edges: {len(edges_used)}")
    print(f"  Initial lengths: {len(initial_lengths)}")


def main():
    """
    Main entry point for cluster mesh export.
    Supports both batch export (from NPZ_FILES_TO_EXPORT list) and single file export.
    """
    # Determine project root (parent of export/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Check if batch export is configured
    if NPZ_FILES_TO_EXPORT:
        print(f"\n{'='*70}")
        print(f"BATCH EXPORT MODE: Processing {len(NPZ_FILES_TO_EXPORT)} files")
        print(f"{'='*70}\n")
        
        successful = 0
        failed = 0
        
        for idx, entry in enumerate(NPZ_FILES_TO_EXPORT, 1):
            print(f"\n{'='*70}")
            print(f"FILE {idx}/{len(NPZ_FILES_TO_EXPORT)}")
            print(f"{'='*70}\n")
            
            # Handle both string and dict formats
            if isinstance(entry, dict):
                npz_path_str = entry.get("npz_path")
                experiment_name_str = entry.get("experiment_name")
                output_path_str = entry.get("output_path")
            else:
                # String format - just the path
                npz_path_str = entry
                experiment_name_str = None
                output_path_str = None
            
            if not npz_path_str:
                print(f"⚠️  Skipping entry {idx}: No npz_path specified")
                failed += 1
                continue
            
            try:
                export_cluster_mesh(
                    npz_path=npz_path_str,
                    output_path=output_path_str,
                    experiment_name=experiment_name_str,
                )
                successful += 1
                print(f"\n✓ Completed file {idx}/{len(NPZ_FILES_TO_EXPORT)}")
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                failed += 1
                print(f"\n⚠️  ERROR during export of file {idx}:")
                print(f"   {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"BATCH EXPORT COMPLETE: {successful} successful, {failed} failed")
        print(f"{'='*70}\n")
        
    else:
        # Single file export mode (via command-line arguments)
        # Parse command-line arguments
        if len(sys.argv) >= 2:
            npz_path_str = sys.argv[1]
        else:
            print("=" * 60)
            print("⚠️  No NPZ file specified.")
            print("Either:")
            print("  1. Add files to NPZ_FILES_TO_EXPORT list in export_cluster_mesh.py")
            print("  2. Pass the path as a command-line argument:")
            print("     python export_cluster_mesh.py <npz_path> [output_path] [experiment_name]")
            print("=" * 60)
            return

        output_path_str = None
        if len(sys.argv) >= 3:
            output_path_str = sys.argv[2]

        experiment_name_str = None
        if len(sys.argv) >= 4:
            experiment_name_str = sys.argv[3]

        try:
            export_cluster_mesh(
                npz_path=npz_path_str,
                output_path=output_path_str,
                experiment_name=experiment_name_str,
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n⚠️  ERROR during export:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

