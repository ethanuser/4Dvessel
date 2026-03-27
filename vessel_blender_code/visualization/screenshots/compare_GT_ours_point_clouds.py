#!/usr/bin/env python3
"""
Compare ground truth and ours point clouds by calculating chamfer distance
between cluster data from two sources.

This script:
1. Loads cluster positions from the cluster_data numpy file (GT)
2. Loads ours clusters (either from clustered numpy or computes DBSCAN from npz)
3. Calculates chamfer distance between the two cluster sets
4. Creates screenshots showing both cluster sets for each frame
5. Saves renders to a timestamped folder

Usage:
    python compare_GT_ours_point_clouds.py [cluster_npy_path] [npz_path]

Arguments:
    cluster_npy_path: Path to the .npy file with cluster positions (GT)
    npz_path: Path to the .npz file with raw point data (only used if READ_CLUSTERED_NUMPY = False)
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple
from scipy.spatial.distance import cdist
import datetime

import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import vessel_utils as vu

# ============================================================================
# CONFIGURATION
# ============================================================================
# If READ_CLUSTERED_NUMPY is True, loads pre-computed clusters from exported numpy file for ours side.
# If False, loads npz file and computes clusters using DBSCAN.
READ_CLUSTERED_NUMPY = True  # Set to True to read from exported cluster mesh numpy file for ours side

# Default paths
CLUSTER_NPY_FILE = r"cluster_data/cluster_mesh_timeseries_GT_synthetic_translate.npy"
NPZ_FILE = str(Path.home() / "Documents/Blender/Vessel-Renders/synthetic_translate_160_frames/blender-data/mesh_data_single.npz")

# Path to clustered numpy file for ours side (used when READ_CLUSTERED_NUMPY = True)
# This should be the file exported by export_cluster_mesh.py
OURS_CLUSTERED_NUMPY_FILE = r"cluster_data/mesh_exports_bundle/cluster_export_synthetic_translate_rgd2.npy"

# DBSCAN parameters (matching numpy_to_displacements_interactive.py)
DBSCAN_EPS = 0.045
DBSCAN_MIN_SAMPLES = 2

# Visualization constants
CLUSTER_POINT_SIZE = 20
STRESS_CLUSTER_COLOR = "blue"  # Clusters from cluster_data numpy (GT)
DISPLACEMENT_CLUSTER_COLOR = "orange"  # Clusters from DBSCAN on npz data (Ours)

# Rendering configuration
SAVE_RENDERS = True  # If True, save screenshots to render folder
OUTPUT_DIR = r"renders"  # Base output directory for renders
FRAME_DELAY = 0.0  # Delay between frames (not used for screenshots, but kept for consistency)


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


def compute_dbscan_clusters_for_frames(
    frames_data: Dict[int, Any],
) -> Dict[int, np.ndarray]:
    """
    Compute DBSCAN cluster means for each frame, matching the logic from
    numpy_to_displacements_interactive.py.

    Args:
        frames_data: Dictionary mapping frame indices to frame data

    Returns:
        Dictionary mapping frame indices to cluster means arrays
    """
    frame_indices = sorted(frames_data.keys())
    cluster_means_dict = {}

    # Cluster the first frame to establish base structure
    first_frame_idx = frame_indices[0]
    first_frame_data = frames_data[first_frame_idx]
    first_coords = first_frame_data["coords"]

    if first_coords.shape[0] == 0:
        print("Error: First frame has no coordinates!")
        return cluster_means_dict

    # Apply DBSCAN clustering to first frame
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(
        first_coords
    )
    base_labels = clustering.labels_
    base_cluster_means, _ = vu.calculate_cluster_means(first_coords, base_labels)

    if len(base_cluster_means) == 0:
        print("Warning: No clusters found in first frame!")
        base_cluster_means = np.array([]).reshape(0, 3)

    print(f"  Base clusters: {len(base_cluster_means)} cluster means")

    # Process all frames
    for idx, frame_idx in enumerate(frame_indices):
        vu.print_progress_bar(idx + 1, len(frame_indices), bar_length=20)

        frame_data = frames_data[frame_idx]
        coords = frame_data["coords"]

        if coords.shape[0] == 0 or len(base_cluster_means) == 0:
            cluster_means_dict[frame_idx] = np.array([]).reshape(0, 3)
            continue

        # For first frame, use the base cluster means
        if idx == 0:
            cluster_means = base_cluster_means.copy()
        else:
            # Subsequent frames: use labels from first frame to assign points to clusters
            new_cluster_means = []

            # Get unique labels (excluding noise label -1), sorted to maintain order
            unique_labels = sorted([k for k in np.unique(base_labels) if k != -1])

            # If the point count matches, we can reuse base_labels directly
            if len(coords) == len(base_labels):
                for k in unique_labels:
                    cluster_mask = base_labels == k
                    if np.any(cluster_mask):
                        new_cluster_mean = np.mean(coords[cluster_mask], axis=0)
                        new_cluster_means.append(new_cluster_mean)
                    else:
                        # No points for this label, fall back to previous frame's cluster mean
                        prev_frame_idx = frame_indices[idx - 1]
                        prev_cluster_means = cluster_means_dict.get(
                            prev_frame_idx, base_cluster_means
                        )
                        cluster_idx = len(new_cluster_means)
                        if cluster_idx < len(prev_cluster_means):
                            new_cluster_means.append(prev_cluster_means[cluster_idx])
                        else:
                            new_cluster_means.append(base_cluster_means[cluster_idx])
            else:
                # Different number of points - assign based on proximity to previous cluster positions
                prev_frame_idx = frame_indices[idx - 1]
                prev_cluster_means = cluster_means_dict.get(
                    prev_frame_idx, base_cluster_means
                )

                if len(prev_cluster_means) != len(base_cluster_means):
                    prev_cluster_means = base_cluster_means.copy()

                new_cluster_means = np.zeros_like(base_cluster_means)
                cluster_point_counts = np.zeros(len(base_cluster_means), dtype=int)

                for point in coords:
                    distances = np.linalg.norm(prev_cluster_means - point, axis=1)
                    closest_cluster_idx = np.argmin(distances)
                    new_cluster_means[closest_cluster_idx] += point
                    cluster_point_counts[closest_cluster_idx] += 1

                for k in range(len(base_cluster_means)):
                    if cluster_point_counts[k] > 0:
                        new_cluster_means[k] /= cluster_point_counts[k]
                    else:
                        new_cluster_means[k] = prev_cluster_means[k]

            # Convert to numpy array if it's a list
            if isinstance(new_cluster_means, list):
                cluster_means = np.array(new_cluster_means)
            else:
                cluster_means = new_cluster_means

            # Ensure cluster_means has the same length as base_cluster_means
            if len(cluster_means) != len(base_cluster_means):
                if len(cluster_means) < len(base_cluster_means):
                    padded = np.zeros_like(base_cluster_means)
                    padded[: len(cluster_means)] = cluster_means
                    padded[len(cluster_means) :] = base_cluster_means[
                        len(cluster_means) :
                    ]
                    cluster_means = padded
                else:
                    cluster_means = cluster_means[: len(base_cluster_means)]

        cluster_means_dict[frame_idx] = cluster_means

    print()  # New line after progress bar
    return cluster_means_dict


def create_screenshot_comparison(
    cluster_positions: np.ndarray,
    dbscan_cluster_means: Dict[int, np.ndarray],
    times_cluster: np.ndarray,
    times_npz: Dict[int, float],
    chamfer_distances: Dict[int, Tuple[float, float, float]],
    save_path: str,
) -> None:
    """
    Create screenshots comparing clusters from both sources for each frame.

    Args:
        cluster_positions: Array of shape (T, K, 3) from cluster_data numpy
        dbscan_cluster_means: Dict mapping frame indices to cluster means from DBSCAN
        times_cluster: Array of times from cluster_data numpy
        times_npz: Dict mapping frame indices to times from npz data
        chamfer_distances: Dict mapping frame indices to chamfer distance tuples
        save_path: Path to save screenshots
    """
    num_frames = cluster_positions.shape[0]
    frame_indices_cluster = list(range(num_frames))

    # Get frame indices from DBSCAN data
    frame_indices_dbscan = sorted(dbscan_cluster_means.keys())

    # Find common frame indices (use cluster_data as reference)
    common_frames = []
    for frame_idx in frame_indices_cluster:
        # Try to find matching frame in DBSCAN data
        if frame_idx in frame_indices_dbscan:
            common_frames.append(frame_idx)
        elif len(frame_indices_dbscan) > 0:
            # Use closest frame index
            closest_idx = min(
                frame_indices_dbscan, key=lambda x: abs(x - frame_idx)
            )
            common_frames.append(frame_idx)

    if len(common_frames) == 0:
        print("ERROR: No common frames found between the two data sources!")
        return

    print(f"\nCreating screenshots for {len(common_frames)} frames...")

    # Precompute global bounds across all frames
    all_positions = []
    for frame_idx in common_frames:
        if frame_idx < num_frames:
            all_positions.append(cluster_positions[frame_idx])
        if frame_idx in dbscan_cluster_means:
            all_positions.append(dbscan_cluster_means[frame_idx])

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

    # Create PyVista plotter (non-interactive)
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")

    # Add axes
    plotter.add_axes()

    # Set camera position
    plotter.camera_position = "xy"
    plotter.reset_camera(bounds=bounds)

    # Render each frame
    for frame_list_idx, frame_idx in enumerate(common_frames):
        # Progress
        progress = int((frame_list_idx / len(common_frames)) * 50)
        bar = "█" * progress + "░" * (50 - progress)
        print(f"\r[{bar}] Frame {frame_list_idx + 1}/{len(common_frames)}", end="", flush=True)

        # Clear previous meshes
        plotter.clear()

        # Add axes again after clear
        plotter.add_axes()

        # Cluster positions from cluster_data numpy (blue - GT)
        if frame_idx < num_frames:
            stress_clusters = cluster_positions[frame_idx]
            if len(stress_clusters) > 0:
                stress_poly = pv.PolyData(stress_clusters)
                plotter.add_mesh(
                    stress_poly,
                    render_points_as_spheres=True,
                    point_size=CLUSTER_POINT_SIZE,
                    color=STRESS_CLUSTER_COLOR,
                )

        # Cluster means from DBSCAN on npz data (orange - Ours)
        if frame_idx in dbscan_cluster_means:
            dbscan_clusters = dbscan_cluster_means[frame_idx]
            if len(dbscan_clusters) > 0:
                dbscan_poly = pv.PolyData(dbscan_clusters)
                plotter.add_mesh(
                    dbscan_poly,
                    render_points_as_spheres=True,
                    point_size=CLUSTER_POINT_SIZE,
                    color=DISPLACEMENT_CLUSTER_COLOR,
                )

        # Get frame info
        time_val = (
            float(times_cluster[frame_idx])
            if times_cluster is not None
            and len(times_cluster) > frame_idx
            else float(frame_idx)
        )

        stress_count = (
            len(cluster_positions[frame_idx])
            if frame_idx < num_frames
            else 0
        )
        dbscan_count = (
            len(dbscan_cluster_means[frame_idx])
            if frame_idx in dbscan_cluster_means
            else 0
        )

        # Get chamfer distance info
        chamfer_info = "N/A"
        if frame_idx in chamfer_distances:
            chamfer_dist, a_to_b, b_to_a = chamfer_distances[frame_idx]
            chamfer_info = f"{chamfer_dist:.4f}"

        # Add text overlay
        text_content = (
            f"Frame: {frame_idx}/{common_frames[-1]}\n"
            f"Time: {time_val:.4f}\n"
            f"Blue (GT): {stress_count} clusters\n"
            f"Orange (Ours): {dbscan_count} clusters\n"
            f"Chamfer Distance: {chamfer_info}"
        )
        plotter.add_text(text_content, font_size=6, position="upper_left")

        # Save screenshot
        screenshot_path = os.path.join(save_path, f"frame_{frame_idx:04d}.png")
        plotter.screenshot(screenshot_path)

    print()  # New line after progress bar
    plotter.close()


def main():
    """
    Main entry point for GT vs Ours point cloud comparison.
    """
    # Determine project root (script directory)
    script_dir = Path(__file__).parent
    project_root = script_dir

    # Parse command-line arguments
    if len(sys.argv) >= 2:
        cluster_npy_path_str = sys.argv[1]
    else:
        cluster_npy_path_str = CLUSTER_NPY_FILE

    if not READ_CLUSTERED_NUMPY:
        if len(sys.argv) >= 3:
            npz_path_str = sys.argv[2]
        else:
            npz_path_str = NPZ_FILE
    else:
        npz_path_str = None  # Not needed when reading clustered numpy

    if not cluster_npy_path_str:
        print("=" * 60)
        print("⚠️  Missing cluster numpy file path.")
        print("Edit CLUSTER_NPY_FILE in compare_GT_ours_point_clouds.py")
        print("or pass the path as a command-line argument:")
        print(
            "  python compare_GT_ours_point_clouds.py <cluster_npy_path> [npz_path]"
        )
        print("=" * 60)
        return

    if not READ_CLUSTERED_NUMPY and not npz_path_str:
        print("=" * 60)
        print("⚠️  Missing NPZ file path (required when READ_CLUSTERED_NUMPY = False).")
        print("Edit NPZ_FILE in compare_GT_ours_point_clouds.py")
        print("or pass the path as a command-line argument:")
        print(
            "  python compare_GT_ours_point_clouds.py <cluster_npy_path> <npz_path>"
        )
        print("=" * 60)
        return

    resolved_cluster_path = _resolve_path(cluster_npy_path_str, project_root)

    # Validate files exist
    if not resolved_cluster_path.exists():
        print(f"⚠️  ERROR: Cluster file not found: {cluster_npy_path_str}")
        print(f"   Resolved path: {resolved_cluster_path}")
        return

    if not READ_CLUSTERED_NUMPY:
        resolved_npz_path = _resolve_path(npz_path_str, project_root)
        if not resolved_npz_path.exists():
            print(f"⚠️  ERROR: NPZ file not found: {npz_path_str}")
            print(f"   Resolved path: {resolved_npz_path}")
            return
    else:
        resolved_npz_path = None  # Not needed when reading clustered numpy

    # Create output directory
    if SAVE_RENDERS:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"gt_ours_comparison_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        print(f"Renders will be saved to: {save_path}")

    try:
        print(f"\n=== GT vs OURS POINT CLOUD COMPARISON ===")
        print(f"Loading GT cluster data: {resolved_cluster_path}")
        if READ_CLUSTERED_NUMPY:
            print(f"Loading ours clustered numpy data: {OURS_CLUSTERED_NUMPY_FILE}")
        else:
            print(f"Loading ours npz data: {resolved_npz_path}")

        # Load cluster data numpy
        cluster_data = np.load(str(resolved_cluster_path), allow_pickle=True).item()
        cluster_positions = cluster_data.get("cluster_positions")  # (T, K, 3)
        times_cluster = cluster_data.get("times")  # (T,)
        experiment_name = cluster_data.get("experiment_name", "UnknownExperiment")

        if cluster_positions is None:
            print("ERROR: Cluster file is missing 'cluster_positions' key.")
            return

        num_frames = cluster_positions.shape[0]
        print(f"Cluster data: {num_frames} frames, {cluster_positions.shape[1]} clusters per frame")

        # Load ours data (either from npz or clustered numpy)
        if READ_CLUSTERED_NUMPY:
            print(f"\nLoading ours clustered numpy file...")
            ours_resolved_path = _resolve_path(OURS_CLUSTERED_NUMPY_FILE, project_root)
            
            if not ours_resolved_path.exists():
                print(f"⚠️  ERROR: Ours clustered numpy file not found: {OURS_CLUSTERED_NUMPY_FILE}")
                print(f"   Resolved path: {ours_resolved_path}")
                return
            
            ours_data = np.load(str(ours_resolved_path), allow_pickle=True).item()
            ours_cluster_positions = ours_data.get("cluster_positions")  # (T, K, 3)
            ours_times = ours_data.get("times")  # (T,)
            
            if ours_cluster_positions is None:
                print("ERROR: Ours clustered numpy file missing 'cluster_positions' key.")
                return
            
            # Convert to dict format matching compute_dbscan_clusters_for_frames output
            dbscan_cluster_means = {}
            for frame_idx in range(ours_cluster_positions.shape[0]):
                dbscan_cluster_means[frame_idx] = ours_cluster_positions[frame_idx]
            
            # Create times dict
            times_npz = {}
            for frame_idx in range(len(ours_times)):
                times_npz[frame_idx] = float(ours_times[frame_idx])
            
            print(f"Ours clustered numpy: {ours_cluster_positions.shape[0]} frames, {ours_cluster_positions.shape[1]} clusters per frame")
        else:
            # Load npz data
            print(f"\nLoading npz file and computing DBSCAN clusters...")
            npz_data = np.load(str(resolved_npz_path))
            coords = npz_data["coords"]
            frame_numbers = npz_data["frame_numbers"]
            times_npz_array = npz_data["times"]

            # Organize npz data by frames
            frames_data = vu.get_all_frames_data(coords, frame_numbers, times_npz_array)

            # Create times dict for npz data
            times_npz = {}
            for frame_idx, frame_data in frames_data.items():
                times_npz[frame_idx] = frame_data["time"]

            print(f"NPZ data: {len(frames_data)} frames")

            # Compute DBSCAN clusters for each frame
            print("\nComputing DBSCAN clusters for each frame...")
            dbscan_cluster_means = compute_dbscan_clusters_for_frames(frames_data)

        # Compute chamfer distances for each frame
        print("\nComputing chamfer distances for each frame...")
        chamfer_distances = {}
        frame_indices_cluster = list(range(num_frames))
        frame_indices_dbscan = sorted(dbscan_cluster_means.keys())

        for frame_idx in frame_indices_cluster:
            if frame_idx < num_frames:
                stress_clusters = cluster_positions[frame_idx]
            else:
                continue

            # Find matching frame in DBSCAN data
            if frame_idx in frame_indices_dbscan:
                dbscan_clusters = dbscan_cluster_means[frame_idx]
            elif len(frame_indices_dbscan) > 0:
                # Use closest frame
                closest_idx = min(frame_indices_dbscan, key=lambda x: abs(x - frame_idx))
                dbscan_clusters = dbscan_cluster_means[closest_idx]
            else:
                continue

            # Compute chamfer distance
            chamfer_dist, a_to_b, b_to_a = compute_chamfer_distance(
                stress_clusters, dbscan_clusters
            )
            chamfer_distances[frame_idx] = (chamfer_dist, a_to_b, b_to_a)

            if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                print(
                    f"  Frame {frame_idx}/{num_frames - 1}: "
                    f"Chamfer = {chamfer_dist:.4f}"
                )

        # Print summary statistics for chamfer distance
        if len(chamfer_distances) > 0:
            all_chamfer = [cd[0] for cd in chamfer_distances.values()]
            print(f"\nChamfer Distance Statistics:")
            print(f"  Frames analyzed: {len(chamfer_distances)}")
            print(f"  Min chamfer distance: {np.min(all_chamfer):.4f}")
            print(f"  Max chamfer distance: {np.max(all_chamfer):.4f}")
            print(f"  Mean chamfer distance: {np.mean(all_chamfer):.4f}")
            print(f"  Median chamfer distance: {np.median(all_chamfer):.4f}")
            print(f"  Std dev: {np.std(all_chamfer):.4f}")

        # Create screenshots
        if SAVE_RENDERS:
            create_screenshot_comparison(
                cluster_positions=cluster_positions,
                dbscan_cluster_means=dbscan_cluster_means,
                times_cluster=times_cluster,
                times_npz=times_npz,
                chamfer_distances=chamfer_distances,
                save_path=save_path,
            )
            print(f"\n✓ Screenshots saved to: {save_path}")

        print(f"\n✓ Completed GT vs Ours comparison for: {experiment_name}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n⚠️  ERROR during comparison:")
        print(f"   {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

