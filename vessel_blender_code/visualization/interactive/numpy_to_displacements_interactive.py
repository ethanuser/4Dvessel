import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import vessel_utils as vu
from utils.interactive_utils import (
    create_interactive_visualization as setup_interactive_visualization,
    create_default_text_generator,
    create_default_instructions,
)

# ============================================================================
# DBSCAN CLUSTERING HYPERPARAMETERS
# ============================================================================
DBSCAN_EPS = 0.045  # Maximum distance between samples in the same neighborhood
DBSCAN_MIN_SAMPLES = 2  # Minimum number of samples in a neighborhood to form a cluster

# ============================================================================
# PHYSICAL-SCALE CONSTANTS (UNIT -> mm CONVERSION)
# ============================================================================
# These constants mirror the displacement module you provided, so that cluster
# displacements are reported in physical millimetres.
CHESS_TOTAL_LENGTH_M = 6.195 / 100.0     # length across 8 squares (metres)
CHESS_SQUARES_ALONG_X = 8               # number of squares measured
CHESS_SQUARE_LENGTH_M = CHESS_TOTAL_LENGTH_M / CHESS_SQUARES_ALONG_X
CALIB_SCALING_FACTOR = 0.667125         # same factor used in calibration objp (units per square)
CORRECTION_FACTOR = 6.195 / 5.053       # ratio of true / calculated chessboard length
UNIT_TO_MM = (CHESS_SQUARE_LENGTH_M / CALIB_SCALING_FACTOR) * 1000.0 * CORRECTION_FACTOR

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================
CLUSTER_POINT_SIZE = 20  # Size of cluster mean points
DEFAULT_MAX_DISPLACEMENT_MM = vu.MAX_DISPLACEMENT_MM  # Default max displacement mapped to red

# ============================================================================
# AUTO-SCALING CONFIGURATION
# ============================================================================
AUTO_SCALE_MAX_DISPLACEMENT = True   # If True, overrides DEFAULT_MAX_DISPLACEMENT_MM
AUTO_SCALE_METRIC = "percentile"     # 'percentile' or 'max'
AUTO_SCALE_PERCENTILE_VALUE = 100    # Percentile for auto-scale when using 'percentile'

# ============================================================================
# INPUT DATA CONFIGURATION
# ============================================================================
# If READ_CLUSTERED_NUMPY is True, loads pre-computed clusters from exported numpy file.
# If False, loads npz file and computes clusters using DBSCAN.
READ_CLUSTERED_NUMPY = True  # Set to True to read from exported cluster mesh numpy file

# Path to npz file (used when READ_CLUSTERED_NUMPY = False)
# base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/blender-data")
# base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/s_pulls/lattice_strength_0_25/blender-data")
# single_file_path = f'{base_path}/mesh_data_single.npz'

# Path to clustered numpy file (used when READ_CLUSTERED_NUMPY = True)
# This should be the file exported by export_cluster_mesh.py
# CLUSTERED_NUMPY_FILE = r"cluster_data/cluster_mesh_timeseries_synthetic_translate_real_params.npy"
CLUSTERED_NUMPY_FILE = r"cluster_data/cluster_export_synthetic_translate.npy"


# Removed local generate_displacement_colormap and get_color_from_displacement, now using vu.get_colors


def load_clustered_numpy_and_convert(clustered_npy_path):
    """
    Load clustered numpy file and convert to frames_data format.
    
    Args:
        clustered_npy_path: Path to the exported cluster mesh numpy file
        
    Returns:
        frames_data: Dictionary mapping frame indices to cluster data
        initial_positions: Initial cluster positions (K, 3)
    """
    import os
    from pathlib import Path
    
    # Resolve path
    if not os.path.isabs(clustered_npy_path):
        script_dir = Path(__file__).parent
        resolved_path = script_dir / clustered_npy_path
        if not resolved_path.exists():
            resolved_path = Path(clustered_npy_path).resolve()
    else:
        resolved_path = Path(clustered_npy_path)
    
    if not resolved_path.exists():
        raise FileNotFoundError(f"Clustered numpy file not found: {clustered_npy_path}")
    
    print(f"Loading clustered numpy file: {resolved_path}")
    data = np.load(str(resolved_path), allow_pickle=True).item()
    
    cluster_positions = data.get("cluster_positions")  # (T, K, 3)
    times = data.get("times")  # (T,)
    initial_cluster_means = data.get("initial_cluster_means")  # (K, 3)
    experiment_name = data.get("experiment_name", "UnknownExperiment")
    
    if cluster_positions is None:
        raise ValueError("Clustered numpy file missing 'cluster_positions' key")
    
    num_frames, num_clusters, _ = cluster_positions.shape
    
    print(f"  Loaded {num_frames} frames, {num_clusters} clusters")
    print(f"  Experiment: {experiment_name}")
    
    # Convert to frames_data format
    frames_data = {}
    for frame_idx in range(num_frames):
        cluster_means = cluster_positions[frame_idx]  # (K, 3)
        time_val = float(times[frame_idx]) if times is not None and len(times) > frame_idx else float(frame_idx)
        
        frames_data[frame_idx] = {
            "cluster_means": cluster_means,
            "num_clusters": num_clusters,
            "time": time_val,
        }
    
    initial_positions = initial_cluster_means if initial_cluster_means is not None else cluster_positions[0]
    
    return frames_data, initial_positions


def create_interactive_visualization(frames_data, initial_positions=None, from_clustered_numpy=False):
    """
    Create an interactive 3D visualization of DBSCAN cluster means with
    colors representing per-cluster displacement (in mm) relative to the
    initial frame.
    """

    # Get all frame indices
    frame_indices = sorted(frames_data.keys())
    num_frames = len(frame_indices)

    if from_clustered_numpy:
        print(
            f"Interactive displacement visualization: {num_frames} frames "
            f"(from clustered numpy file)"
        )
        # Initial positions should be provided when loading from clustered numpy
        if initial_positions is None:
            first_frame_idx = frame_indices[0]
            initial_positions = frames_data[first_frame_idx]["cluster_means"]
    else:
        print(
            f"Interactive displacement visualization: {num_frames} frames "
            f"(DBSCAN eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})"
        )

    if not from_clustered_numpy:
        # Cluster the first frame to establish the base cluster structure
        first_frame_idx = frame_indices[0]
        first_frame_data = frames_data[first_frame_idx]
        first_coords = first_frame_data["coords"]

        if first_coords.shape[0] == 0:
            print("Error: First frame has no coordinates!")
            return

        # Apply DBSCAN clustering to first frame
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(first_coords)
        base_labels = clustering.labels_
        base_cluster_means, _ = vu.calculate_cluster_means(first_coords, base_labels)

        if len(base_cluster_means) == 0:
            print("Warning: No clusters found in first frame!")
            base_cluster_means = np.array([]).reshape(0, 3)

        print(f"  Base clusters: {len(base_cluster_means)} cluster means")

        # Initial positions (reference for displacement)
        initial_positions = base_cluster_means.copy()
    else:
        # When loading from clustered numpy, we already have cluster means
        base_cluster_means = initial_positions.copy()
        base_labels = None  # Not needed when loading from clustered numpy
        print(f"  Base clusters: {len(initial_positions)} cluster means")

    # Colormap for displacements
    cmap = vu.get_colormap()

    # Process all frames: reuse first-frame labels to track clusters
    frame_cluster_data = {}
    all_cluster_means = []
    all_displacements_mm_flat = []

    for idx, frame_idx in enumerate(frame_indices):
        vu.print_progress_bar(idx + 1, len(frame_indices), bar_length=20)

        frame_data = frames_data[frame_idx]
        
        if from_clustered_numpy:
            # When loading from clustered numpy, cluster_means are already computed
            cluster_means = frame_data["cluster_means"]
            if len(cluster_means) == 0 or len(initial_positions) == 0:
                frame_cluster_data[frame_idx] = {
                    "cluster_means": np.array([]).reshape(0, 3),
                    "num_clusters": 0,
                    "displacements_mm": np.array([]),
                    "displacement_colors": None,
                }
                continue
        else:
            coords = frame_data["coords"]

            if coords.shape[0] == 0 or len(base_cluster_means) == 0:
                frame_cluster_data[frame_idx] = {
                    "cluster_means": np.array([]).reshape(0, 3),
                    "num_clusters": 0,
                    "displacements_mm": np.array([]),
                    "displacement_colors": None,
                }
                continue

            # For first frame, use the base cluster means from clustering
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
                            prev_cluster_means = frame_cluster_data[prev_frame_idx]["cluster_means"]
                            cluster_idx = len(new_cluster_means)
                            if cluster_idx < len(prev_cluster_means):
                                new_cluster_means.append(prev_cluster_means[cluster_idx])
                            else:
                                new_cluster_means.append(base_cluster_means[cluster_idx])
                else:
                    # Different number of points - assign based on proximity to previous cluster positions
                    prev_frame_idx = frame_indices[idx - 1]
                    prev_cluster_means = frame_cluster_data[prev_frame_idx]["cluster_means"]

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
                        padded[len(cluster_means) :] = base_cluster_means[len(cluster_means) :]
                        cluster_means = padded
                    else:
                        cluster_means = cluster_means[: len(base_cluster_means)]

        # Store cluster means for bounding box
        if len(cluster_means) > 0:
            all_cluster_means.append(cluster_means)

        # Compute displacements relative to initial_positions
        displacements_mm = np.array([])
        if len(cluster_means) == len(initial_positions) and len(initial_positions) > 0:
            d_vec = cluster_means - initial_positions  # in model units
            displacements_units = np.linalg.norm(d_vec, axis=1)
            displacements_mm = displacements_units * UNIT_TO_MM
            all_displacements_mm_flat.append(displacements_mm)

        frame_cluster_data[frame_idx] = {
            "cluster_means": cluster_means,
            "num_clusters": len(cluster_means),
            "displacements_mm": displacements_mm,
            "displacement_colors": None,  # filled after we know the global scaling
        }

    # Determine max displacement value for color scaling
    if AUTO_SCALE_MAX_DISPLACEMENT and len(all_displacements_mm_flat) > 0:
        flat_disps = np.concatenate(all_displacements_mm_flat)
        if AUTO_SCALE_METRIC == "max":
            max_disp_val = float(np.max(flat_disps))
            print(f"\nAuto-scaling displacement colors using MAX = {max_disp_val:.4f} mm")
        elif AUTO_SCALE_METRIC == "percentile":
            max_disp_val = float(np.percentile(flat_disps, AUTO_SCALE_PERCENTILE_VALUE))
            print(
                f"\nAuto-scaling displacement colors using {AUTO_SCALE_PERCENTILE_VALUE}th percentile "
                f"= {max_disp_val:.4f} mm"
            )
        else:
            max_disp_val = float(DEFAULT_MAX_DISPLACEMENT_MM)
            print(
                f"\nUnknown AUTO_SCALE_METRIC '{AUTO_SCALE_METRIC}'. "
                f"Falling back to fixed max displacement = {max_disp_val:.4f} mm"
            )
    else:
        max_disp_val = float(DEFAULT_MAX_DISPLACEMENT_MM)
        print(f"\nUsing fixed max displacement for colors = {max_disp_val:.4f} mm")

    # Now compute per-frame displacement colors with the chosen max_disp_val
    for frame_idx in frame_indices:
        cluster_data = frame_cluster_data[frame_idx]
        disp_mm = cluster_data["displacements_mm"]
        if disp_mm.size > 0:
            cluster_data["displacement_colors"] = vu.get_colors(
                disp_mm, cmap, vmax=max_disp_val
            )

    # Calculate global bounding box across all cluster means
    if len(all_cluster_means) > 0:
        all_means = np.vstack(all_cluster_means)
        bounds = [
            all_means[:, 0].min(),
            all_means[:, 0].max(),
            all_means[:, 1].min(),
            all_means[:, 1].max(),
            all_means[:, 2].min(),
            all_means[:, 2].max(),
        ]
    else:
        # Fallback to original coords if no clusters found
        all_coords = np.vstack([frames_data[idx]["coords"] for idx in frame_indices])
        bounds = [
            all_coords[:, 0].min(),
            all_coords[:, 0].max(),
            all_coords[:, 1].min(),
            all_coords[:, 1].max(),
            all_coords[:, 2].min(),
            all_coords[:, 2].max(),
        ]

    # Pre-create PolyData for cluster means only (no meshing)
    frame_mesh_data = {}
    for idx, frame_idx in enumerate(frame_indices):
        vu.print_progress_bar(idx + 1, len(frame_indices), bar_length=20)

        cluster_data = frame_cluster_data[frame_idx]
        cluster_means = cluster_data["cluster_means"]

        if len(cluster_means) > 0:
            means_polydata = pv.PolyData(cluster_means)
        else:
            means_polydata = None

        frame_mesh_data[frame_idx] = {
            "means": means_polydata,
            "displacement_colors": cluster_data.get("displacement_colors"),
        }

    print(f"\n  Loaded {len(frame_mesh_data)} frames")

    # Create a PyVista plotter with interactive mode
    plotter = pv.Plotter(title="Interactive Displacement Visualization - Frame Navigation")

    # Store current actor for smooth updates
    current_means_actor = None

    def update_callback(frame_list_idx, frame_idx, plotter):
        """Update callback for frame changes - handles mesh updates."""
        nonlocal current_means_actor

        if frame_idx in frame_mesh_data:
            mesh_data = frame_mesh_data[frame_idx]

            old_means_actor = current_means_actor

            if mesh_data["means"] is not None:
                disp_colors = mesh_data.get("displacement_colors")
                if disp_colors is not None and len(disp_colors) > 0:
                    current_means_actor = plotter.add_mesh(
                        mesh_data["means"],
                        scalars=disp_colors,
                        rgb=True,
                        render_points_as_spheres=True,
                        point_size=CLUSTER_POINT_SIZE,
                        name=f"cluster_means_{frame_idx}",
                        lighting=False,
                    )
                else:
                    current_means_actor = plotter.add_mesh(
                        mesh_data["means"],
                        render_points_as_spheres=True,
                        point_size=CLUSTER_POINT_SIZE,
                        color="blue",
                        name=f"cluster_means_{frame_idx}",
                        lighting=False,
                    )
            else:
                current_means_actor = None

            if old_means_actor is not None:
                try:
                    plotter.remove_actor(old_means_actor)
                except Exception:
                    pass

    def text_generator(frame_idx, frame_data):
        """Generate text for frame display."""
        cluster_data = frame_cluster_data[frame_idx]
        displacements_mm = cluster_data.get("displacements_mm", np.array([]))

        max_disp_text = f"{np.max(displacements_mm):.2f}" if displacements_mm.size > 0 else "0.00"
        mean_disp_text = f"{np.mean(displacements_mm):.2f}" if displacements_mm.size > 0 else "0.00"

        # Calculate padding width based on max frame number
        max_frame = frame_indices[-1]
        frame_width = len(str(max_frame))
        frame_str = f"{frame_idx:0{frame_width}d}"
        max_frame_str = f"{max_frame:0{frame_width}d}"

        # Build text parts conditionally
        text_parts = [
            f"Frame: {frame_str}/{max_frame_str}",
            f"Time: {frame_data['time']:.4f}",
        ]
        
        # Only include original points if available (not present when loading from clustered numpy)
        if 'num_vertices' in frame_data:
            text_parts.append(f"Original Points: {frame_data['num_vertices']:,}")
        
        text_parts.extend([
            f"Clusters: {cluster_data['num_clusters']}",
            f"Max Disp: {max_disp_text} mm",
            f"Mean Disp: {mean_disp_text} mm",
            f"Scale Max: {max_disp_val:.2f} mm",
        ])
        
        return " | ".join(text_parts)

    # Prepare instructions
    instructions = create_default_instructions(
        visualization_description="Colored points: DBSCAN cluster means",
        additional_info=[
            "Blue = Low displacement, Red = High displacement",
            f"Clustering: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}",
            f"Displacements reported in mm (UNIT_TO_MM={UNIT_TO_MM:.4f})",
        ],
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
        title="Interactive Displacement Visualization - Frame Navigation",
        font='courier',
    )

    print("Displaying DBSCAN cluster means with per-cluster displacements (mm).\n")
    plotter.show()


def get_all_frames_data_local(coords, frame_numbers, times):
    """
    Convenience wrapper to match the pattern in numpy_to_stress_interactive.py.
    """
    return vu.get_all_frames_data(coords, frame_numbers, times)


if __name__ == "__main__":
    try:
        if READ_CLUSTERED_NUMPY:
            # Load from clustered numpy file
            frames_data, initial_positions = load_clustered_numpy_and_convert(CLUSTERED_NUMPY_FILE)
            print(f"Loaded {len(frames_data)} frames from clustered numpy")
            
            # Run interactive displacement visualization
            create_interactive_visualization(frames_data, initial_positions=initial_positions, from_clustered_numpy=True)
        else:
            # Load the single file (same pattern as numpy_to_stress_interactive.py)
            print(f"Loading numpy file: {single_file_path}")
            mesh_data = np.load(single_file_path)

            # Extract the arrays
            coords = mesh_data["coords"]
            frame_numbers = mesh_data["frame_numbers"]
            times = mesh_data["times"]

            # Organize data by frames
            frames_data = vu.get_all_frames_data(coords, frame_numbers, times)

            total_vertices = sum(frame["num_vertices"] for frame in frames_data.values())
            print(f"Loaded {len(frames_data)} frames ({total_vertices} total points)")

            # Run interactive displacement visualization
            create_interactive_visualization(frames_data, from_clustered_numpy=False)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()