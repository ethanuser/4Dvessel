import numpy as np
from numpy._core.numeric import True_
import pyvista as pv
import sys
import os
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay

import datetime
from pathlib import Path

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
# DELAUNAY MESHING HYPERPARAMETERS
# ============================================================================
OUTLIER_EDGE_THRESHOLD = 2.0  # Multiplier of median edge length to consider an edge an outlier
                               # Edges longer than (median_length * OUTLIER_EDGE_THRESHOLD) will be removed

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================
CLUSTER_POINT_SIZE = 0  # Size of cluster mean points
CLUSTER_POINT_COLOR = 'red'  # Color of cluster mean points
EDGE_LINE_WIDTH = 5  # Width of Delaunay edges
EDGE_COLOR = 'blue'  # Color of Delaunay edges
EDGE_OPACITY = 0.7  # Opacity of Delaunay edges
YOUNG_MODULUS_SILICON = vu.YOUNG_MODULUS_SILICON
MAX_STRESS_PA = vu.MAX_STRESS_PA

# Removed local generate_color_map, now using vu.get_colors

# ============================================================================
# INPUT DATA CONFIGURATION
# ============================================================================
# If READ_CLUSTERED_NUMPY is True, loads pre-computed clusters from exported numpy file.
# If False, loads npz file and computes clusters using DBSCAN.
READ_CLUSTERED_NUMPY = True  # Set to True to read from exported cluster mesh numpy file

# Path to npz file (used when READ_CLUSTERED_NUMPY = False)
# base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/blender-data")
# base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/synthetic_pull/blender-data")
# base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/s_pulls/lattice_strength_0_25/blender-data")
base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/synthetic_translate_real_params/blender-data")
# base_path = str(Path.home() / "Documents/Blender/blender-data")
single_file_path = f'{base_path}/mesh_data_single.npz'

# Path to clustered numpy file (used when READ_CLUSTERED_NUMPY = True)
# This should be the file exported by export_cluster_mesh.py
# CLUSTERED_NUMPY_FILE = r"cluster_data/cluster_mesh_timeseries_lattice_strength_0_50.npy"
CLUSTERED_NUMPY_FILE = r"cluster_data/cluster_export_synthetic_translate.npy"
# CLUSTERED_NUMPY_FILE = r"cluster_data/cluster_mesh_timeseries_synthetic_translate.npy"



# ============================================================================
# CLUSTERING AND MESHING FUNCTIONS
# ============================================================================


# =======================================================================c=====
# CLUSTERING AND MESHING FUNCTIONS
# ============================================================================
# Core functions moved to vessel_utils.py

def load_clustered_numpy_and_convert(clustered_npy_path):
    """
    Load clustered numpy file and convert to frames_data format.
    
    Args:
        clustered_npy_path: Path to the exported cluster mesh numpy file
        
    Returns:
        frames_data: Dictionary mapping frame indices to cluster data
        initial_cluster_means: Initial cluster positions (K, 3)
        edges: Edge indices (E, 2)
        initial_lengths: Initial edge lengths (E,)
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
    edges = data.get("edges")  # (E, 2)
    initial_lengths = data.get("initial_lengths")  # (E,)
    experiment_name = data.get("experiment_name", "UnknownExperiment")
    
    if cluster_positions is None:
        raise ValueError("Clustered numpy file missing 'cluster_positions' key")
    
    num_frames, num_clusters, _ = cluster_positions.shape
    
    print(f"  Loaded {num_frames} frames, {num_clusters} clusters")
    print(f"  Edges: {len(edges) if edges is not None else 0}")
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
    
    return frames_data, initial_cluster_means, edges, initial_lengths


def create_interactive_visualization(frames_data, initial_cluster_means=None, edges=None, initial_lengths=None, from_clustered_numpy=False):
    """Create an interactive 3D visualization with DBSCAN clustering and Delaunay meshing"""
    
    # Get all frame indices
    frame_indices = sorted(frames_data.keys())
    num_frames = len(frame_indices)
    
    if from_clustered_numpy:
        print(f"Interactive visualization: {num_frames} frames (from clustered numpy file)")
        if initial_cluster_means is None or edges is None or initial_lengths is None:
            raise ValueError("When from_clustered_numpy=True, initial_cluster_means, edges, and initial_lengths must be provided")
        base_cluster_means = initial_cluster_means.copy()
        valid_edges = edges.copy()
        removed_count = 0
        base_labels = None  # Not needed when loading from clustered numpy
    else:
        print(f"Interactive visualization: {num_frames} frames (DBSCAN eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})")
        
        # Cluster the first frame to establish the base cluster structure and edges
        first_frame_idx = frame_indices[0]
        first_frame_data = frames_data[first_frame_idx]
        first_coords = first_frame_data['coords']
        
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
            valid_edges = np.array([]).reshape(0, 2)
            removed_count = 0
        else:
            # Create Delaunay edges from first frame (this will be used for ALL frames)
            if len(base_cluster_means) >= 4:
                vertices, edges_raw, simplices = vu.create_delaunay_edges(base_cluster_means)
                # Remove outlier edges
                valid_edges, removed_count = vu.remove_outlier_edges(vertices, edges_raw, OUTLIER_EDGE_THRESHOLD)
            else:
                valid_edges = np.array([]).reshape(0, 2)
                removed_count = 0
        
        # Calculate initial edge lengths for strain calculation
        if len(valid_edges) > 0 and len(base_cluster_means) > 0:
            initial_lengths = np.linalg.norm(base_cluster_means[valid_edges[:, 0]] - base_cluster_means[valid_edges[:, 1]], axis=1)
        else:
            initial_lengths = None
    
    print(f"  Base mesh: {len(base_cluster_means)} clusters, {len(valid_edges)} edges")

    # Define the colormap
    cm = vu.get_colormap()

    # Process all frames: use labels from first frame to assign points to clusters
    frame_cluster_data = {}
    all_cluster_means = []
    
    for idx, frame_idx in enumerate(frame_indices):
        vu.print_progress_bar(idx + 1, len(frame_indices), bar_length=20)
        
        frame_data = frames_data[frame_idx]
        
        if from_clustered_numpy:
            # When loading from clustered numpy, cluster_means are already computed
            cluster_means = frame_data["cluster_means"]
            if len(cluster_means) == 0 or len(base_cluster_means) == 0:
                frame_cluster_data[frame_idx] = {
                    'cluster_means': np.array([]).reshape(0, 3),
                    'edges': valid_edges,
                    'num_clusters': 0,
                    'num_edges': len(valid_edges),
                    'removed_edges': removed_count,
                    'stress_colors': None
                }
                continue
        else:
            coords = frame_data['coords']
            
            if coords.shape[0] == 0 or len(base_cluster_means) == 0:
                frame_cluster_data[frame_idx] = {
                    'cluster_means': np.array([]).reshape(0, 3),
                    'edges': np.array([]).reshape(0, 2),
                    'num_clusters': 0,
                    'num_edges': 0
                }
                continue
            
            # CRITICAL: Follow reference code pattern perfectly
            # For first frame, use base cluster means
            # For subsequent frames, use labels to determine which points belong to which cluster
            # This matches: for k in unique(labels), cluster_mask = labels == k, then mean(points[cluster_mask])
            if idx == 0:
                # First frame: use the base cluster means from clustering
                cluster_means = base_cluster_means.copy()
            else:
                # Subsequent frames: use labels from first frame to assign points to clusters
                # This follows the reference code pattern exactly
                new_cluster_means = []
                
                # Get unique labels (excluding noise label -1), sorted to maintain order
                unique_labels = sorted([k for k in np.unique(base_labels) if k != -1])
                
                # Check if we have the same number of points (allows direct label usage)
                if len(coords) == len(base_labels):
                    # Same number of points - use labels directly (reference code pattern)
                    for k in unique_labels:
                        cluster_mask = base_labels == k
                        if np.any(cluster_mask):
                            new_cluster_mean = np.mean(coords[cluster_mask], axis=0)
                            new_cluster_means.append(new_cluster_mean)
                        else:
                            # No points for this label, use previous cluster mean
                            prev_frame_idx = frame_indices[idx - 1]
                            prev_cluster_means = frame_cluster_data[prev_frame_idx]['cluster_means']
                            cluster_idx = len(new_cluster_means)
                            if cluster_idx < len(prev_cluster_means):
                                new_cluster_means.append(prev_cluster_means[cluster_idx])
                            else:
                                new_cluster_means.append(base_cluster_means[cluster_idx])
                else:
                    # Different number of points - need to assign based on proximity to previous cluster positions
                    # This is a fallback when point counts don't match
                    prev_frame_idx = frame_indices[idx - 1]
                    prev_cluster_means = frame_cluster_data[prev_frame_idx]['cluster_means']
                    
                    if len(prev_cluster_means) != len(base_cluster_means):
                        prev_cluster_means = base_cluster_means.copy()
                    
                    # Assign each point to closest cluster from previous frame
                    new_cluster_means = np.zeros_like(base_cluster_means)
                    cluster_point_counts = np.zeros(len(base_cluster_means), dtype=int)
                    
                    for point in coords:
                        distances = np.linalg.norm(prev_cluster_means - point, axis=1)
                        closest_cluster_idx = np.argmin(distances)
                        new_cluster_means[closest_cluster_idx] += point
                        cluster_point_counts[closest_cluster_idx] += 1
                    
                    # Calculate mean for each cluster
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
                        # Pad with base positions
                        padded = np.zeros_like(base_cluster_means)
                        padded[:len(cluster_means)] = cluster_means
                        padded[len(cluster_means):] = base_cluster_means[len(cluster_means):]
                        cluster_means = padded
                    else:
                        cluster_means = cluster_means[:len(base_cluster_means)]
        
        # Store cluster means for bounding box calculation
        if len(cluster_means) > 0:
            all_cluster_means.append(cluster_means)
        
        
        # Calculate stress if we have initial lengths and valid edges
        stress_colors = None
        if initial_lengths is not None and len(valid_edges) > 0 and len(cluster_means) == len(base_cluster_means):
            # Use central Neo-Hookean calculation from vessel_utils
            stress = vu.compute_edge_stress(cluster_means, valid_edges, initial_lengths, YOUNG_MODULUS_SILICON)
            stress_colors = vu.get_colors(stress, cm, vmin=0, vmax=MAX_STRESS_PA, use_abs=True)

        # Store frame data - use the SAME valid_edges for all frames
        frame_cluster_data[frame_idx] = {
            'cluster_means': cluster_means,
            'edges': valid_edges,  # Same edges for all frames
            'num_clusters': len(cluster_means),
            'num_edges': len(valid_edges),
            'removed_edges': removed_count,
            'stress_colors': stress_colors
        }
    
    # Calculate bounding box across all cluster means
    if len(all_cluster_means) > 0:
        all_means = np.vstack(all_cluster_means)
        bounds = [
            all_means[:, 0].min(), all_means[:, 0].max(),  # x range
            all_means[:, 1].min(), all_means[:, 1].max(),  # y range
            all_means[:, 2].min(), all_means[:, 2].max()   # z range
        ]
    else:
        # Fallback: try to use coords if available, otherwise use initial cluster means or default bounds
        if from_clustered_numpy and initial_cluster_means is not None and len(initial_cluster_means) > 0:
            # Use initial cluster means as fallback when loading from clustered numpy
            bounds = [
                initial_cluster_means[:, 0].min(), initial_cluster_means[:, 0].max(),
                initial_cluster_means[:, 1].min(), initial_cluster_means[:, 1].max(),
                initial_cluster_means[:, 2].min(), initial_cluster_means[:, 2].max()
            ]
        elif len(frame_indices) > 0 and 'coords' in frames_data[frame_indices[0]]:
            # Fallback to original coords if available (when loading from npz)
            all_coords = np.vstack([frames_data[idx]['coords'] for idx in frame_indices if 'coords' in frames_data[idx]])
            if len(all_coords) > 0:
                bounds = [
                    all_coords[:, 0].min(), all_coords[:, 0].max(),
                    all_coords[:, 1].min(), all_coords[:, 1].max(),
                    all_coords[:, 2].min(), all_coords[:, 2].max()
                ]
            else:
                # Default bounds if nothing else works
                bounds = [-1, 1, -1, 1, -1, 1]
        else:
            # Default bounds if nothing else works
            bounds = [-1, 1, -1, 1, -1, 1]
    
    # Pre-create PolyData objects for cluster means and edges
    frame_mesh_data = {}
    for idx, frame_idx in enumerate(frame_indices):
        vu.print_progress_bar(idx + 1, len(frame_indices), bar_length=20)
        
        cluster_data = frame_cluster_data[frame_idx]
        cluster_means = cluster_data['cluster_means']
        edges = cluster_data['edges']
        
        # Create PolyData for cluster means
        if len(cluster_means) > 0:
            means_polydata = pv.PolyData(cluster_means)
        else:
            means_polydata = None
        
        # Create PolyData for edges
        if len(edges) > 0 and len(cluster_means) > 0:
            lines = np.hstack([np.full((edges.shape[0], 1), 2), edges])
            edges_polydata = pv.PolyData(cluster_means, lines=lines)
        else:
            edges_polydata = None
        
        frame_mesh_data[frame_idx] = {
            'means': means_polydata,
            'edges': edges_polydata,
            'stress_colors': cluster_data.get('stress_colors')
        }
    
    print(f"\n  Loaded {len(frame_mesh_data)} frames")
    
    # Create a PyVista plotter with interactive mode
    plotter = pv.Plotter(title=f"Interactive Vessel Visualization - Frame Navigation")
    
    # Store the current actors
    current_means_actor = None
    current_edges_actor = None
    
    def update_callback(frame_list_idx, frame_idx, plotter):
        """Update callback for frame changes - handles mesh updates."""
        nonlocal current_means_actor, current_edges_actor
        
        # Get mesh data for this frame
        if frame_idx in frame_mesh_data:
            mesh_data = frame_mesh_data[frame_idx]
            
            # Store old actors for removal after adding new ones (double-buffering to avoid flash)
            old_means_actor = current_means_actor
            old_edges_actor = current_edges_actor
            
            # Add new cluster means FIRST (before removing old ones to avoid flash)
            if mesh_data['means'] is not None:
                current_means_actor = plotter.add_mesh(
                    mesh_data['means'],
                    render_points_as_spheres=True,
                    point_size=CLUSTER_POINT_SIZE,
                    color=CLUSTER_POINT_COLOR,
                    name=f'cluster_means_{frame_idx}',
                    opacity=1.0 if CLUSTER_POINT_SIZE > 0 else 0.0,
                    lighting=False
                )
            else:
                current_means_actor = None
            
            # Add new Delaunay edges FIRST (before removing old ones to avoid flash)
            if mesh_data['edges'] is not None:
                # Use stress colors if available
                stress_colors = mesh_data.get('stress_colors')
                
                if stress_colors is not None:
                    current_edges_actor = plotter.add_mesh(
                        mesh_data['edges'],
                        scalars=stress_colors,
                        rgb=True,  # Important for RGBA colors
                        opacity=EDGE_OPACITY,
                        line_width=EDGE_LINE_WIDTH,
                        name=f'delaunay_edges_{frame_idx}',
                        lighting=False
                    )
                else:
                    current_edges_actor = plotter.add_mesh(
                        mesh_data['edges'],
                        color=EDGE_COLOR,
                        line_width=EDGE_LINE_WIDTH,
                        name=f'delaunay_edges_{frame_idx}',
                        lighting=False
                    )
            else:
                current_edges_actor = None
            
            # Render after adding new actors to ensure they're visible before removing old ones
            plotter.render()
            
            # NOW remove old actors (after new ones are already added and rendered)
            if old_means_actor is not None:
                try:
                    plotter.remove_actor(old_means_actor)
                except:
                    pass
            
            if old_edges_actor is not None:
                try:
                    plotter.remove_actor(old_edges_actor)
                except:
                    pass
    
    def text_generator(frame_idx, frame_data):
        """Generate text for frame display."""
        cluster_data = frame_cluster_data[frame_idx]
        
        # Calculate padding width based on max frame number
        max_frame = frame_indices[-1]
        frame_width = len(str(max_frame))
        frame_str = f"{frame_idx:0{frame_width}d}"
        max_frame_str = f"{max_frame:0{frame_width}d}"
        
        text_parts = [
            f"Frame: {frame_str}/{max_frame_str}",
            f"Time: {frame_data['time']:.4f}",
        ]
        
        # Only include original points if available
        if 'num_vertices' in frame_data:
            text_parts.append(f"Original Points: {frame_data['num_vertices']:,}")
        
        text_parts.extend([
            f"Clusters: {cluster_data['num_clusters']}",
            f"Edges: {cluster_data['num_edges']}"
        ])
        
        return " | ".join(text_parts)
    
    desc = "Red points: DBSCAN cluster means" if CLUSTER_POINT_SIZE > 0 else "Cluster means: Hidden"
    instructions = create_default_instructions(
        visualization_description=desc,
        additional_info=[
            "Blue lines: Delaunay mesh edges",
            f"Clustering: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}",
            f"Edge filtering: {OUTLIER_EDGE_THRESHOLD}x median length",
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
        title="Interactive Vessel Visualization - Frame Navigation",
        font='courier',
        slider_pointa=(0.1, 0.02),
        slider_pointb=(0.9, 0.02),
    )
    
    print("Displaying DBSCAN cluster means and Delaunay mesh edges\n")
    plotter.show()

def get_all_frames_data_local(coords, frame_numbers, times):
    # This might be redundant if using vu.get_all_frames_data, fitting to the existing structure
    return vu.get_all_frames_data(coords, frame_numbers, times)

try:
    if READ_CLUSTERED_NUMPY:
        # Load from clustered numpy file
        frames_data, initial_cluster_means, edges, initial_lengths = load_clustered_numpy_and_convert(CLUSTERED_NUMPY_FILE)
        print(f"Loaded {len(frames_data)} frames from clustered numpy")
        
        # Create interactive visualization
        create_interactive_visualization(
            frames_data,
            initial_cluster_means=initial_cluster_means,
            edges=edges,
            initial_lengths=initial_lengths,
            from_clustered_numpy=True
        )
    else:
        # Load the single file
        print(f"Loading numpy file: {single_file_path}")
        mesh_data = np.load(single_file_path)
        
        # Extract the arrays
        coords = mesh_data['coords']
        frame_numbers = mesh_data['frame_numbers']
        times = mesh_data['times']
        
        # Extract data organized by frames
        frames_data = vu.get_all_frames_data(coords, frame_numbers, times)
        
        # Print summary after processing (brief)
        total_vertices = sum(frame['num_vertices'] for frame in frames_data.values())
        print(f"Loaded {len(frames_data)} frames ({total_vertices} total points)")
        
        # Create interactive visualization
        create_interactive_visualization(frames_data, from_clustered_numpy=False)
    
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Error processing data: {e}")
    import traceback
    traceback.print_exc()