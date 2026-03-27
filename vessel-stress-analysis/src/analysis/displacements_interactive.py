"""
Interactive displacement visualization module.
Creates an interactive 3D visualization with frame navigation showing cluster point displacements.
Uses color-then-spatial clustering and supports loading saved clustering state.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import glob
from typing import List, Tuple, Optional
from matplotlib.colors import LinearSegmentedColormap

# Flag to control whether to use cluster mesh export file
# If True: checks for export file and uses it if found (default behavior)
# If False: skips export file check and always computes cluster positions on the fly
USE_CLUSTER_EXPORT = True

from core.clustering import (
    color_then_spatial_clustering,
)
import utils.vessel_utils as vu
from utils.clustering_state_utils import (
    load_saved_clustering_state,
    remap_labels_to_saved_means
)
import utils.vessel_utils as vu

# ── Physical-scale constants ──────────────────────────────────────────────
# Scale factors for converting 3D units to real-world MM
CHESS_TOTAL_LENGTH_M    = 6.195 / 100        # length across 8 squares (metres)
CHESS_SQUARES_ALONG_X   = 8              # number of squares measured
CHESS_SQUARE_LENGTH_M   = CHESS_TOTAL_LENGTH_M / CHESS_SQUARES_ALONG_X  # ~0.774375 cm per square
CALIB_SCALING_FACTOR    = 0.667125       # same factor used in calibration objp (units per square)
CORRECTION_FACTOR       = 6.195 / 5.053 # ratio of the length of the chessboard in real life to the length of the chessboard we've calculated
UNIT_TO_MM              = (CHESS_SQUARE_LENGTH_M / CALIB_SCALING_FACTOR) * 1000 * CORRECTION_FACTOR  # Convert to mm

# ── Visualization constants ──────────────────────────────────────────────
DEFAULT_MAX_DISPLACEMENT_MM = vu.MAX_DISPLACEMENT_MM
USE_DEFAULT_POINT_SIZE = True
DEFAULT_POINT_SIZE = 20

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================
AUTO_SCALE_MAX_DISPLACEMENT = False
AUTO_SCALE_METRIC = 'percentile'
AUTO_SCALE_PERCENTILE_VALUE = 95

# Colors and colormaps are now handled by vessel_utils (vu.get_colors)

def create_interactive_visualization(frames_data: dict, config_manager=None, saved_means=None, cluster_positions_timeseries=None):
    """
    Create an interactive 3D visualization showing cluster point displacements.
    
    Args:
        frames_data: Dictionary of frame data with 'coords', 'time', 'num_vertices' keys
        config_manager: Configuration manager instance (optional)
        saved_means: Saved cluster means from clustering state (optional)
        cluster_positions_timeseries: Pre-computed cluster positions (T, K, 3) array (optional)
    """
    # Get all frame indices
    frame_indices = sorted(frames_data.keys())
    num_frames = len(frame_indices)
    
    print(f"Interactive displacement visualization: {num_frames} frames")
    
    first_frame_idx = frame_indices[0]
    first_frame_data = frames_data[first_frame_idx]
    first_coords = first_frame_data['coords']
    
    # If we have pre-computed trajectory, we don't strictly NEED first frame coords for clustering
    skip_clustering = cluster_positions_timeseries is not None and saved_means is not None
    
    if first_coords.shape[0] == 0 and not skip_clustering:
        print("Error: First frame has no coordinates and no pre-computed trajectory provided!")
        return
    
    # Get clustering parameters from config or use defaults
    if config_manager is not None:
        config = config_manager.config
        clustering_config = config.get('clustering', {})
        n_color_clusters = clustering_config.get('n_color_clusters', 5)
        spatial_eps = clustering_config.get('spatial_eps', 0.045)
        spatial_min_samples = clustering_config.get('min_cluster_points', 2)
        if USE_DEFAULT_POINT_SIZE:
            point_size = DEFAULT_POINT_SIZE
        else:
            point_size = config.get('visualization', {}).get('initial_point_size', DEFAULT_POINT_SIZE)
        max_disp_config_mm = config.get('visualization', {}).get('max_displacement_mm', DEFAULT_MAX_DISPLACEMENT_MM)
    else:
        n_color_clusters = 5
        spatial_eps = 0.045
        spatial_min_samples = 2
        point_size = DEFAULT_POINT_SIZE
        max_disp_config_mm = DEFAULT_MAX_DISPLACEMENT_MM
    
    # Determine Base Clusters
    if skip_clustering:
        print("Skipping initial clustering (using pre-computed trajectory)")
        base_cluster_means = saved_means
        base_labels = None # Not needed when using pre-computed trajectory
    else:
        # First, apply color clustering to get initial labels
        if 'rgb' in first_frame_data and first_frame_data['rgb'] is not None:
            first_rgb = np.clip(first_frame_data['rgb'], 0.0, 1.0)
            if config_manager is not None:
                initial_labels, _ = color_then_spatial_clustering(first_coords, first_rgb, config_manager.config)
            else:
                # Fallback to DBSCAN if no config
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=spatial_eps, min_samples=spatial_min_samples).fit(first_coords)
                initial_labels = clustering.labels_
        else:
            # No RGB data, use DBSCAN
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=spatial_eps, min_samples=spatial_min_samples).fit(first_coords)
            initial_labels = clustering.labels_
        
        # Calculate initial cluster means from labels
        initial_cluster_means, _ = vu.calculate_cluster_means(first_coords, initial_labels)
    
    # Now handle saved state vs. fresh clustering
    if not skip_clustering:
        if saved_means is not None:
            print("Using SAVED CLUSTERING STATE from mesh editor.")
            print(f"Applied filtering: {len(initial_cluster_means)} -> {len(saved_means)} cluster means")
            
            # Remap labels to saved cluster means
            base_labels, _ = remap_labels_to_saved_means(
                first_coords, initial_labels, initial_cluster_means, saved_means
            )
            
            # Use saved means
            base_cluster_means = saved_means
            
            # Calculate actual first frame cluster means from remapped labels
            if cluster_positions_timeseries is not None:
                # Use initial cluster means from export instead of re-calculating from raw points
                base_cluster_means = saved_means
            elif len(base_cluster_means) > 0:
                temp_means = np.zeros_like(base_cluster_means)
                for k in range(len(base_cluster_means)):
                    cluster_mask = base_labels == k
                    if np.any(cluster_mask):
                        temp_means[k] = np.mean(first_coords[cluster_mask], axis=0)
                    else:
                        temp_means[k] = base_cluster_means[k]  # Use saved position if no points
                base_cluster_means = temp_means
        else:
            print(f"No saved state. Using color-then-spatial clustering.")
            # Check if first frame has RGB data
            if 'rgb' in first_frame_data and first_frame_data['rgb'] is not None:
                first_rgb = np.clip(first_frame_data['rgb'], 0.0, 1.0)
                if config_manager is not None:
                    base_labels, _ = color_then_spatial_clustering(first_coords, first_rgb, config_manager.config)
                else:
                    # Fallback to DBSCAN if no config
                    from sklearn.cluster import DBSCAN
                    clustering = DBSCAN(eps=spatial_eps, min_samples=spatial_min_samples).fit(first_coords)
                    base_labels = clustering.labels_
            else:
                # No RGB data, use DBSCAN
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=spatial_eps, min_samples=spatial_min_samples).fit(first_coords)
                base_labels = clustering.labels_
            
            base_cluster_means, _ = vu.calculate_cluster_means(first_coords, base_labels)
            
            if len(base_cluster_means) == 0:
                print("Warning: No clusters found in first frame!")
                base_cluster_means = np.array([]).reshape(0, 3)
    
    print(f"  Base clusters: {len(base_cluster_means)} cluster means")
    
    # Initial Positions (t=0) - reference for displacement calculation
    initial_positions = base_cluster_means.copy()
    
    # Define the displacement colormap
    cm = vu.get_colormap()
    
    # Process all frames: use labels from first frame to assign points to clusters
    frame_cluster_data = {}
    all_cluster_means = []
    all_displacements_mm_flat = []  # For auto-scaling
    
    # Determine if we're using pre-computed cluster positions
    use_precomputed = cluster_positions_timeseries is not None
    
    if use_precomputed:
        print("\nUsing pre-computed cluster positions from cluster mesh export")
        num_frames_traj = cluster_positions_timeseries.shape[0]
        
        for idx, frame_idx in enumerate(frame_indices):
            vu.print_progress_bar(idx + 1, len(frame_indices), bar_length=20)
            
            # Map frame_idx (usually integer 0..T) to trajectory index
            traj_idx = min(idx, num_frames_traj - 1)
            cluster_means = cluster_positions_timeseries[traj_idx]
            
            # Store cluster means for bounding box calculation
            if len(cluster_means) > 0:
                all_cluster_means.append(cluster_means)
            
            # Calculate displacement from initial positions (t=0)
            displacement_colors = None
            if len(cluster_means) == len(initial_positions) and len(initial_positions) > 0:
                d_vec = cluster_means - initial_positions
                displacements_units = np.linalg.norm(d_vec, axis=1)
                displacements_mm = displacements_units * UNIT_TO_MM
                all_displacements_mm_flat.append(displacements_mm)
            else:
                displacements_mm = np.array([])
            
            # Store frame data
            frame_cluster_data[frame_idx] = {
                'cluster_means': cluster_means,
                'displacements_mm': displacements_mm,
                'num_clusters': len(cluster_means),
                'displacement_colors': None
            }
    else:
        for idx, frame_idx in enumerate(frame_indices):
            vu.print_progress_bar(idx + 1, len(frame_indices), bar_length=20)
            
            frame_data = frames_data[frame_idx]
            coords = frame_data['coords']
            
            if coords.shape[0] == 0 or len(base_cluster_means) == 0:
                frame_cluster_data[frame_idx] = {
                    'cluster_means': np.array([]).reshape(0, 3),
                    'displacements_mm': np.array([]),
                    'num_clusters': 0,
                    'displacement_colors': None
                }
                continue
            
            # For first frame, use base cluster means
            if idx == 0:
                cluster_means = base_cluster_means.copy()
            else:
                # Subsequent frames: use labels from first frame to assign points to clusters
                new_cluster_means = []
                
                # Get unique labels (excluding noise label -1), sorted to maintain order
                unique_labels = sorted([k for k in np.unique(base_labels) if k != -1])
                
                # Check if we have the same number of points (allows direct label usage)
                if len(coords) == len(base_labels):
                    # Same number of points - use labels directly
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
                    # Different number of points - assign based on proximity to previous cluster positions
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
            
            # Calculate displacement from initial positions
            displacement_colors = None
            if len(cluster_means) == len(initial_positions) and len(initial_positions) > 0:
                # Calculate displacement in UNITS
                d_vec = cluster_means - initial_positions
                displacements_units = np.linalg.norm(d_vec, axis=1)
                
                # Convert to MM
                displacements_mm = displacements_units * UNIT_TO_MM
                all_displacements_mm_flat.append(displacements_mm)
                
                # Calculate colors (will be set after auto-scaling)
                displacement_colors = None  # Will be calculated after determining max_disp_val
            else:
                displacements_mm = np.array([])
            
            # Store frame data
            frame_cluster_data[frame_idx] = {
                'cluster_means': cluster_means,
                'displacements_mm': displacements_mm,
                'num_clusters': len(cluster_means),
                'displacement_colors': displacement_colors
            }
    
    # Determine Max Displacement for Scaling
    if AUTO_SCALE_MAX_DISPLACEMENT and len(all_displacements_mm_flat) > 0:
        flat_disps = np.concatenate(all_displacements_mm_flat)
        if AUTO_SCALE_METRIC == 'max':
            max_disp_val = np.max(flat_disps)
            print(f"\nAuto-scaling: Using MAX displacement = {max_disp_val:.4f} mm")
        elif AUTO_SCALE_METRIC == 'percentile':
            max_disp_val = np.percentile(flat_disps, AUTO_SCALE_PERCENTILE_VALUE)
            print(f"\nAuto-scaling: Using {AUTO_SCALE_PERCENTILE_VALUE}th percentile = {max_disp_val:.4f} mm")
        else:
            max_disp_val = max_disp_config_mm  # Fallback
    else:
        max_disp_val = max_disp_config_mm
        print(f"\nUsing fixed max displacement = {max_disp_val:.4f} mm")
    
    # Now calculate colors for all frames with the determined max_disp_val
    for frame_idx in frame_indices:
        cluster_data = frame_cluster_data[frame_idx]
        if len(cluster_data['displacements_mm']) > 0:
            cluster_data['displacement_colors'] = vu.get_colors(
                cluster_data['displacements_mm'], cm, vmin=0, vmax=max_disp_val
            )
    
    # Calculate bounding box across all cluster means
    if len(all_cluster_means) > 0:
        all_means = np.vstack(all_cluster_means)
        bounds = [
            all_means[:, 0].min(), all_means[:, 0].max(),
            all_means[:, 1].min(), all_means[:, 1].max(),
            all_means[:, 2].min(), all_means[:, 2].max()
        ]
    else:
        # Fallback to original coords if no clusters found
        all_coords = np.vstack([frames_data[idx]['coords'] for idx in frame_indices])
        bounds = [
            all_coords[:, 0].min(), all_coords[:, 0].max(),
            all_coords[:, 1].min(), all_coords[:, 1].max(),
            all_coords[:, 2].min(), all_coords[:, 2].max()
        ]
    
    # Pre-create PolyData objects for cluster means
    frame_mesh_data = {}
    for idx, frame_idx in enumerate(frame_indices):
        vu.print_progress_bar(idx + 1, len(frame_indices), bar_length=20)
        
        cluster_data = frame_cluster_data[frame_idx]
        cluster_means = cluster_data['cluster_means']
        
        # Create PolyData for cluster means
        if len(cluster_means) > 0:
            means_polydata = pv.PolyData(cluster_means)
        else:
            means_polydata = None
        
        frame_mesh_data[frame_idx] = {
            'means': means_polydata,
            'displacement_colors': cluster_data.get('displacement_colors'),
            'displacements_mm': cluster_data.get('displacements_mm', np.array([]))
        }
    
    print(f"\n  Loaded {len(frame_mesh_data)} frames")
    
    # Create a PyVista plotter with interactive mode
    plotter = pv.Plotter(title="Interactive Displacement Visualization - Frame Navigation")
    plotter.set_background('white')
    
    # Store the current actors
    current_means_actor = None
    current_text_actor = None
    current_frame_idx = 0
    
    def update_frame_display(frame_list_idx):
        """Update the display to show a specific frame with cluster means colored by displacement"""
        nonlocal current_means_actor, current_text_actor, current_frame_idx
        
        if frame_list_idx < 0 or frame_list_idx >= len(frame_indices):
            return
        
        current_frame_idx = frame_list_idx
        frame_idx = frame_indices[frame_list_idx]
        
        # Get mesh data for this frame
        if frame_idx in frame_mesh_data:
            mesh_data = frame_mesh_data[frame_idx]
            cluster_data = frame_cluster_data[frame_idx]
            
            # Store old actor for removal after adding new one
            old_means_actor = current_means_actor
            
            # Add new cluster means with displacement colors
            if mesh_data['means'] is not None:
                displacement_colors = mesh_data.get('displacement_colors')
                
                if displacement_colors is not None and len(displacement_colors) > 0:
                    current_means_actor = plotter.add_mesh(
                        mesh_data['means'],
                        scalars=displacement_colors,
                        rgb=True,
                        render_points_as_spheres=True,
                        point_size=point_size,
                        name=f'cluster_means_{frame_idx}'
                    )
                else:
                    # Fallback: use default color if no displacement colors
                    current_means_actor = plotter.add_mesh(
                        mesh_data['means'],
                        render_points_as_spheres=True,
                        point_size=point_size,
                        color='blue',
                        name=f'cluster_means_{frame_idx}'
                    )
            else:
                current_means_actor = None
            
            # NOW remove old actor
            if old_means_actor is not None:
                try:
                    plotter.remove_actor(old_means_actor)
                except:
                    pass
        
        # Update text
        if current_text_actor is not None:
            try:
                plotter.remove_actor(current_text_actor)
            except:
                pass
            current_text_actor = None
        
        # Update title with current frame info
        frame_data = frames_data[frame_idx]
        cluster_data = frame_cluster_data[frame_idx]
        displacements_mm = cluster_data.get('displacements_mm', np.array([]))
        
        max_disp_text = f"{np.max(displacements_mm):.2f}" if len(displacements_mm) > 0 else "0.00"
        mean_disp_text = f"{np.mean(displacements_mm):.2f}" if len(displacements_mm) > 0 else "0.00"
        
        current_text_actor = plotter.add_text(
            f"Frame: {frame_idx}/{frame_indices[-1]} | "
            f"Time: {frame_data['time']:.4f} | "
            f"Original Points: {frame_data['num_vertices']:,} | "
            f"Clusters: {cluster_data['num_clusters']} | "
            f"Max Disp: {max_disp_text} mm | "
            f"Mean Disp: {mean_disp_text} mm | "
            f"Scale Max: {max_disp_val:.2f} mm",
            font_size=12,
            position='upper_left'
        )
        
        # Force render update
        plotter.render()
    
    def slider_callback(value):
        """Callback for slider widget"""
        frame_list_idx = int(value)
        update_frame_display(frame_list_idx)
    
    def navigate_frame(direction):
        """Navigate to next or previous frame"""
        nonlocal current_frame_idx
        
        if direction == 'next':
            if current_frame_idx < len(frame_indices) - 1:
                new_idx = current_frame_idx + 1
                update_frame_display(new_idx)
                if len(plotter.slider_widgets) > 0:
                    plotter.slider_widgets[0].GetRepresentation().SetValue(new_idx)
        elif direction == 'prev':
            if current_frame_idx > 0:
                new_idx = current_frame_idx - 1
                update_frame_display(new_idx)
                if len(plotter.slider_widgets) > 0:
                    plotter.slider_widgets[0].GetRepresentation().SetValue(new_idx)
        elif direction == 'first':
            update_frame_display(0)
            if len(plotter.slider_widgets) > 0:
                plotter.slider_widgets[0].GetRepresentation().SetValue(0)
        elif direction == 'last':
            last_idx = len(frame_indices) - 1
            update_frame_display(last_idx)
            if len(plotter.slider_widgets) > 0:
                plotter.slider_widgets[0].GetRepresentation().SetValue(last_idx)
    
    # Add coordinate axes
    plotter.add_axes()
    
    # Add slider widget for frame navigation
    plotter.add_slider_widget(
        callback=slider_callback,
        rng=(0, len(frame_indices) - 1),
        value=0,
        title="Frame",
        pointa=(0.1, 0.02),
        pointb=(0.9, 0.02),
        style='modern',
        tube_width=0.005,
        slider_width=0.02,
        interaction_event='always'
    )
    
    # Initialize with first frame
    update_frame_display(0)
    
    # Set camera to fit ALL frames
    plotter.camera_position = 'xy'
    plotter.reset_camera(bounds=bounds)
    
    # Add instructions text
    instructions = (
        "Controls:\n"
        "  Slider (bottom): Drag to navigate frames\n"
        "  Arrow Keys: Left/Right to navigate frames\n"
        "  A/D keys: Previous/Next frame\n"
        "  Mouse: Rotate/zoom/pan view\n"
        "  Close window to exit\n"
        "\n"
        "Visualization:\n"
        "  Colored points: Cluster means colored by displacement\n"
        "  Blue = Low displacement, Red = High displacement"
    )
    plotter.add_text(instructions, font_size=9, position='lower_left')
    
    # Register keyboard events
    def on_next_frame():
        navigate_frame('next')
    
    def on_prev_frame():
        navigate_frame('prev')
    
    plotter.add_key_event('Right', on_next_frame)
    plotter.add_key_event('Left', on_prev_frame)
    plotter.add_key_event('d', on_next_frame)
    plotter.add_key_event('D', on_next_frame)
    plotter.add_key_event('a', on_prev_frame)
    plotter.add_key_event('A', on_prev_frame)
    
    print("\nInteractive visualization ready!")
    print("Use the slider or arrow keys (Left/Right) to navigate frames")
    print("Close the window to exit\n")
    
    # Show the interactive plotter
    plotter.show()


def run_fast_interactive_analysis(config_manager):
    """
    Load data and state, then run the interactive displacement visualization.
    
    Args:
        config_manager: Configuration manager instance
    """
    config = config_manager.config
    experiment_name = config['experiment']['name']
    output_dir = config['experiment']['output_dir']
    data_path = config['experiment']['data_path']
    
    print(f"\n=== INTERACTIVE DISPLACEMENT VISUALIZATION: {experiment_name} ===")
    
    # Load data
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print("Data file not found.")
        return
    
    # Load clustering state
    saved_means, _, _, _ = load_saved_clustering_state(experiment_name, output_dir)
    
    cluster_positions_timeseries = None
    
    # Check for cluster mesh export file
    if USE_CLUSTER_EXPORT:
        export_dir = os.path.join(output_dir, "cluster_mesh_export")
        
        # Try different possible filenames
        # Handle cases where experiment_name might include .json extension
        base_names = [experiment_name]
        if experiment_name.endswith('.json'):
            base_names.append(experiment_name[:-5])
            
        possible_filenames = []
        for name in base_names:
            possible_filenames.append(f"cluster_export_{name}.npy")
            possible_filenames.append(f"cluster_mesh_timeseries_{name}.npy")
        
        export_path = None
        for filename in possible_filenames:
            path = os.path.join(export_dir, filename)
            if os.path.exists(path):
                export_path = path
                break
        
        if export_path:
            print(f"\n✓ Found cluster mesh export: {export_path}")
            print("Loading pre-computed cluster positions...")
            export_data = np.load(export_path, allow_pickle=True).item()
            
            cluster_positions_timeseries = export_data["cluster_positions"]
            saved_means = export_data["initial_cluster_means"]
            
            print(f"  Loaded {cluster_positions_timeseries.shape[0]} frames")
            print(f"  Loaded {cluster_positions_timeseries.shape[1]} clusters")
        else:
            print(f"\nNo cluster mesh export found in: {export_dir}")
            print("Will compute cluster positions on the fly...")
    
    # Load raw data
    data_exists = os.path.exists(data_path)
    if data_exists:
        print(f"Loading raw data from: {data_path}")
        dataset = np.load(data_path)
        
        # Create frames data - check if dataset has RGB information
        if len(dataset.shape) == 3 and dataset.shape[2] == 6:
            # Has RGB data (X, Y, Z, R, G, B)
            frames_data = {}
            times = np.arange(dataset.shape[0]) * 0.1  # Default time step
            for i, frame in enumerate(dataset):
                coords = frame[:, :3]
                rgb = frame[:, 3:6]
                frames_data[i] = {
                    'coords': coords,
                    'rgb': rgb,
                    'time': times[i],
                    'num_vertices': len(coords)
                }
        else:
            # Use vessel_utils to extract frames
            frames_data = vu.get_all_frames_data(dataset)
    else:
        if cluster_positions_timeseries is not None:
            print(f"⚠️ Raw data NOT found at: {data_path}")
            print("But cluster export is present. Proceeding with export data only (no background points).")
            # Create dummy frames data based on export
            frames_data = {}
            for i in range(cluster_positions_timeseries.shape[0]):
                frames_data[i] = {
                    'coords': np.array([]).reshape(0, 3),
                    'rgb': None,
                    'time': i * 0.1,
                    'num_vertices': 0
                }
        else:
            print(f"❌ ERROR: Raw data NOT found at: {data_path}")
            print("And no cluster export available. Cannot proceed.")
            return

    if saved_means is None:
        print("⚠️  No saved clustering state found. Will use color-then-spatial clustering.")
    else:
        print(f"✓ Using {len(saved_means)} clusters from state/export")
        print(f"✓ Using {len(saved_means)} clusters from state/export")
    
    # Launch visualization
    create_interactive_visualization(frames_data, config_manager, saved_means, cluster_positions_timeseries)

