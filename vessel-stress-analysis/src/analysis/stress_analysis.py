"""
Stress analysis module using centralized configuration.
"""

# Flag to control whether to use cluster mesh export file
# If True: checks for export file and uses it if found (default behavior)
# If False: skips export file check and always computes cluster positions on the fly
USE_CLUSTER_EXPORT = True
USE_CONFIG_STRESS_SCALE = False
MAX_STRESS_SCALE_MIN = 0.0
MAX_STRESS_SCALE_MAX = 0.15e6

# --- Stress Visualization Threshold Settings ---
# Set to True to enable special visualization for stress below a certain threshold
ENABLE_THRESHOLD_VISUALIZATION = False
# Threshold in Pa (e.g., 0.05e6 = 50 kPa)
STRESS_THRESHOLD = 0.003e6
# How transparent edges below the threshold are (0 to 10)
# 0 = not transparent at all (fully visible), 10 = fully transparent (invisible)
LOW_STRESS_TRANSPARENCY = 9.0

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pyvista as pv
import time
import os
import datetime
from typing import List
import glob


from core.clustering import (
    color_then_spatial_clustering,
)
import utils.vessel_utils as vu

from utils.clustering_state_utils import (
    find_clustering_state_for_experiment,
    load_saved_clustering_state,
    remap_labels_to_saved_means
)

def plot_average_stress(all_stresses: List[np.ndarray], experiment_name: str, output_dir: str, total_experiment_time: float):
    """
    Plot average stress over time and save to experiment's plots directory.
    
    Args:
        all_stresses: List of stress arrays for each time point
        experiment_name: Name of the experiment for the plot title
        output_dir: Output directory for the experiment
        total_experiment_time: Total experiment time in seconds
    """
    # Compute the average stress for each time point (convert to MPa)
    average_stresses_mpa = [np.mean(stress) / 1e6 for stress in all_stresses]
    
    # Create time array in seconds
    num_time_points = len(average_stresses_mpa)
    time_points = np.linspace(0, total_experiment_time, num_time_points)
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot the average stress as a function of time
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, average_stresses_mpa, marker='o', linestyle='-', color='b', linewidth=3, markersize=8)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Average Stress (MPa)', fontsize=16)
    plt.title(f'Average Stress Over Time - {experiment_name}', fontsize=18, fontweight='bold')
    
    # Make tick labels larger
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot as SVG
    plot_filename = f'average_stress_{experiment_name}.svg'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    
    # Also save as PNG
    png_filename = f'average_stress_{experiment_name}.png'
    png_path = os.path.join(plots_dir, png_filename)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Average stress plot saved as '{plot_path}' and '{png_path}'")


# Colors and colormaps are now handled by vessel_utils (vu.get_colors)

def visualize_deformation(labels: np.ndarray, cluster_means: np.ndarray, 
                         cluster_points: List[np.ndarray], dataset: np.ndarray, 
                         config_manager, saved_edges=None, stress_min=None, stress_max=None,
                         cluster_positions_timeseries=None, initial_lengths=None) -> List[np.ndarray]:
    """
    Visualize deformation with customizable stress scale.
    
    Args:
        labels: Cluster labels
        cluster_means: Cluster mean positions
        cluster_points: Points in each cluster
        dataset: Time series data
        config_manager: Configuration manager instance
        saved_edges: Saved edge indices (optional)
        stress_min: Custom minimum stress value (optional)
        stress_max: Custom maximum stress value (optional)
        cluster_positions_timeseries: Pre-computed cluster positions (T, K, 3) array (optional)
        initial_lengths: Pre-computed initial edge lengths (E,) array (optional)
        
    Returns:
        List of stress values for each time frame
    """
    # Extract parameters from config
    edge_outlier_threshold = config_manager.get('clustering.edge_outlier_threshold', 0.25)
    save_render = config_manager.get('visualization.save_render', True)
    frame_delay = config_manager.get('visualization.frame_delay', 0.1)
    edge_line_width = config_manager.get('visualization.edge_line_width', 2.0)
    edge_opacity = config_manager.get('visualization.edge_opacity', 1.00)
    initial_edge_line_width = config_manager.get('visualization.initial_edge_line_width', 1)
    initial_point_size = config_manager.get('visualization.initial_point_size', 5)
    camera_position = config_manager.get('camera.position')
    young_modulus = config_manager.get('material_properties.young_modulus_silicone', vu.YOUNG_MODULUS_SILICON)
    
    # Create save path in the experiment's stress analysis folder
    if save_render:
        experiment_name = config_manager.config['experiment']['name']
        output_dir = config_manager.config['experiment']['output_dir']
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, "stress_analysis", f"renders_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        print(f"Renders will be saved to: {save_path}")
    
    plotter = pv.Plotter()

    # Determine if we're using pre-computed cluster positions
    use_precomputed = cluster_positions_timeseries is not None
    
    # Create and plot the Delaunay edges
    # Use saved edges if available (from loaded state), otherwise generate fresh Delaunay edges
    if saved_edges is not None and len(saved_edges) > 0:
        print(f"Using saved edges from clustering state ({len(saved_edges)} edges)")
        valid_edges = saved_edges.copy()
        vertices = cluster_means
        # Calculate initial lengths from saved edges (unless already provided)
        if initial_lengths is None:
            initial_lengths = np.linalg.norm(vertices[valid_edges[:, 0]] - vertices[valid_edges[:, 1]], axis=1)
    else:
        print(f"No saved edges, generating fresh Delaunay edges")
        vertices, edges_raw, _ = vu.create_delaunay_edges(cluster_means)
        valid_edges, removed_count = vu.remove_outlier_edges(vertices, edges_raw, edge_outlier_threshold)
        if initial_lengths is None:
            initial_lengths = np.linalg.norm(vertices[valid_edges[:, 0]] - vertices[valid_edges[:, 1]], axis=1)

    # Generate lines for the edges
    lines = np.hstack([np.full((valid_edges.shape[0], 1), 2), valid_edges])
    edge_mesh = pv.PolyData(vertices, lines=lines)

    plotter.add_mesh(edge_mesh, color='blue', line_width=initial_edge_line_width, style='surface')
    plotter.add_points(cluster_means, color='red', point_size=initial_point_size, render_points_as_spheres=True)

    # Set the camera position from configuration
    if camera_position:
        plotter.camera_position = camera_position
        print(f"Set camera position from configuration")
    else:
        print("No camera position found in configuration, using default")
    
    plotter.show(interactive_update=True)
    print("Visualization initiated.")

    cm = vu.get_colormap()

    all_stresses = []
    
    print("\nCalculating stresses over time...")
    
    if use_precomputed:
        # Use pre-computed cluster positions from export
        print("Using pre-computed cluster positions from cluster mesh export")
        num_frames = cluster_positions_timeseries.shape[0]
        total_frames = num_frames - 1
        
        for j in range(1, num_frames):
            # Progress bar
            progress = int((j / total_frames) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
            
            # Get pre-computed cluster positions for this frame
            new_cluster_means = cluster_positions_timeseries[j]
            
            # Calculate stress using Neo-Hookean model from vessel_utils
            stress = vu.compute_edge_stress(new_cluster_means, valid_edges, initial_lengths, young_modulus)
            all_stresses.append(stress)
            
            # Map stress to RGB colors (use vmin=0 for absolute stress)
            rgb_colors = vu.get_colors(stress, cm, vmin=0, vmax=stress_max, use_abs=True)
            # Add alpha channel
            colors_rgba = np.zeros((len(rgb_colors), 4))
            colors_rgba[:, :3] = rgb_colors

            # Get cluster positions for visualization
            viz_cluster_means = new_cluster_means
            
            plotter.clear()
            edge_mesh.points = viz_cluster_means
            
            # Apply threshold transparency if enabled
            if ENABLE_THRESHOLD_VISUALIZATION:
                # Calculate opacities based on threshold
                # 0 transparency (0/10) means multiplier 1.0 (fully opaque relative to base)
                # 10 transparency (10/10) means multiplier 0.0 (fully transparent)
                opacity_multiplier = (10.0 - LOW_STRESS_TRANSPARENCY) / 10.0
                
                # Start with base opacity for all edges
                opacities = np.full(len(stress), edge_opacity)
                # Apply reduction factor to edges below threshold
                opacities[stress < STRESS_THRESHOLD] *= opacity_multiplier
            else:
                opacities = np.full(len(stress), edge_opacity)

            # Apply opacity to the alpha channel of colors_rgba
            colors_rgba[:, 3] = opacities

            plotter.add_mesh(edge_mesh, rgb=True, scalars=colors_rgba, line_width=edge_line_width)
            plotter.add_text(f'Time {j}', font_size=12)

            if save_render:
                plotter.screenshot(os.path.join(save_path, f'frame_{j:04d}.png'))

            plotter.update()
            time.sleep(frame_delay)
    else:
        # Compute cluster positions on the fly (original behavior)
        total_frames = len(dataset) - 1
        
        # Track previous frame's cluster means for continuity
        prev_cluster_means = cluster_means.copy()

        for j in range(1, len(dataset)):
            # Progress bar
            progress = int((j / total_frames) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)

            new_frame = dataset[j].astype(np.float64)
            
            # Extract only XYZ coordinates from the new frame
            new_points = new_frame[:, :3]  # Only use first 3 columns (X, Y, Z)

            # CRITICAL FIX: Create new cluster means matching the shape of cluster_means
            # cluster_means has indices 0 to len(cluster_means)-1, so new_cluster_means must match
            # This ensures edge indices remain valid even if some labels don't exist in current frame
            new_cluster_means = np.zeros_like(cluster_means)
            
            for k in range(len(cluster_means)):
                cluster_mask = labels == k
                if np.any(cluster_mask):
                    new_cluster_means[k] = np.mean(new_points[cluster_mask], axis=0)
                else:
                    # If no points for this label in current frame, use previous frame's position
                    # (or initial position if first frame)
                    new_cluster_means[k] = prev_cluster_means[k]
            
            # Update previous cluster means for next iteration
            prev_cluster_means = new_cluster_means.copy()
            
            # Calculate stress using Neo-Hookean model from vessel_utils
            stress = vu.compute_edge_stress(new_cluster_means, valid_edges, initial_lengths, young_modulus)
            all_stresses.append(stress)
            
            # Map stress to RGB colors (use vmin=0 for absolute stress)
            rgb_colors = vu.get_colors(stress, cm, vmin=0, vmax=stress_max, use_abs=True)
            # Add alpha channel
            colors_rgba = np.zeros((len(rgb_colors), 4))
            colors_rgba[:, :3] = rgb_colors

            # Get cluster positions for visualization
            viz_cluster_means = new_cluster_means
            
            plotter.clear()
            edge_mesh.points = viz_cluster_means

            # Apply threshold transparency if enabled
            if ENABLE_THRESHOLD_VISUALIZATION:
                opacity_multiplier = (10.0 - LOW_STRESS_TRANSPARENCY) / 10.0
                opacities = np.full(len(stress), edge_opacity)
                opacities[stress < STRESS_THRESHOLD] *= opacity_multiplier
            else:
                opacities = np.full(len(stress), edge_opacity)
                
            # Apply opacity to the alpha channel of colors_rgba
            colors_rgba[:, 3] = opacities

            plotter.add_mesh(edge_mesh, rgb=True, scalars=colors_rgba, line_width=edge_line_width)
            plotter.add_text(f'Time {j}', font_size=12)

            if save_render:
                plotter.screenshot(os.path.join(save_path, f'frame_{j:04d}.png'))

            plotter.update()
            time.sleep(frame_delay)

    print()  # New line after progress bar
    plotter.close()
    print("Visualization complete.")

    return all_stresses

def run_stress_analysis(config_manager):
    """
    Run stress analysis using the provided configuration manager.
    
    Args:
        config_manager: Configuration manager instance
    """
    config = config_manager.config
    
    # Validate total experiment time early
    total_experiment_time = config['experiment'].get('total_time')
    if total_experiment_time is None or total_experiment_time <= 0:
        print("\n" + "="*60)
        print("⚠️  MISSING EXPERIMENT TIME!")
        print("="*60)
        print("The stress plot requires the total experiment time to be specified.")
        print("Please add 'total_time' to your experiment configuration:")
        print()
        print("experiment:")
        print("  name: your_experiment_name")
        print("  total_time: 10.0  # Total experiment time in seconds")
        print()
        print("Then run this analysis again.")
        print("="*60)
        return
    
    # Load data
    data_path = config['experiment'].get('data_path')
    # We will check if this exists later, as we might be using a cluster export instead
    
    # Get experiment info
    experiment_name = config['experiment']['name']
    output_dir = config['experiment']['output_dir']
    
    print(f"\nExperiment: {experiment_name}")
    
    # Pre-declare variables
    cluster_positions_timeseries = None
    initial_lengths_from_export = None
    filtered_edges = None
    cluster_means_from_export = None
    use_cluster_export = USE_CLUSTER_EXPORT
    
    if use_cluster_export:
        # Check for cluster mesh export file
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
            filtered_edges = export_data["edges"]
            initial_lengths_from_export = export_data["initial_lengths"]
            cluster_means_from_export = export_data["initial_cluster_means"]
            
            print(f"  Loaded {cluster_positions_timeseries.shape[0]} frames")
            print(f"  Loaded {cluster_positions_timeseries.shape[1]} clusters")
            print(f"  Loaded {len(filtered_edges)} edges")
            
            # Use cluster means from export
            cluster_means = cluster_means_from_export
        else:
            print(f"\nNo cluster mesh export found in: {export_dir}")
            print("Will compute cluster positions on the fly...")
            use_cluster_export = False
    
    # Check for existing clustering state if we're not using export
    state_path = find_clustering_state_for_experiment(experiment_name, output_dir)
    
    if not use_cluster_export and (state_path is None or not os.path.exists(state_path)):
        print("\n" + "="*60)
        print("⚠️  NO CLUSTERING STATE OR EXPORT FOUND!")
        print("="*60)
        print(f"This analysis requires either a cluster export or a clustering state file.")
        print(f"Experiment: {experiment_name}")
        print("Please run the mesh editor first to create a clustering state or ensure an export exists.")
        print("="*60)
        return

    # Load raw data
    data_exists = data_path and os.path.exists(data_path)
    if data_exists:
        print(f"Loading raw data from: {data_path}")
        dataset = np.load(data_path)
        print(f"Loaded data shape: {dataset.shape}")
        
        # Validate data format and extract XYZ and RGB coordinates
        if len(dataset.shape) == 3:
            frame = dataset[0]
        elif len(dataset.shape) == 2:
            frame = dataset
        else:
            print("Error: Dataset must be 2D (points, 6) or 3D (time_frames, points, 6)")
            return
            
        xyz = frame[:, :3]
        rgb = np.clip(frame[:, 3:], 0.0, 1.0)
        
        # Apply color-then-spatial clustering
        print("\nApplying color-then-spatial clustering...")
        labels, color_centroids = color_then_spatial_clustering(xyz, rgb, config_manager.config)
        
        # Calculate mean locations of clusters
        cluster_means, cluster_points = vu.calculate_cluster_means(xyz, labels)
        
        # Load clustering state to map raw clusters to edited state
        saved_final_means, saved_final_edges, _, _ = load_saved_clustering_state(experiment_name, output_dir)
        
        if saved_final_means is not None:
            print(f"Applied filtering: {len(cluster_means)} -> {len(saved_final_means)} cluster means")
            labels, cluster_points = remap_labels_to_saved_means(xyz, labels, cluster_means, saved_final_means)
            cluster_means = saved_final_means
            # Use state edges if we don't have export edges
            if filtered_edges is None:
                filtered_edges = saved_final_edges
        else:
            if not use_cluster_export:
                print("No saved state loaded, cannot proceed without clustering state")
                return
            else:
                print("No saved state loaded, but using cluster export for geometry.")
    else:
        if use_cluster_export:
            print(f"⚠️ Raw data NOT found at: {data_path}")
            print("But cluster export is present. Proceeding with export data only (no background points).")
            # Dummy variables for compatibility
            dataset = None
            labels = None
            cluster_points = None
            cluster_means = cluster_means_from_export
            # filtered_edges were already loaded from export
        else:
            print(f"❌ ERROR: Raw data NOT found at: {data_path}")
            print("And no cluster export available. Cannot proceed.")
            return

    # Ask user for stress scale customization
    print("\n=== STRESS SCALE CUSTOMIZATION ===")
    if USE_CONFIG_STRESS_SCALE:
        stress_scale_min = config_manager.get('visualization.stress_scale_min', 0.0)
        stress_scale_max = config_manager.get('visualization.stress_scale_max', vu.MAX_STRESS_PA)
    else:
        stress_scale_min = MAX_STRESS_SCALE_MIN
        stress_scale_max = MAX_STRESS_SCALE_MAX
    print(f"Current default stress scale: {stress_scale_min:.2e} to {stress_scale_max:.2e} Pa")
    
    # Visualize the deformation using Delaunay triangulation and calculate stress
    all_stresses = visualize_deformation(labels, cluster_means, cluster_points, dataset, 
                                       config_manager, saved_edges=filtered_edges, 
                                       stress_min=stress_scale_min, stress_max=stress_scale_max,
                                       cluster_positions_timeseries=cluster_positions_timeseries,
                                       initial_lengths=initial_lengths_from_export)
    
    # Save stress values
    output_file = config_manager.config['analysis']['stress']['output_file']
    output_path = config_manager.get_output_path('stress', output_file)
    np.save(output_path, all_stresses)
    print(f'Stress calculations saved to {output_path}') 

    # Plot average stress
    plot_average_stress(all_stresses, experiment_name, output_dir, total_experiment_time) 