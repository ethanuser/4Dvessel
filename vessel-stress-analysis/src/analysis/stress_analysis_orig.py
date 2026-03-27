"""
Stress analysis module using centralized configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pyvista as pv
import time
import os
import datetime
from typing import List
import glob

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================
# Camera behavior: if True, camera resets each frame to fit mesh
#                  if False, camera stays fixed in world space (allows seeing translation)
CAMERA_RESET_PER_FRAME = False
# ============================================================================

from core.clustering import (
    color_then_spatial_clustering,
    calculate_cluster_means,
    create_delaunay_edges,
    remove_outlier_edges
)
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


def generate_color_map(deformation: np.ndarray, colormap, config: dict):
    """
    Generate color map for stress visualization with customizable scale.
    
    Args:
        deformation: Stress values in Pa
        colormap: Matplotlib colormap
        config: Configuration dictionary
    """
    stress_min = config.get('visualization.stress_scale_min', -0.001e9)
    stress_max = config.get('visualization.stress_scale_max', 0.001e9)
    
    norm = plt.Normalize(vmin=stress_min, vmax=stress_max, clip=True)
    normalized_deformation = norm(deformation)
    colors = colormap(normalized_deformation)
    colors = colors[:, 0:3]
    return colors

def visualize_deformation(labels: np.ndarray, cluster_means: np.ndarray, 
                         cluster_points: List[np.ndarray], dataset: np.ndarray, 
                         config_manager, saved_edges=None, stress_min=None, stress_max=None) -> List[np.ndarray]:
    """
    Visualize deformation with customizable stress scale.
    
    Args:
        labels: Cluster labels
        cluster_means: Cluster mean positions
        cluster_points: Points in each cluster
        dataset: Time series data
        config_manager: Configuration manager instance
        stress_min: Custom minimum stress value (optional)
        stress_max: Custom maximum stress value (optional)
        
    Returns:
        List of stress values for each time frame
    """
    # Extract parameters from config
    edge_outlier_threshold = config_manager.get('clustering.edge_outlier_threshold', 0.25)
    save_render = config_manager.get('visualization.save_render', True)
    frame_delay = config_manager.get('visualization.frame_delay', 0.1)
    edge_line_width = config_manager.get('visualization.edge_line_width', 2.0)
    edge_opacity = config_manager.get('visualization.edge_opacity', 0.25)
    initial_edge_line_width = config_manager.get('visualization.initial_edge_line_width', 1)
    camera_position = config_manager.get('camera.position')
    young_modulus = config_manager.get('material_properties.young_modulus_silicone', 0.0255e9)
    
    # Create save path in the experiment's stress analysis folder
    if save_render:
        experiment_name = config_manager.config['experiment']['name']
        output_dir = config_manager.config['experiment']['output_dir']
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, "stress_analysis", f"renders_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        print(f"Renders will be saved to: {save_path}")
    
    plotter = pv.Plotter()

    # Create and plot the Delaunay edges
    # Use saved edges if available (from loaded state), otherwise generate fresh Delaunay edges
    if saved_edges is not None and len(saved_edges) > 0:
        print(f"Using saved edges from clustering state ({len(saved_edges)} edges)")
        valid_edges = saved_edges.copy()
        vertices = cluster_means
        # Calculate initial lengths from saved edges
        initial_lengths = np.linalg.norm(vertices[valid_edges[:, 0]] - vertices[valid_edges[:, 1]], axis=1)
    else:
        print(f"No saved edges, generating fresh Delaunay edges")
        vertices, edges, _ = create_delaunay_edges(cluster_means)
        valid_edges, initial_lengths = remove_outlier_edges(vertices, edges, config_manager.config)

    # Generate lines for the edges
    lines = np.hstack([np.full((valid_edges.shape[0], 1), 2), valid_edges])
    edge_mesh = pv.PolyData(vertices, lines=lines)

    # Store initial actors for updating (when not resetting camera)
    edge_actor = plotter.add_mesh(edge_mesh, color='blue', line_width=initial_edge_line_width, style='surface')
    text_actor = None

    # Calculate bounds from all frames to see bulk movement
    # Use raw dataset points directly - much faster than calculating cluster means
    print("\nCalculating bounds from all frames...")
    all_coords = np.vstack([dataset[i][:, :3] for i in range(len(dataset))])
    bounds = [
        all_coords[:, 0].min(), all_coords[:, 0].max(),
        all_coords[:, 1].min(), all_coords[:, 1].max(),
        all_coords[:, 2].min(), all_coords[:, 2].max()
    ]
    print(f"Bounds: X[{bounds[0]:.3f}, {bounds[1]:.3f}], "
          f"Y[{bounds[2]:.3f}, {bounds[3]:.3f}], Z[{bounds[4]:.3f}, {bounds[5]:.3f}]")

    # Print camera reset behavior
    if CAMERA_RESET_PER_FRAME:
        print("Camera will reset per frame (to fit mesh)")
    else:
        print("Camera position will be preserved between frames (allows seeing translation)")
    
    plotter.show(interactive_update=True)
    print("Visualization initiated.")
    
    # Set the camera position AFTER show() is called (like stress_interactive.py does)
    # This ensures PyVista doesn't override our bounds-based camera setting
    if camera_position:
        plotter.camera_position = camera_position
        # plotter.camera_position = 'xy'
        plotter.reset_camera(bounds=bounds)
        print(f"Set camera position from configuration")
    else:
        print("No camera position found in configuration, using default")
        # Set camera to fit all frames (like stress_interactive.py does)
        plotter.camera_position = 'xy'
        plotter.reset_camera(bounds=bounds)
        print("Set camera to fit all frames")
    
    # Initialize fixed camera variables (will be set if needed)
    fixed_camera_position = None
    fixed_camera_clipping_range = None
    
    # # Store initial camera state if we want to preserve it (after showing plotter and setting camera)
    # if not CAMERA_RESET_PER_FRAME:
    #     # Force a render to ensure camera is set
    #     plotter.render()
    #     # Update once to ensure everything is displayed
    #     # plotter.update()
    #     # Deep copy camera position (it's a tuple/list)
    #     cam_pos = plotter.camera_position
    #     if isinstance(cam_pos, (list, tuple)):
    #         fixed_camera_position = [list(pos) if isinstance(pos, (list, tuple)) else pos for pos in cam_pos]
    #     else:
    #         fixed_camera_position = cam_pos
    #     # Copy clipping range (it's a tuple)
    #     fixed_camera_clipping_range = tuple(plotter.camera.clipping_range)
    #     print("Camera position locked - will stay fixed in world space (allows seeing bulk movement)")

    colors_list = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # R -> Y -> G -> B -> G -> Y -> R
    cmap_name = 'stress_strain_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=1024)

    all_stresses = []
    
    print("\nCalculating stresses over time...")
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
        
        # Calculate new lengths and deformation
        new_lengths = np.linalg.norm(new_cluster_means[valid_edges[:, 0]] - new_cluster_means[valid_edges[:, 1]], axis=1)
        deformation = (new_lengths - initial_lengths)
        strain = deformation / initial_lengths  # Calculate strain
        stress = np.maximum(young_modulus * strain, 0)  # Only tensile stress, clip compression to 0
        all_stresses.append(stress)
        
        # Use custom stress scale if provided, otherwise use config defaults
        colors_rgba = generate_color_map(stress, cm, config_manager.config)

        if CAMERA_RESET_PER_FRAME:
            # Original behavior: clear and reset camera
            plotter.clear()
            edge_mesh.points = new_cluster_means
            plotter.add_mesh(edge_mesh, rgb=True, scalars=colors_rgba, line_width=edge_line_width, opacity=edge_opacity)
            plotter.add_text(f'Time {j}', font_size=12)
        else:
            # Preserve camera: update mesh points and actors without clearing
            # Update mesh points
            edge_mesh.points = new_cluster_means
            
            # Remove old actors
            if edge_actor is not None:
                try:
                    plotter.remove_actor(edge_actor)
                except Exception:
                    pass
            if text_actor is not None:
                try:
                    plotter.remove_actor(text_actor)
                except Exception:
                    pass
            
            # Add updated actors with reset_camera=False to prevent auto-adjustment
            edge_actor = plotter.add_mesh(edge_mesh, rgb=True, scalars=colors_rgba, line_width=edge_line_width, opacity=edge_opacity, reset_camera=False)
            text_actor = plotter.add_text(f'Time {j}', font_size=12)
            
            # # Force restore fixed camera position (in case PyVista adjusted it)
            # plotter.camera_position = fixed_camera_position
            # plotter.camera.clipping_range = fixed_camera_clipping_range

        if save_render:
            plotter.screenshot(os.path.join(save_path, f'frame_{j:04d}.png'))

        # Use render() instead of update() to avoid camera auto-adjustment
        plotter.render()

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
    data_path = config['experiment']['data_path']
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return
    
    # Get experiment info
    experiment_name = config['experiment']['name']
    output_dir = config['experiment']['output_dir']
    
    # Check for existing clustering state
    state_path = find_clustering_state_for_experiment(experiment_name, output_dir)
    
    if not os.path.exists(state_path):
        print("\n" + "="*60)
        print("⚠️  NO CLUSTERING STATE FOUND!")
        print("="*60)
        print(f"This analysis requires a clustering state file for experiment: {experiment_name}")
        print(f"Expected location: {state_path}")
        print("Please run the mesh editor first to create a clustering state:")
        print()
        print("  python scripts/run_mesh_editor.py")
        print()
        print("Then run this analysis again.")
        print("="*60)
        return
    
    print(f"\nFound clustering state: {state_path}")
    
    dataset = np.load(data_path)
    print(f"Loaded data shape: {dataset.shape}")
    
    # Validate data format and extract XYZ and RGB coordinates
    if len(dataset.shape) == 3:
        # Multi-time frame data
        frame = dataset[0]  # (N, 6) - first 3 columns are XYZ, last 3 are RGB
    elif len(dataset.shape) == 2:
        # Single time frame data
        frame = dataset
    else:
        print("Error: Dataset must be 2D (points, 6) or 3D (time_frames, points, 6)")
        return
        
    print(f"Original data shape: {frame.shape}")
    
    # Validate that we have 6 features (X, Y, Z, R, G, B)
    if frame.shape[1] != 6:
        print(f"Error: Expected 6 features (X, Y, Z, R, G, B), got {frame.shape[1]}")
        return
    
    # Extract both XYZ coordinates and RGB color information
    xyz = frame[:, :3]  # First 3 columns (X, Y, Z)
    rgb = np.clip(frame[:, 3:], 0.0, 1.0)  # Last 3 columns (R, G, B), clipped to [0,1]
    print(f"Using XYZ coordinates shape: {xyz.shape}")
    print(f"Using RGB color data shape: {rgb.shape}")
    
    # Apply color-then-spatial clustering
    print("\nApplying color-then-spatial clustering...")
    labels, color_centroids = color_then_spatial_clustering(xyz, rgb, config_manager.config)
    
    # Print clustering statistics
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    n_noise = np.sum(labels == -1)
    print(f"Clustering complete:")
    print(f"  - Total clusters: {n_clusters}")
    print(f"  - Noise points: {n_noise}")
    print(f"  - Clustered points: {len(labels) - n_noise}")
    
    # Calculate mean locations of clusters
    cluster_means, cluster_points = calculate_cluster_means(xyz, labels)
    print(f"Calculated {len(cluster_means)} cluster means")
    
    # Print some statistics about cluster sizes
    if len(cluster_points) > 0:
        cluster_sizes = [len(points) for points in cluster_points]
        print(f"Cluster sizes - Min: {min(cluster_sizes)}, Max: {max(cluster_sizes)}, Mean: {np.mean(cluster_sizes):.1f}")
    
    # Load clustering state (required) - use saved final state directly like mesh_editor does
    print("\nLoading clustering state...")
    
    # Use utility function to load saved state
    saved_final_means, saved_final_edges, saved_original_means, saved_original_edges = load_saved_clustering_state(
        experiment_name, output_dir
    )
    
    if saved_final_means is not None:
        print(f"Applied filtering: {len(cluster_means)} -> {len(saved_final_means)} cluster means")
        
        # Use utility function to remap labels to saved cluster means
        labels, cluster_points = remap_labels_to_saved_means(
            xyz, labels, cluster_means, saved_final_means
        )
        
        # Update cluster means to use the saved state
        cluster_means = saved_final_means
        filtered_edges = saved_final_edges
        
        print(f"New unique labels: {np.unique(labels[labels != -1])}")
        print(f"New cluster means shape: {cluster_means.shape}")
    else:
        print("No saved state loaded, using original clustering")
        filtered_edges = None
    
    # Ask user for stress scale customization
    print("\n=== STRESS SCALE CUSTOMIZATION ===")
    stress_scale_min = config_manager.get('visualization.stress_scale_min', -0.001e9)
    stress_scale_max = config_manager.get('visualization.stress_scale_max', 0.001e9)
    print(f"Current default stress scale: {stress_scale_min:.2e} to {stress_scale_max:.2e} Pa")
    
    # Visualize the deformation using Delaunay triangulation and calculate stress
    all_stresses = visualize_deformation(labels, cluster_means, cluster_points, dataset, 
                                       config_manager, saved_edges=filtered_edges, 
                                       stress_min=stress_scale_min, stress_max=stress_scale_max)
    
    # Save stress values
    output_file = config_manager.config['analysis']['stress']['output_file']
    output_path = config_manager.get_output_path('stress', output_file)
    np.save(output_path, all_stresses)
    print(f'Stress calculations saved to {output_path}') 

    # Plot average stress
    plot_average_stress(all_stresses, experiment_name, output_dir, total_experiment_time) 