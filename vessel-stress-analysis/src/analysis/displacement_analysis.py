"""
Displacement analysis module using centralized configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from typing import List
import pyvista as pv
import time
import glob

# Flag to control whether to use cluster mesh export file
# If True: checks for export file and uses it if found (default behavior)
# If False: skips export file check and always computes cluster positions on the fly
USE_CLUSTER_EXPORT = True

from core.clustering import (
    color_then_spatial_clustering,
)
import utils.vessel_utils as vu
from utils.clustering_state_utils import (
    find_clustering_state_for_experiment,
    load_saved_clustering_state,
    remap_labels_to_saved_means
)

# ── Physical-scale constants ──────────────────────────────────────────────
CHESS_TOTAL_LENGTH_M    = 6.195 / 100        # length across 8 squares (metres)
CHESS_SQUARES_ALONG_X   = 8              # number of squares measured
CHESS_SQUARE_LENGTH_M   = CHESS_TOTAL_LENGTH_M / CHESS_SQUARES_ALONG_X  # ~0.774375 cm per square
CALIB_SCALING_FACTOR    = 0.667125       # same factor used in calibration objp (units per square)
CORRECTION_FACTOR       = 6.195 / 5.053 # ratio of the length of the chessboard in real life to the length of the chessboard we've calculated
UNIT_TO_MM              = (CHESS_SQUARE_LENGTH_M / CALIB_SCALING_FACTOR) * 1000 * CORRECTION_FACTOR  # Convert to mm

print("Scaling Factor: ", str(UNIT_TO_MM))
#  → multiply any 4DGS distance by UNIT_TO_MM to get millimetres

# ── CONFIGURABLE CONSTANTS ───────────────────────────────────────────────────
PLOT_TITLE_PREFIX = "Average Displacement of"
PLOT_TITLE_SUFFIX = "vs. Time"
SAVE_DIRECTORY = "displacement_region_analysis"
OUTPUT_FILE_PREFIX = "calculated_displacements"

def interactive_point_selection(cluster_means, camera_position):
    """
    Interactive point selection using PyVista.
    
    Args:
        cluster_means: Initial cluster mean positions
        camera_position: Initial camera position
        
    Returns:
        Selected point indices and initial positions
    """
    plotter = pv.Plotter()

    # Add points
    plotter.add_points(cluster_means, color='red', point_size=5, render_points_as_spheres=True)

    # Set initial camera position
    plotter.camera_position = camera_position

    # State variables
    selected = {
        'indices': [],   # list of chosen cluster indices
        'p0': None,      # (num_selected, 3) array of initial positions
        'disp': []       # list of per-point displacement lists
    }
    selection_complete = {'done': False}

    def _picked_point(point):
        """Add a clicked cluster centroid to the selection list."""
        dists = np.linalg.norm(cluster_means - point, axis=1)
        sel_idx = int(dists.argmin())
        if sel_idx in selected['indices']:
            return  # ignore duplicates
        selected['indices'].append(sel_idx)
        print(f"Selected cluster {sel_idx} @ {cluster_means[sel_idx]}")
        # magenta, large, always visible
        plotter.add_points(cluster_means[sel_idx][None, :],
                        color='magenta',
                        point_size=58,
                        render_points_as_spheres=True,
                        name=f"picked_{sel_idx}",
                        opacity=1.0,
                        lighting=False,
                        pickable=False)

    def _confirm():
        """Called when user presses Enter/Return"""
        selection_complete['done'] = True
        if selected['indices']:
            selected['p0'] = cluster_means[selected['indices']].copy()
            selected['disp'] = [[] for _ in selected['indices']]
            print("Starting playback with points:", selected['indices'])
        else:
            print("No points selected – starting playback anyway.")
        plotter.close()

    # Enable point picking
    plotter.enable_point_picking(callback=_picked_point, show_message=True)
    
    # Add instruction text
    instruction = plotter.add_text(
        "Click centroids (they will turn magenta). Press ENTER to start playback.",
        position='upper_left', font_size=12
    )

    # Add key events
    plotter.add_key_event('Return', _confirm)

    print("=== INTERACTIVE POINT SELECTION ===")
    print("Click cluster means to select them for displacement tracking")
    print("Selected points will turn magenta")
    print("Press ENTER when done selecting points")

    # Show the plotter and wait for user interaction
    plotter.show(interactive_update=True)
    print("Visualization initiated.")

    # Wait for user to complete point selection
    while not selection_complete['done']:
        plotter.update()
        time.sleep(0.05)

    return selected

def plot_displacement_over_time(displacement_data: List[List[float]], region_name: str, experiment_name: str, 
                               output_dir: str, total_experiment_time: float):
    """
    Plot displacement over time with publication-quality formatting and save to experiment's plots directory.
    
    Args:
        displacement_data: List of displacement lists for each tracked point
        region_name: Name of the selected region
        experiment_name: Name of the experiment for the plot title
        output_dir: Output directory for the experiment
        total_experiment_time: Total experiment time in seconds
    """
    if not displacement_data or not displacement_data[0]:
        print("No displacement data to plot")
        return
    
    # Convert to numpy array and handle NaN values
    disp_array = np.array(displacement_data)
    num_points, num_frames = disp_array.shape
    
    # Create time array in seconds
    time_points = np.linspace(0, total_experiment_time, num_frames)
    
    # Calculate mean across points for each frame (no std for error bars)
    mean_disp = np.nanmean(disp_array, axis=0)
    
    # Calculate and print the maximum average displacement
    max_avg_displacement = np.nanmax(mean_disp)
    print(f"Maximum average displacement: {max_avg_displacement:.4f} mm")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create title using constants
    plot_title = f'{PLOT_TITLE_PREFIX} {region_name} {PLOT_TITLE_SUFFIX}'
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure with publication-quality dimensions and DPI
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Plot the displacement as a function of time (no error bars)
    ax.plot(time_points, mean_disp, marker='o', linestyle='-', color='#1f77b4', 
            linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
    
    # Customize axes
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='normal')
    ax.set_ylabel('Displacement (mm)', fontsize=12, fontweight='normal')
    ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=20)
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=1, length=6)
    ax.tick_params(axis='both', which='minor', width=0.5, length=3)
    
    # Add minor grid lines for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    
    # Set axis limits with some padding
    ax.set_xlim(-total_experiment_time * 0.02, total_experiment_time * 1.02)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom spines slightly thicker
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create a clean filename from the region name
    clean_region_name = region_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    clean_region_name = ''.join(c for c in clean_region_name if c.isalnum() or c == '_')
    
    # Save the plot as SVG and PNG
    plot_filename = f'average_displacement_{experiment_name}_{clean_region_name}.svg'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, format='svg', bbox_inches='tight', dpi=300)
    
    # Also save as PNG
    png_filename = f'average_displacement_{experiment_name}_{clean_region_name}.png'
    png_path = os.path.join(plots_dir, png_filename)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"Publication-quality displacement plot saved as:")
    print(f"  - SVG: {plot_path}")
    print(f"  - PNG: {png_path}")

def run_displacement_analysis(config_manager):
    """
    Run displacement analysis using the provided configuration manager.
    
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
        print("The displacement plot requires the total experiment time to be specified.")
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
    cluster_means, cluster_points = vu.calculate_cluster_means(xyz, labels)
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
        # Note: displacement_analysis doesn't use edges, but we load them for consistency
        
        print(f"New unique labels: {np.unique(labels[labels != -1])}")
        print(f"New cluster means shape: {cluster_means.shape}")
    else:
        print("No saved state loaded, cannot proceed without clustering state")
        return
    
    # Check if we should look for cluster mesh export file
    # If USE_CLUSTER_EXPORT is False, skip the export file check entirely
    use_cluster_export = USE_CLUSTER_EXPORT
    cluster_positions_timeseries = None
    
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
            cluster_means_from_export = export_data["initial_cluster_means"]
            
            print(f"  Loaded {cluster_positions_timeseries.shape[0]} frames")
            print(f"  Loaded {cluster_positions_timeseries.shape[1]} clusters")
            
            # Use cluster means from export
            cluster_means = cluster_means_from_export
            
            # Still need labels for compatibility (interactive_point_selection expects them to match)
            # but we've already remapped them to saved_final_means which should match export
        else:
            print(f"\nNo cluster mesh export found in: {export_dir}")
            print("Will compute cluster positions on the fly...")
            use_cluster_export = False  # Fall back to computing on the fly
    else:
        print(f"\nCluster export check disabled (use_cluster_export=False)")
        print("Will compute cluster positions on the fly...")
    
    # Get camera position from config
    camera_position = config_manager.config.get('camera', {}).get('position')
    if not camera_position:
        print("Warning: No camera position found in config. Using default camera position.")
        camera_position = [(-0.31676350915969265, -17.727903759830316, -5.178437900637502),
                          (0.23947606203271332, 0.26414248443613014, -1.2407439169630974),
                          (-0.9948541431039188, 0.050037906334993124, -0.08809904584374163)]
    
    # Interactive point selection
    selected = interactive_point_selection(cluster_means, camera_position)
    
    # Calculate displacement for each time frame
    print("\nCalculating displacements...")
    
    total_frames = len(dataset) - 1
    
    if use_cluster_export:
        print("Using pre-computed cluster positions from cluster mesh export")
        num_frames = cluster_positions_timeseries.shape[0]
        
        for j in range(1, num_frames):
            # Progress bar
            progress = int((j / total_frames) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
            
            # Get pre-computed cluster positions for this frame
            new_cluster_means = cluster_positions_timeseries[j]
            
            # Calculate displacement for selected points
            if selected['indices']:
                for idx_pos, idx in enumerate(selected['indices']):
                    try:
                        current = new_cluster_means[idx]
                        if np.any(np.isnan(current)):
                            raise ValueError("NaN")
                        # Calculate displacement magnitude in mm
                        disp_val = float(np.linalg.norm(current - selected['p0'][idx_pos])) * UNIT_TO_MM
                        selected['disp'][idx_pos].append(disp_val)
                    except (IndexError, ValueError):
                        # Stop tracking this point if it's lost
                        selected['disp'][idx_pos].append(np.nan)
    else:
        # Track previous frame's cluster means for continuity
        prev_cluster_means = cluster_means.copy()
        
        for j in range(1, len(dataset)):
            # Progress bar
            progress = int((j / total_frames) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
            
            # Get new frame data
            new_frame = dataset[j].astype(np.float64)
            new_points = new_frame[:, :3]  # Only use first 3 columns (X, Y, Z)
            
            # Update cluster means matching the shape of cluster_means
            # This ensures indices remain valid even if some labels don't exist in current frame
            new_cluster_means = np.zeros_like(cluster_means)
            
            for k in range(len(cluster_means)):
                cluster_mask = labels == k
                if np.any(cluster_mask):
                    new_cluster_means[k] = np.mean(new_points[cluster_mask], axis=0)
                else:
                    # If no points for this label in current frame, use previous frame's position
                    new_cluster_means[k] = prev_cluster_means[k]
            
            # Update previous cluster means for next iteration
            prev_cluster_means = new_cluster_means.copy()
            
            # Calculate displacement for selected points
            if selected['indices']:
                for idx_pos, idx in enumerate(selected['indices']):
                    try:
                        current = new_cluster_means[idx]
                        if np.any(np.isnan(current)):
                            raise ValueError("NaN")
                        # Calculate displacement magnitude in mm
                        disp_val = float(np.linalg.norm(current - selected['p0'][idx_pos])) * UNIT_TO_MM
                        selected['disp'][idx_pos].append(disp_val)
                    except (IndexError, ValueError):
                        # Stop tracking this point if it's lost
                        selected['disp'][idx_pos].append(np.nan)
                        print(f"Lost point idx {idx} at frame {j}")
    
    print()  # New line after progress bar
    print(f"Displacement calculation complete!")
    
    # Prompt user for region name
    print("\n" + "="*50)
    print("📝 REGION NAME PROMPT")
    print("="*50)
    print("Enter a name for the selected region.")
    print("This will be used in the plot title and filename.")
    print("Examples: 'M1 Convex', 'M1/M2 Bifurcation', 'M2 Caudal'")
    print("="*50)
    
    region_name = input("Enter region name: ").strip()
    
    if not region_name:
        region_name = "Selected Region"
        print(f"Using default region name: '{region_name}'")
    else:
        print(f"Using region name: '{region_name}'")
    
    # Create output directory for region analysis
    region_output_dir = os.path.join(output_dir, SAVE_DIRECTORY)
    os.makedirs(region_output_dir, exist_ok=True)
    
    # Save displacement values
    clean_region_name = region_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    clean_region_name = ''.join(c for c in clean_region_name if c.isalnum() or c == '_')
    output_filename = f"{OUTPUT_FILE_PREFIX}_{clean_region_name}.npy"
    output_path = os.path.join(region_output_dir, output_filename)
    np.save(output_path, np.array(selected['disp']))
    print(f'Displacement calculations saved to {output_path}')
    
    # Also save selected indices
    indices_filename = f"{OUTPUT_FILE_PREFIX}_{clean_region_name}_indices.npy"
    indices_path = os.path.join(region_output_dir, indices_filename)
    np.save(indices_path, np.array(selected['indices'], dtype=int))
    print(f'Tracked indices saved to {indices_path}')
    
    # Plot displacement over time
    if selected['disp']:
        plot_displacement_over_time(selected['disp'], region_name, experiment_name, output_dir, total_experiment_time)
 