"""
Region-specific stress analysis module using centralized configuration.
This module allows users to select regions using rectangle selection and analyze
average stresses of edges connected to vertices in that region.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import time
import os
import datetime
from typing import List, Tuple, Optional
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
from utils.rectangle_select_points import select_points_with_rectangle
from matplotlib.colors import LinearSegmentedColormap

# ── CONFIGURABLE CONSTANTS ───────────────────────────────────────────────────
PLOT_TITLE_PREFIX = "Average Stress of"
PLOT_TITLE_SUFFIX = "vs. Time"
SAVE_DIRECTORY = "stress_region_analysis"
OUTPUT_FILE_PREFIX = "calculated_stresses"
# Note: Young's modulus and edge outlier threshold are now read from config_manager
# to ensure consistency with stress_analysis.py

# Colors and colormaps are now handled by vessel_utils (vu.get_colors)

def plot_average_stress_region(all_stresses: List[np.ndarray], region_name: str, experiment_name: str, 
                             output_dir: str, total_experiment_time: float):
    """
    Plot average stress over time for a specific region and save to experiment's plots directory.
    
    Args:
        all_stresses: List of stress arrays for each time point
        region_name: Name of the selected region
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
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure with publication-quality dimensions and DPI
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Plot the average stress as a function of time
    ax.plot(time_points, average_stresses_mpa, marker='o', linestyle='-', color='#d62728', 
            linewidth=2, markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    
    # Customize axes
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='normal')
    ax.set_ylabel('Average Stress (MPa)', fontsize=12, fontweight='normal')
    ax.set_title(f'{PLOT_TITLE_PREFIX} {region_name} {PLOT_TITLE_SUFFIX}', fontsize=14, fontweight='bold', pad=20)
    
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
    plot_filename = f'average_stress_{experiment_name}_{clean_region_name}.svg'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, format='svg', bbox_inches='tight', dpi=300)
    
    # Also save as PNG
    png_filename = f'average_stress_{experiment_name}_{clean_region_name}.png'
    png_path = os.path.join(plots_dir, png_filename)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"Publication-quality stress plot saved as:")
    print(f"  - SVG: {plot_path}")
    print(f"  - PNG: {png_path}")

def interactive_multi_selection(cluster_means, camera_position, point_selection_threshold=0.5) -> List[int]:
    """
    Interactive multi-mode selection allowing both clicking and rectangle selection.
    Similar to mesh_editor.py but focused on accumulating selections.
    
    Args:
        cluster_means: Cluster mean positions
        camera_position: Initial camera position
        point_selection_threshold: Distance threshold for point picking
        
    Returns:
        List of selected vertex indices
    """
    print("\n" + "="*60)
    print("=== INTERACTIVE MULTI-MODE SELECTION ===")
    print("="*60)
    print("CONTROLS:")
    print("  • Click near points: Select/deselect individual vertices")
    print("  • Press 'R': Rectangle selection (add multiple points)")
    print("  • Press 'C': Clear all selections")
    print("  • Press 'Q' or 'Enter': Mark complete (then close window)")
    print("  • Close window (X button): Finish and continue with analysis")
    print("  • Mouse: Rotate, zoom, and pan the view")
    print("  • Selected points: Highlighted in GREEN with larger size")
    print("="*60)
    print("TIP: Press 'Q' when happy with selection, then close the window")
    print("="*60 + "\n")
    
    plotter = pv.Plotter(title="Region Selection - Close window when done")
    
    # Track selected points using indices directly (more reliable)
    selected_indices_set = set()
    
    # Track if user indicated they're done (but don't close programmatically)
    selection_complete = {'done': False}
    
    # Add all cluster means as red points
    plotter.add_points(cluster_means, color='red', point_size=5, 
                      render_points_as_spheres=True, name='cluster_means')
    
    # Set camera position
    if camera_position:
        plotter.camera_position = camera_position
    
    def highlight_selected_points():
        """Update visualization to show selected points"""
        # Remove old highlights
        try:
            plotter.remove_actor('selected_points_highlight')
        except Exception:
            pass
        
        # Add new highlights
        if selected_indices_set:
            selected_coords = cluster_means[list(selected_indices_set)]
            plotter.add_points(selected_coords, color='green', point_size=12, 
                             render_points_as_spheres=True, 
                             name='selected_points_highlight', 
                             opacity=1.0, lighting=False)
        
        # Update window title with selection count
        if selected_indices_set:
            plotter.title = f"Region Selection - {len(selected_indices_set)} points selected - Close window when done"
    
    def on_pick_point(point):
        """Callback for point picking - select/deselect individual points"""
        if point is None or len(point) == 0:
            return
        
        # Find closest cluster mean
        if len(cluster_means) > 0:
            dists = np.linalg.norm(cluster_means - point, axis=1)
            min_dist_idx = int(np.argmin(dists))
            min_dist = dists[min_dist_idx]
            
            if min_dist < point_selection_threshold:
                # Toggle selection using index directly
                if min_dist_idx in selected_indices_set:
                    selected_indices_set.remove(min_dist_idx)
                    print(f"Deselected point {min_dist_idx}. Total selected: {len(selected_indices_set)}")
                else:
                    selected_indices_set.add(min_dist_idx)
                    print(f"Selected point {min_dist_idx}. Total selected: {len(selected_indices_set)}")
                
                highlight_selected_points()
            else:
                print(f"No point near click: closest distance={min_dist:.3f} (threshold={point_selection_threshold})")
    
    def on_rectangle_select():
        """Use rectangle selection to add multiple points"""
        print("\n--- Starting Rectangle Selection ---")
        
        # Get current camera position
        camera_pos = plotter.camera_position
    
    # Use the utility function for rectangle selection
        new_selected_indices = select_points_with_rectangle(
        points=cluster_means,
            camera_position=camera_pos,
        point_size=5,
        point_color='red',
        selected_color='green',
            selected_size=12
        )
        
        if new_selected_indices:
            # Add to existing selection
            for idx in new_selected_indices:
                selected_indices_set.add(idx)
            
            print(f"Added {len(new_selected_indices)} points. Total selected: {len(selected_indices_set)}")
            
            # Update main visualization
            highlight_selected_points()
        else:
            print("No points selected in rectangle")
        
        print("--- Rectangle Selection Complete ---\n")
    
    def on_clear_selection():
        """Clear all selected points"""
        selected_indices_set.clear()
        highlight_selected_points()
        print(f"Selection cleared. Total selected: 0")
    
    def on_finish_selection():
        """Mark selection as complete - user must close window manually"""
        print(f"\n{'='*60}")
        print(f"✓ Selection marked complete: {len(selected_indices_set)} points selected")
        print(f"{'='*60}")
        print("CLOSE THE VISUALIZATION WINDOW to continue")
        print("(Click the X button or press ALT+F4)")
        print(f"{'='*60}\n")
        selection_complete['done'] = True
        plotter.title = f"✓ DONE - {len(selected_indices_set)} points selected - CLOSE THIS WINDOW"
    
    # Enable point picking
    plotter.enable_point_picking(callback=on_pick_point, show_message=False)
    
    # Add key events
    plotter.add_key_event('r', on_rectangle_select)
    plotter.add_key_event('R', on_rectangle_select)
    plotter.add_key_event('c', on_clear_selection)
    plotter.add_key_event('C', on_clear_selection)
    plotter.add_key_event('q', on_finish_selection)
    plotter.add_key_event('Q', on_finish_selection)
    plotter.add_key_event('Return', on_finish_selection)
    
    print("[DEBUG] About to call plotter.show() - this is a blocking call")
    print("Close the visualization window when you're done selecting points\n")
    
    # Show the plotter (blocking call - returns when window is closed)
    plotter.show()
    
    print(f"\n[DEBUG] plotter.show() returned! Window was closed.")
    print(f"[DEBUG] selected_indices_set type: {type(selected_indices_set)}")
    print(f"[DEBUG] selected_indices_set length: {len(selected_indices_set)}")
    print(f"[DEBUG] selection_complete['done']: {selection_complete['done']}")
    
    # Convert set to sorted list
    result = sorted(list(selected_indices_set))
    print(f"[DEBUG] Returning {len(result)} selected indices")
    if len(result) > 0:
        print(f"[DEBUG] First few indices: {result[:min(5, len(result))]}")
    return result

def visualize_region_deformation(labels: np.ndarray, cluster_means: np.ndarray,
                                dataset: np.ndarray, selected_vertex_indices: List[int],
                                region_edges: np.ndarray, region_initial_lengths: np.ndarray,
                                region_name: str, config_manager, all_stresses: List[np.ndarray],
                                all_edges: np.ndarray, cluster_positions_timeseries=None) -> None:
    """
    Visualize deformation for the selected region with stress color coding.
    
    Args:
        labels: Cluster labels
        cluster_means: All cluster mean positions
        dataset: Time series data
        selected_vertex_indices: Indices of selected vertices
        region_edges: Edges in the selected region
        region_initial_lengths: Initial lengths of region edges
        region_name: Name of the region
        config_manager: Configuration manager instance
        all_stresses: Pre-calculated stress values for each frame
        all_edges: All edges in the mesh (for context)
        cluster_positions_timeseries: Pre-computed cluster positions (T, K, 3) (optional)
    """
    # Extract visualization parameters from config
    # Force save_render to False for region analysis
    save_render = False
    frame_delay = config_manager.get('visualization.frame_delay', 0.1)
    edge_line_width = config_manager.get('visualization.edge_line_width', 3.0)
    edge_opacity = config_manager.get('visualization.edge_opacity', 0.8)
    non_region_edge_opacity = 0.15  # Much more transparent for non-analyzed edges
    non_region_edge_width = 1.0  # Thinner lines for non-analyzed edges
    initial_edge_line_width = config_manager.get('visualization.initial_edge_line_width', 2)
    initial_point_size = config_manager.get('visualization.initial_point_size', 8)
    selected_point_size = config_manager.get('visualization.selected_point_size', 12)
    camera_position = config_manager.get('camera.position')
    
    plotter = pv.Plotter()
    
    # Generate lines for ALL edges (for context, shown dimmed)
    all_lines = np.hstack([np.full((all_edges.shape[0], 1), 2), all_edges])
    all_edge_mesh = pv.PolyData(cluster_means, lines=all_lines)
    
    # Generate lines for the region edges (will be shown with stress colors)
    region_lines = np.hstack([np.full((region_edges.shape[0], 1), 2), region_edges])
    region_edge_mesh = pv.PolyData(cluster_means, lines=region_lines)
    
    # Add ALL edges first (dimmed, for context)
    plotter.add_mesh(all_edge_mesh, color='gray', line_width=non_region_edge_width, 
                    opacity=non_region_edge_opacity, style='surface', name='all_edges')
    
    # Add the region edges on top (initially blue)
    plotter.add_mesh(region_edge_mesh, color='blue', line_width=initial_edge_line_width, 
                    style='surface', name='region_edges')
    
    # Add all cluster means in red (dimmed)
    plotter.add_points(cluster_means, color='red', point_size=initial_point_size, 
                      render_points_as_spheres=True, opacity=0.3, name='all_points')
    
    # Highlight selected vertices in green
    selected_positions = cluster_means[selected_vertex_indices]
    plotter.add_points(selected_positions, color='green', point_size=selected_point_size,
                      render_points_as_spheres=True, name='selected_points')
    
    # Set camera position
    if camera_position:
        plotter.camera_position = camera_position
        print(f"Set camera position from configuration")
    else:
        print("No camera position found in configuration, using default")
    
    plotter.show(interactive_update=True)
    print(f"Visualization initiated for region: {region_name}")
    
    cm = vu.get_colormap()
    
    print("\nVisualizing stress over time...")
    
    # Determine if we're using pre-computed cluster positions
    use_precomputed = cluster_positions_timeseries is not None
    
    if use_precomputed:
        num_frames = cluster_positions_timeseries.shape[0]
        total_frames = num_frames - 1
        
        for j in range(1, num_frames):
            # Progress bar
            progress = int((j / total_frames) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
            
            # Use pre-computed cluster positions
            new_cluster_means = cluster_positions_timeseries[j]
            
            # Get pre-calculated stress for this frame
            stress = all_stresses[j-1]
            
            stress_min = config_manager.get('visualization.stress_scale_min', 0)
            stress_max = config_manager.get('visualization.stress_scale_max', vu.MAX_STRESS_PA)
            
            colors_rgba = vu.get_colors(stress, cm, vmin=stress_min, vmax=stress_max, use_abs=True)
            
            # Update visualization
            plotter.clear()
            
            all_edge_mesh.points = new_cluster_means
            plotter.add_mesh(all_edge_mesh, color='gray', line_width=non_region_edge_width,
                            opacity=non_region_edge_opacity, style='surface', name='all_edges')
            
            region_edge_mesh.points = new_cluster_means
            plotter.add_mesh(region_edge_mesh, rgb=True, scalars=colors_rgba, 
                            line_width=edge_line_width, opacity=edge_opacity, name='region_edges')
            
            plotter.add_points(new_cluster_means, color='red', point_size=initial_point_size,
                              render_points_as_spheres=True, opacity=0.3, name='all_points')
            selected_positions = new_cluster_means[selected_vertex_indices]
            plotter.add_points(selected_positions, color='green', point_size=selected_point_size,
                              render_points_as_spheres=True, name='selected_points')
            
            avg_stress_mpa = np.mean(stress) / 1e6
            plotter.add_text(f'{region_name} - Frame {j}/{total_frames} - Avg Stress: {avg_stress_mpa:.4f} MPa', 
                            font_size=12, name='info_text')
            
            plotter.update()
            time.sleep(frame_delay)
    else:
        total_frames = len(dataset) - 1
        
        # Track previous frame's cluster means for continuity
        prev_cluster_means = cluster_means.copy()
        
        for j in range(1, len(dataset)):
            # Progress bar
            progress = int((j / total_frames) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
            
            new_frame = dataset[j].astype(np.float64)
            new_points = new_frame[:, :3]  # Only use first 3 columns (X, Y, Z)
            
            # Update cluster means matching the shape of cluster_means
            new_cluster_means = np.zeros_like(cluster_means)
            
            for k in range(len(cluster_means)):
                cluster_mask = labels == k
                if np.any(cluster_mask):
                    new_cluster_means[k] = np.mean(new_points[cluster_mask], axis=0)
                else:
                    new_cluster_means[k] = prev_cluster_means[k]
            
            # Update previous cluster means for next iteration
            prev_cluster_means = new_cluster_means.copy()
            
            # Get pre-calculated stress for this frame
            stress = all_stresses[j-1]  # j-1 because all_stresses starts from frame 1
            
            stress_min = config_manager.get('visualization.stress_scale_min', 0)
            stress_max = config_manager.get('visualization.stress_scale_max', vu.MAX_STRESS_PA)
            
            # Generate colors based on stress (use vmin=0 for absolute stress)
            colors_rgba = vu.get_colors(stress, cm, vmin=0, vmax=stress_max, use_abs=True)
            
            # Update visualization
            plotter.clear()
            
            # Update all edges (dimmed, for context)
            all_edge_mesh.points = new_cluster_means
            plotter.add_mesh(all_edge_mesh, color='gray', line_width=non_region_edge_width,
                            opacity=non_region_edge_opacity, style='surface', name='all_edges')
            
            # Update region edges with stress colors
            region_edge_mesh.points = new_cluster_means
            plotter.add_mesh(region_edge_mesh, rgb=True, scalars=colors_rgba, 
                            line_width=edge_line_width, opacity=edge_opacity, name='region_edges')
            
            # Update points
            plotter.add_points(new_cluster_means, color='red', point_size=initial_point_size,
                              render_points_as_spheres=True, opacity=0.3, name='all_points')
            selected_positions = new_cluster_means[selected_vertex_indices]
            plotter.add_points(selected_positions, color='green', point_size=selected_point_size,
                              render_points_as_spheres=True, name='selected_points')
            
            # Add text with current frame and average stress
            avg_stress_mpa = np.mean(stress) / 1e6
            plotter.add_text(f'{region_name} - Frame {j}/{total_frames} - Avg Stress: {avg_stress_mpa:.4f} MPa', 
                            font_size=12, name='info_text')
            
            plotter.update()
            time.sleep(frame_delay)
    
    print()  # New line after progress bar
    plotter.close()
    print("Visualization complete.")

def calculate_region_stresses(labels: np.ndarray, cluster_means: np.ndarray, 
                            dataset: np.ndarray, selected_vertex_indices: List[int], 
                            config_manager, saved_edges=None,
                            cluster_positions_timeseries=None, initial_lengths_from_export=None) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Calculate stresses for edges connected to selected vertices.
    
    Args:
        labels: Cluster labels
        cluster_means: Cluster mean positions
        dataset: Time series data
        selected_vertex_indices: Indices of selected vertices
        config_manager: Configuration manager instance
        saved_edges: Saved edges from clustering state (optional)
        cluster_positions_timeseries: Pre-computed cluster positions (optional)
        initial_lengths_from_export: Pre-computed initial lengths (optional)
        
    Returns:
        Tuple of (all_stresses, region_edges, region_initial_lengths)
    """
    # Get Young's modulus from config (same as stress_analysis.py)
    young_modulus = config_manager.get('material_properties.young_modulus_silicone', vu.YOUNG_MODULUS_SILICON)
    
    # Create edges - use saved edges if available
    if saved_edges is not None and len(saved_edges) > 0:
        print(f"Using saved edges from mesh editor: {len(saved_edges)} edges")
        vertices = cluster_means
        valid_edges = saved_edges
        # Calculate initial lengths from saved edges
        initial_lengths = np.linalg.norm(vertices[valid_edges[:, 0]] - vertices[valid_edges[:, 1]], axis=1)
    else:
        print("No saved edges found, generating fresh Delaunay edges")
        vertices, edges, _ = vu.create_delaunay_edges(cluster_means)
        valid_edges, initial_lengths = vu.remove_outlier_edges(vertices, edges, 2.0) # Default outlier thresh
    
    # Find edges that have at least one vertex in the selected region
    region_edges = []
    region_edge_indices = []
    
    for i, edge in enumerate(valid_edges):
        if edge[0] in selected_vertex_indices or edge[1] in selected_vertex_indices:
            region_edges.append(edge)
            region_edge_indices.append(i)
    
    print(f"Found {len(region_edges)} edges connected to selected vertices")
    
    if len(region_edges) == 0:
        print("Warning: No edges found in selected region!")
        return [], np.array([]), np.array([])
    
    region_edges = np.array(region_edges)
    
    if initial_lengths_from_export is not None:
        region_initial_lengths = initial_lengths_from_export[region_edge_indices]
    else:
        region_initial_lengths = initial_lengths[region_edge_indices]
    
    all_stresses = []
    
    print("\nCalculating stresses over time...")
    
    if cluster_positions_timeseries is not None:
        print("Using pre-computed cluster positions from cluster mesh export")
        num_frames = cluster_positions_timeseries.shape[0]
        total_frames = num_frames - 1
        
        for j in range(1, num_frames):
            # Progress bar
            progress = int((j / total_frames) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
            
            # Use pre-computed cluster positions
            new_cluster_means = cluster_positions_timeseries[j]
            
            # Calculate stress using Neo-Hookean model from vessel_utils
            stress = vu.compute_edge_stress(new_cluster_means, region_edges, region_initial_lengths, young_modulus)
            all_stresses.append(stress)
    else:
        total_frames = len(dataset) - 1
        
        # Track previous frame's cluster means for continuity
        prev_cluster_means = cluster_means.copy()
        
        for j in range(1, len(dataset)):
            # Progress bar
            progress = int((j / total_frames) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
            
            new_frame = dataset[j].astype(np.float64)
            new_points = new_frame[:, :3]  # Only use first 3 columns (X, Y, Z)
            
            # Update cluster means matching the shape of cluster_means
            new_cluster_means = np.zeros_like(cluster_means)
            
            for k in range(len(cluster_means)):
                cluster_mask = labels == k
                if np.any(cluster_mask):
                    new_cluster_means[k] = np.mean(new_points[cluster_mask], axis=0)
                else:
                    new_cluster_means[k] = prev_cluster_means[k]
            
            # Update previous cluster means for next iteration
            prev_cluster_means = new_cluster_means.copy()
            
            # Calculate stress using Neo-Hookean model from vessel_utils
            stress = vu.compute_edge_stress(new_cluster_means, region_edges, region_initial_lengths, young_modulus)
            all_stresses.append(stress)
    
    print()  # New line after progress bar
    print(f"Stress calculation complete!")
    
    return all_stresses, region_edges, region_initial_lengths

def run_stress_analysis_region(config_manager):
    """
    Run region-specific stress analysis using the provided configuration manager.
    
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
    
    # Get experiment info
    experiment_name = config['experiment']['name']
    output_dir = config['experiment']['output_dir']
    data_path = config['experiment']['data_path']
    
    # Pre-declare variables
    cluster_positions_timeseries = None
    initial_lengths_from_export = None
    use_cluster_export = USE_CLUSTER_EXPORT
    saved_final_means = None
    saved_edges = None
    
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
            saved_edges = export_data["edges"]
            initial_lengths_from_export = export_data["initial_lengths"]
            cluster_means_from_export = export_data["initial_cluster_means"]
            
            print(f"  Loaded {cluster_positions_timeseries.shape[0]} frames")
            print(f"  Loaded {cluster_positions_timeseries.shape[1]} clusters")
            print(f"  Loaded {len(saved_edges)} edges")
            
            # Use cluster means from export
            cluster_means = cluster_means_from_export
        else:
            print(f"\nNo cluster mesh export found in: {export_dir}")
            print("Will compute cluster positions on the fly...")
            use_cluster_export = False

    # Load raw data
    data_exists = os.path.exists(data_path)
    if data_exists:
        print(f"Loading raw data from: {data_path}")
        dataset = np.load(data_path)
        print(f"Loaded data shape: {dataset.shape}")
        
        # Check for existing clustering state
        state_path = find_clustering_state_for_experiment(experiment_name, output_dir)
        if not os.path.exists(state_path):
            print(f"Error: Clustering state not found at {state_path}")
            return
        print(f"Found clustering state: {state_path}")

        # Validate data format and extract XYZ and RGB coordinates
        if len(dataset.shape) == 3:
            frame = dataset[0]  # (N, 6)
        elif len(dataset.shape) == 2:
            frame = dataset
        else:
            print("Error: Dataset must be 2D (points, 6) or 3D (time_frames, points, 6)")
            return
            
        xyz = frame[:, :3]
        rgb = np.clip(frame[:, 3:], 0.0, 1.0)
        
        # Apply color-then-spatial clustering
        print("\nApplying color-then-spatial clustering...")
        labels_raw, _ = color_then_spatial_clustering(xyz, rgb, config_manager.config)
        
        # Calculate mean locations of clusters
        cluster_means_raw, _ = vu.calculate_cluster_means(xyz, labels_raw)
        
        # Load clustering state (consistency)
        print("Loading clustering state...")
        saved_final_means, saved_final_edges, _, _ = load_saved_clustering_state(experiment_name, output_dir)
        
        if saved_final_means is not None:
            print(f"Applied filtering: {len(cluster_means_raw)} -> {len(saved_final_means)} cluster means")
            labels, cluster_points = remap_labels_to_saved_means(xyz, labels_raw, cluster_means_raw, saved_final_means)
            if cluster_positions_timeseries is None:
                cluster_means = saved_final_means
                saved_edges = saved_final_edges
            print(f"Applied clustering state. {len(cluster_means)} clusters.")
        else:
            if cluster_positions_timeseries is None:
                labels = labels_raw
                cluster_means = cluster_means_raw
                saved_edges = None
            print("No saved state loaded, cannot proceed without clustering state")
            return
    else:
        if cluster_positions_timeseries is not None:
            print(f"⚠️ Raw data NOT found at: {data_path}")
            print("But cluster export is present. Proceeding with export data only (no background points).")
            # Dummy variables for compatibility
            dataset = None
            labels = None
            cluster_points = None
            # cluster_means already set from export
        else:
            print(f"❌ ERROR: Raw data NOT found at: {data_path}")
            print("And no cluster export available. Cannot proceed.")
            return
    
    # Selection and analysis continues using cluster_means
    
    # Get camera position from config
    camera_position = config_manager.config.get('camera', {}).get('position')
    if not camera_position:
        print("Warning: No camera position found in config. Using default camera position.")
        camera_position = [(-0.31676350915969265, -17.727903759830316, -5.178437900637502),
                          (0.23947606203271332, 0.26414248443613014, -1.2407439169630974),
                          (-0.9948541431039188, 0.050037906334993124, -0.08809904584374163)]
    
    # Interactive multi-mode selection (clicking + rectangle selection)
    print("[DEBUG] About to call interactive_multi_selection...")
    selected_vertex_indices = interactive_multi_selection(cluster_means, camera_position)
    
    print(f"[DEBUG] interactive_multi_selection returned")
    print(f"[DEBUG] Return value type: {type(selected_vertex_indices)}")
    print(f"[DEBUG] Return value length: {len(selected_vertex_indices) if selected_vertex_indices else 0}")
    
    if not selected_vertex_indices:
        print("No vertices selected. Exiting.")
        return
    
    print(f"[DEBUG] Selection validated, proceeding...")
    print(f"Selected {len(selected_vertex_indices)} vertices: {selected_vertex_indices[:10]}...")
    
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
    
    # Calculate stresses for the selected region
    print("\nCalculating stresses for selected region...")
    all_stresses, region_edges, region_initial_lengths = calculate_region_stresses(
        labels, cluster_means, dataset, selected_vertex_indices, config_manager, saved_edges,
        cluster_positions_timeseries=cluster_positions_timeseries,
        initial_lengths_from_export=initial_lengths_from_export
    )
    
    if not all_stresses:
        print("No stresses calculated. Exiting.")
        return
    
    # Calculate average stress over time and find maximum
    average_stresses_mpa = [np.mean(stress) / 1e6 for stress in all_stresses]
    max_avg_stress = max(average_stresses_mpa)
    min_avg_stress = min(average_stresses_mpa)
    mean_avg_stress = np.mean(average_stresses_mpa)
    
    # Print stress statistics
    print("\n" + "="*60)
    print("📊 STRESS ANALYSIS RESULTS")
    print("="*60)
    print(f"Region: {region_name}")
    print(f"Selected vertices: {len(selected_vertex_indices)}")
    print(f"Edges analyzed: {len(all_stresses[0]) if all_stresses else 0}")
    print("-"*60)
    print(f"Maximum average stress: {max_avg_stress:.4f} MPa")
    print(f"Minimum average stress: {min_avg_stress:.4f} MPa")
    print(f"Mean average stress:    {mean_avg_stress:.4f} MPa")
    print("="*60 + "\n")
    
    # # Visualize the deformation with stress color coding
    # print("Starting visualization...")
    # visualize_region_deformation(labels, cluster_means, dataset, selected_vertex_indices,
    #                             region_edges, region_initial_lengths, region_name, 
    #                             config_manager, all_stresses, saved_edges)
    
    # Create output directory for region analysis
    region_output_dir = os.path.join(output_dir, SAVE_DIRECTORY)
    os.makedirs(region_output_dir, exist_ok=True)
    
    # Save stress values
    clean_region_name = region_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    clean_region_name = ''.join(c for c in clean_region_name if c.isalnum() or c == '_')
    output_filename = f"{OUTPUT_FILE_PREFIX}_{clean_region_name}.npy"
    output_path = os.path.join(region_output_dir, output_filename)
    np.save(output_path, all_stresses)
    print(f'Stress calculations saved to {output_path}')
    
    # Plot average stress
    plot_average_stress_region(all_stresses, region_name, experiment_name, output_dir, total_experiment_time) 