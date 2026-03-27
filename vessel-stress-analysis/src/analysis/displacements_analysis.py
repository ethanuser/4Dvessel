"""
Displacement analysis module.
Tracks the displacement of cluster centroids over time and visualizes it.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import time
import os
import datetime
import glob
from typing import List

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
# Scale factors for converting 3D units to real-world MM
# Copied from displacement_analysis.py for consistency
CHESS_TOTAL_LENGTH_M    = 6.195 / 100        # length across 8 squares (metres)
CHESS_SQUARES_ALONG_X   = 8              # number of squares measured
CHESS_SQUARE_LENGTH_M   = CHESS_TOTAL_LENGTH_M / CHESS_SQUARES_ALONG_X  # ~0.774375 cm per square
CALIB_SCALING_FACTOR    = 0.667125       # same factor used in calibration objp (units per square)
CORRECTION_FACTOR       = 6.195 / 5.053 # ratio of the length of the chessboard in real life to the length of the chessboard we've calculated
UNIT_TO_MM              = (CHESS_SQUARE_LENGTH_M / CALIB_SCALING_FACTOR) * 1000 * CORRECTION_FACTOR  # Convert to mm

print("Scaling Factor: ", str(UNIT_TO_MM))

# ── Visualization constants ──────────────────────────────────────────────
DEFAULT_MAX_DISPLACEMENT_MM = vu.MAX_DISPLACEMENT_MM
DEFAULT_POINT_SIZE = 10
USE_DEFAULT_POINT_SIZE = True

# ── Auto-Scaling Configuration ───────────────────────────────────────────
AUTO_SCALE_MAX_DISPLACEMENT = False  # If True, overrides DEFAULT_MAX_DISPLACEMENT_MM
AUTO_SCALE_METRIC = 'percentile'    # 'percentile' or 'max'
AUTO_SCALE_PERCENTILE_VALUE = 100    # Percentile to use (e.g. 95, 99) if metric is 'percentile'

def run_displacements_analysis(config_manager):
    """
    Run displacement analysis using the provided configuration manager.
    """
    config = config_manager.config
    
    experiment_name = config['experiment']['name']
    output_dir = config['experiment']['output_dir']
    data_path = config['experiment'].get('data_path')
    
    print(f"\nExperiment: {experiment_name}")
    
    # Pre-declare variables
    cluster_positions_timeseries = None
    use_cluster_export = USE_CLUSTER_EXPORT
    saved_final_means = None
    
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
            cluster_means0 = export_data["initial_cluster_means"] # Initial means from export
            
            print(f"  Loaded {cluster_positions_timeseries.shape[0]} frames")
            print(f"  Loaded {cluster_positions_timeseries.shape[1]} clusters")
        else:
            print(f"\nNo cluster mesh export found at: {export_path}")
            print("Will compute cluster positions on the fly...")
            use_cluster_export = False  # Fall back to computing on the fly

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

    # Load raw data and perform initial clustering if needed
    data_exists = data_path and os.path.exists(data_path)
    if data_exists:
        print(f"Loading raw data from: {data_path}")
        dataset = np.load(data_path)
        print(f"Loaded data shape: {dataset.shape}")
        
        # Process Frame 0 to establish clusters and initial positions
        frame0 = dataset[0]
        xyz0 = frame0[:, :3]
        rgb0 = np.clip(frame0[:, 3:], 0.0, 1.0)
        
        # Initial Clustering
        print("\nApplying initial clustering...")
        labels, _ = color_then_spatial_clustering(xyz0, rgb0, config_manager.config)
        
        # Calculate initial means
        cluster_means0_raw, _ = vu.calculate_cluster_means(xyz0, labels)
        
        # Load clustering state (consistency)
        print("Loading clustering state...")
        saved_final_means, _, _, _ = load_saved_clustering_state(experiment_name, output_dir)
        
        if saved_final_means is not None:
            labels, _ = remap_labels_to_saved_means(xyz0, labels, cluster_means0_raw, saved_final_means)
            # If export was not used, set cluster_means0 from saved_final_means
            if cluster_positions_timeseries is None:
                cluster_means0 = saved_final_means
            print(f"Applied clustering state. {len(cluster_means0)} clusters.")
        else:
            # If export was not used, set cluster_means0 from raw calculation
            if not use_cluster_export:
                cluster_means0 = cluster_means0_raw
                print("Using calculated clusters (no saved state filtering applied).")
            else:
                print("No saved state loaded, but using cluster export for geometry.")
            
        # If we loaded from export, prioritize its means for consistency
        if cluster_positions_timeseries is not None:
            # Note: We keep labels from raw clustering if dataset exists, but computation uses export means
            pass
    else:
        if use_cluster_export:
            print(f"⚠️ Raw data NOT found at: {data_path}")
            print("But cluster export is present. Proceeding with export data only (no background points).")
            # Dummy variables for dataset and labels to allow loops to run
            # dataset is only used for length in the use_cluster_export Phase 1 block
            dataset = [None] * cluster_positions_timeseries.shape[0]
            labels = None
        else:
            print(f"❌ ERROR: Raw data NOT found at: {data_path}")
            print("And no cluster export available. Cannot proceed.")
            return

    # Initial Positions (t=0)
    initial_positions = cluster_means0.copy()
    
    # Visualization Setup
    save_render = config_manager.get('visualization.save_render', True)
    frame_delay = config_manager.get('visualization.frame_delay', 0.1)
    if not USE_DEFAULT_POINT_SIZE:
        point_size = config_manager.get('visualization.initial_point_size', DEFAULT_POINT_SIZE)
    else:
        point_size = DEFAULT_POINT_SIZE
    camera_position = config_manager.get('camera.position')
    
    # Displacement Scale
    max_disp_config_mm = config_manager.get('visualization.max_displacement_mm', DEFAULT_MAX_DISPLACEMENT_MM)
    
    if save_render:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, "displacements_analysis", f"renders_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        print(f"Renders will be saved to: {save_path}")

    plotter = pv.Plotter()
    
    # Initial Plot
    # At t=0, displacement is 0
    cm = vu.get_colormap()
    colors0 = vu.get_colors(np.zeros(len(initial_positions)), cm, vmin=0, vmax=max_disp_config_mm)
    
    # Create PolyData for centers
    cloud = pv.PolyData(initial_positions)
    cloud['colors'] = colors0
    
    plotter.add_mesh(cloud, scalars='colors', rgb=True, point_size=point_size, render_points_as_spheres=True)
    
    if camera_position:
        plotter.camera_position = camera_position
        
    plotter.show(interactive_update=True)
    
    # Main Loop - Step 1: Pre-calculate all displacements
    all_frame_results = []
    all_displacements_mm_flat = [] # For auto-scaling stats
    
    # Paths for caching
    save_data_dir = os.path.join(output_dir, "displacements_analysis")
    os.makedirs(save_data_dir, exist_ok=True)
    displacements_path = os.path.join(save_data_dir, "displacements_mm.npy")
    means_path = os.path.join(save_data_dir, "cluster_means.npy")
    
    # Check if cached data exists
    if os.path.exists(displacements_path) and os.path.exists(means_path):
        print(f"\nPhase 1: Loading cached data from {save_data_dir}...")
        try:
            cached_displacements = np.load(displacements_path, allow_pickle=True)
            cached_means = np.load(means_path, allow_pickle=True)
            
            # Verify lengths match dataset
            if len(cached_displacements) == len(dataset):
                print("Cache hit! Skipping calculation.")
                total_frames = len(cached_displacements)
                # Reconstruct (assuming same frame count)
                # cached_displacements might be object array of arrays
                for j in range(len(cached_displacements)):
                     d_mm = cached_displacements[j]
                     c_means = cached_means[j]
                     # Ensure proper numpy array types when loading from cache
                     d_mm = np.asarray(d_mm, dtype=np.float64)
                     c_means = np.asarray(c_means, dtype=np.float64)
                     all_frame_results.append({
                        'current_means': c_means,
                        'displacements_mm': d_mm
                     })
                     all_displacements_mm_flat.append(d_mm)
            else:
                print("Cache frame count mismatch. Recalculating...")
                raise ValueError("Cache mismatch")
                
        except Exception as e:
            print(f"Error loading cache: {e}. Recalculating...")
            all_frame_results = []
            all_displacements_mm_flat = []
    
    # If cache failed or didn't exist, calculate
    if not all_frame_results:
        total_frames = len(dataset)
        
        if use_cluster_export:
            print(f"\nPhase 1: Using pre-computed displacements for {total_frames} frames...")
            num_frames_traj = cluster_positions_timeseries.shape[0]
            
            for j in range(total_frames):
                # Progress
                progress = int((j / total_frames) * 50)
                bar = "█" * progress + "░" * (50 - progress)
                print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
                
                # Map frame_idx to trajectory index
                traj_idx = min(j, num_frames_traj - 1)
                current_means = cluster_positions_timeseries[traj_idx]
                
                # Calculate displacement in UNITS
                d_vec = current_means - initial_positions
                displacements_units = np.linalg.norm(d_vec, axis=1)
                displacements_mm = displacements_units * UNIT_TO_MM
                
                all_frame_results.append({
                    'current_means': current_means,
                    'displacements_mm': displacements_mm
                })
                all_displacements_mm_flat.append(displacements_mm)
            
            print(f"\nSaving cache to {save_data_dir}...")
            # Reconstruct means array for saving
            means_to_save = [res['current_means'] for res in all_frame_results]
            np.save(displacements_path, np.array(all_displacements_mm_flat, dtype=object))
            np.save(means_path, np.array(means_to_save, dtype=object))
            
        else:
            print(f"\nPhase 1: Calculating displacements for {total_frames} frames...")
            all_means_to_save = []
            
            # Track previous frame's cluster means for continuity
            prev_cluster_means = initial_positions.copy()
            
            for j in range(total_frames):
                # Progress
                progress = int((j / total_frames) * 50)
                bar = "█" * progress + "░" * (50 - progress)
                print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
                
                frame = dataset[j]
                xyz_curr = frame[:, :3]
                
                # Recalculate centroids matching the shape of initial_positions
                current_means = np.zeros_like(initial_positions)
                
                for k in range(len(initial_positions)):
                    cluster_mask = labels == k
                    if np.any(cluster_mask):
                        current_means[k] = np.mean(xyz_curr[cluster_mask], axis=0)
                    else:
                        # If cluster empty, use previous position
                        current_means[k] = prev_cluster_means[k]
                
                # Update previous for next iteration
                prev_cluster_means = current_means.copy()
                
                # Calculate displacement in UNITS
                d_vec = current_means - initial_positions
                displacements_units = np.linalg.norm(d_vec, axis=1)
                
                # Convert to MM
                displacements_mm = displacements_units * UNIT_TO_MM
                
                all_frame_results.append({
                    'current_means': current_means,
                    'displacements_mm': displacements_mm
                })
                all_displacements_mm_flat.append(displacements_mm)
                all_means_to_save.append(current_means)
                
            # Save cache
            print(f"\nSaving cache to {save_data_dir}...")
            np.save(displacements_path, np.array(all_displacements_mm_flat, dtype=object))
            np.save(means_path, np.array(all_means_to_save, dtype=object))

    # Step 2: Determine Max Displacement for Scaling
    if AUTO_SCALE_MAX_DISPLACEMENT:
        flat_disps = np.concatenate(all_displacements_mm_flat)
        if AUTO_SCALE_METRIC == 'max':
            max_disp_val = np.max(flat_disps)
            print(f"\nAuto-scaling: Using MAX displacement = {max_disp_val:.4f} mm")
        elif AUTO_SCALE_METRIC == 'percentile':
            max_disp_val = np.percentile(flat_disps, AUTO_SCALE_PERCENTILE_VALUE)
            print(f"\nAuto-scaling: Using {AUTO_SCALE_PERCENTILE_VALUE}th percentile = {max_disp_val:.4f} mm")
        else:
            max_disp_val = max_disp_config_mm # Fallback
    else:
        max_disp_val = max_disp_config_mm
        print(f"\nUsing fixed max displacement = {max_disp_val:.4f} mm")
    
    # Visualization setup
    cm = vu.get_colormap()

    # Step 3: Visualization Loop
    print(f"\nPhase 2: Visualizing {total_frames} frames...")
    
    for j, res in enumerate(all_frame_results):
        # Progress
        # print(f"\rRender {j}/{total_frames}", end="", flush=True) # Optional clutter
        
        current_means = res['current_means']
        displacements_mm = res['displacements_mm']
        
        # Visualization
        colors = vu.get_colors(displacements_mm, cm, vmin=0, vmax=max_disp_val)
        
        plotter.clear()
        cloud.points = current_means
        cloud['colors'] = colors
        plotter.add_mesh(cloud, scalars='colors', rgb=True, point_size=point_size, render_points_as_spheres=True)
        
        plotter.add_text(f'Time Step {j}\nMax Disp: {np.max(displacements_mm):.2f} mm\nScale Max: {max_disp_val:.2f} mm', font_size=5)
        
        if save_render:
            plotter.screenshot(os.path.join(save_path, f'frame_{j:04d}.png'))
            
        plotter.update()
            
    print(f"\nDisplacement analysis complete.")
    plotter.close()

