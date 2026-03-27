"""
Interactive mesh editor module using centralized configuration.
This module provides the functionality for manually editing cluster means and edges.
"""
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay, distance

import pyvista as pv
import os
import datetime
import json
import signal
import time
import glob
from typing import List, Tuple, Optional

from core.clustering import (
    color_then_spatial_clustering,
    calculate_cluster_means,
    create_delaunay_edges,
    remove_outlier_edges
)
from utils.clustering_state_utils import (
    load_clustering_state,
    apply_clustering_state,
    select_state_file,
    find_clustering_state_for_experiment,
    load_saved_clustering_state
)
from utils.rectangle_select_points import select_points_with_rectangle

def remove_white_cluster_means(cluster_means: np.ndarray, cluster_points: List[np.ndarray], 
                                points: np.ndarray, rgb_values: np.ndarray, 
                                white_threshold: float = 0.95) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    """
    Automatically remove cluster means that are completely white.
    
    Args:
        cluster_means: Array of cluster mean positions
        cluster_points: List of point arrays for each cluster
        points: All points in the dataset
        rgb_values: RGB values for all points
        white_threshold: Threshold for considering a color as white (default: 0.95)
        
    Returns:
        Tuple of (filtered_cluster_means, filtered_cluster_points, kept_indices)
    """
    if len(cluster_means) == 0:
        return cluster_means, cluster_points, []
    
    # Compute the average RGB color for each cluster mean
    cluster_mean_colors = []
    for mean in cluster_means:
        # Find the closest points to this mean and get their average RGB
        distances = np.linalg.norm(points - mean, axis=1)
        closest_indices = np.argsort(distances)[:20]  # Get 20 closest points
        closest_rgb = rgb_values[closest_indices]
        avg_rgb = np.mean(closest_rgb, axis=0)
        cluster_mean_colors.append(avg_rgb)
    
    cluster_mean_colors = np.array(cluster_mean_colors)
    
    # Identify non-white cluster means (keep those that are NOT white)
    # A cluster is white if all RGB channels are above the threshold
    is_white = np.all(cluster_mean_colors >= white_threshold, axis=1)
    keep_mask = ~is_white
    
    # Count how many white cluster means were found
    num_white = np.sum(is_white)
    
    if num_white > 0:
        print(f"\n=== AUTOMATIC WHITE CLUSTER REMOVAL ===")
        print(f"Found {num_white} white cluster means (threshold: {white_threshold})")
        
        # Filter cluster means and points
        filtered_cluster_means = cluster_means[keep_mask]
        filtered_cluster_points = [cluster_points[i] for i in range(len(cluster_points)) if keep_mask[i]]
        kept_indices = np.where(keep_mask)[0].tolist()
        
        print(f"Clusters before: {len(cluster_means)}, after: {len(filtered_cluster_means)}")
        print("=" * 50 + "\n")
        
        return filtered_cluster_means, filtered_cluster_points, kept_indices
    else:
        # No white clusters found, return everything as-is
        return cluster_means, cluster_points, list(range(len(cluster_means)))

def load_experiment_data(config: dict):
    """Load data based on experiment configuration."""
    data_path = config['experiment']['data_path']
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    dataset = np.load(data_path)  # shape (V, N, 6)
    return dataset, data_path


def group_close_clusters(cluster_means: np.ndarray, cluster_points: List[np.ndarray], 
                        distance_threshold: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Group clusters that are close to each other."""
    grouped_means = []
    grouped_points = []
    visited = set()
    
    for i in range(len(cluster_means)):
        if i not in visited:
            cluster_group = cluster_points[i]
            visited.add(i)
            
            for j in range(i + 1, len(cluster_means)):
                if j not in visited and distance.euclidean(cluster_means[i], cluster_means[j]) < distance_threshold:
                    cluster_group = np.vstack((cluster_group, cluster_points[j]))
                    visited.add(j)
            
            grouped_mean = np.mean(cluster_group, axis=0)
            grouped_means.append(grouped_mean)
            grouped_points.append(cluster_group)
    
    return np.array(grouped_means), grouped_points

def save_clustering_state(cluster_means: np.ndarray, current_edges: Optional[np.ndarray], 
                         original_cluster_means: np.ndarray, original_edges: Optional[np.ndarray], 
                         experiment_name: str, output_dir: str, custom_filename: Optional[str] = None) -> str:
    """
    Save the current clustering state (deleted points/edges) to a JSON file.
    
    Args:
        cluster_means: Current cluster means
        current_edges: Current edges
        original_cluster_means: Original cluster means (base for this session)
        original_edges: Original edges (base for this session)
        experiment_name: Name of the experiment
        output_dir: Output directory for the experiment
        custom_filename: Optional custom filename
        
    Returns:
        Path to the saved state file
    """
    # Create state filename based on experiment name
    if custom_filename:
        state_filename = custom_filename
    else:
        state_filename = f"clustering_state_{experiment_name}.json"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to state file
    state_path = os.path.join(output_dir, state_filename)
    
    # Create mapping of which original points are kept
    kept_point_indices = []
    
    # More efficient comparison: create a set of current means for faster lookup
    current_means_set = set()
    for mean in cluster_means:
        # Round to 6 decimal places to avoid floating point precision issues
        rounded_mean = tuple(np.round(mean, 6))
        current_means_set.add(rounded_mean)
    
    for i, original_mean in enumerate(original_cluster_means):
        rounded_original = tuple(np.round(original_mean, 6))
        if rounded_original in current_means_set:
            kept_point_indices.append(i)
    
    # Create mapping of which original edges are kept
    kept_edge_indices = []
    
    if current_edges is not None and original_edges is not None:
        # More efficient comparison: create a set of current edges for faster lookup
        current_edges_set = set()
        for edge in current_edges:
            # Sort edge indices to ensure consistent comparison
            sorted_edge = tuple(sorted(edge))
            current_edges_set.add(sorted_edge)
        
        for i, original_edge in enumerate(original_edges):
            sorted_original = tuple(sorted(original_edge))
            if sorted_original in current_edges_set:
                kept_edge_indices.append(i)
    
    # Save state
    state_data = {
        'experiment_name': experiment_name,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        'kept_point_indices': kept_point_indices,
        'kept_edge_indices': kept_edge_indices,
        'original_cluster_means': original_cluster_means.tolist(),
        'original_edges': original_edges.tolist() if original_edges is not None else None,
        'final_cluster_means': cluster_means.tolist(),
        'final_edges': current_edges.tolist() if current_edges is not None else None
    }
    
    with open(state_path, 'w') as f:
        json.dump(state_data, f, indent=2)
    
    print(f"Clustering state saved to: {state_path}")
    return state_path

def load_and_apply_state(original_cluster_means: np.ndarray, 
                        original_edges: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], List[int], List[int]]:
    """
    Load and apply a saved clustering state.
    
    Args:
        original_cluster_means: Original cluster means
        original_edges: Original edges
        
    Returns:
        Tuple of (filtered_cluster_means, filtered_edges, kept_point_indices, kept_edge_indices)
    """
    state_file = select_state_file()
    if state_file is None:
        print("No state file selected. Using original clustering.")
        return original_cluster_means, original_edges, list(range(len(original_cluster_means))), []
    
    try:
        state_data = load_clustering_state(state_file)
        print(f"Loaded clustering state from: {state_file}")
        print(f"Original clusters: {len(original_cluster_means)}, Filtered clusters: {len(state_data['kept_point_indices'])}")
        
        return apply_clustering_state(original_cluster_means, original_edges, state_data)
    
    except Exception as e:
        print(f"Error loading state file: {e}")
        print("Using original clustering.")
        return original_cluster_means, original_edges, list(range(len(original_cluster_means))), []

def select_points_rectangle_mode(cluster_means: np.ndarray, camera_position: List) -> List[int]:
    """
    Use rectangle selection to select multiple cluster means at once.
    
    Args:
        cluster_means: Array of cluster mean positions
        camera_position: Camera position for the visualization
        
    Returns:
        List of selected point indices
    """
    # Use the rectangle selection utility
    selected_indices = select_points_with_rectangle(
        points=cluster_means,
        camera_position=camera_position,
        point_size=15,
        point_color='red',
        selected_color='magenta',
        selected_size=20
    )
    
    return selected_indices

def create_interactive_colored_visualization(points: np.ndarray, labels: np.ndarray, 
                                           cluster_means: np.ndarray, rgb_values: np.ndarray, 
                                           color_centroids: Optional[np.ndarray] = None, 
                                           title: str = "Interactive Colored Visualization", 
                                           experiment_name: str = None,
                                           output_dir: str = None,
                                           config_manager = None,
                                           base_cluster_means: Optional[np.ndarray] = None,
                                           base_edges: Optional[np.ndarray] = None,
                                           true_original_cluster_means: Optional[np.ndarray] = None,
                                           true_original_edges: Optional[np.ndarray] = None) -> pv.Plotter:
    """Create an interactive PyVista visualization with edge removal and undo functionality."""
    
    # Extract parameters from config
    distance_threshold = config_manager.get('clustering.distance_threshold', 0.1)
    plane_height_limit_factor = config_manager.get('visualization.plane_height_limit_factor', 10.0)
    point_selection_threshold = config_manager.get('visualization.point_selection_threshold', 0.5)
    edge_selection_threshold = config_manager.get('visualization.edge_selection_threshold', 1.0)
    cluster_mean_point_size = config_manager.get('visualization.cluster_mean_point_size', 15)
    selected_point_size = config_manager.get('visualization.selected_point_size', 30)
    edge_line_width = config_manager.get('visualization.edge_line_width', 3)
    selected_edge_line_width = config_manager.get('visualization.selected_edge_line_width', 6)
    cyan_edge_line_width = config_manager.get('visualization.cyan_edge_line_width', 5)
    camera_position = config_manager.get('camera.position')
    
    plotter = pv.Plotter(title=title)
    
    # Store original state for saving - use provided base state or current as fallback
    if base_cluster_means is not None:
        original_cluster_means = base_cluster_means.copy()
        print(f"Using provided base state with {len(original_cluster_means)} cluster means")
    else:
        original_cluster_means = cluster_means.copy()
        print(f"Using current state as base with {len(original_cluster_means)} cluster means")
    
    # Use provided base edges or set to None (will be set when edges are created)
    original_edges = base_edges.copy() if base_edges is not None else None
    
    # Store the TRUE original clustering state for saving (before any edits)
    # This should always be the initial clustering, not the loaded state
    if true_original_cluster_means is not None:
        true_original_cluster_means_for_save = true_original_cluster_means.copy()
        true_original_edges_for_save = true_original_edges.copy() if true_original_edges is not None else None
        print(f"Using provided true original state with {len(true_original_cluster_means_for_save)} cluster means")
    else:
        # Fallback to current state if no true original provided
        true_original_cluster_means_for_save = original_cluster_means.copy()
        true_original_edges_for_save = original_edges.copy() if original_edges is not None else None
        print(f"Using current state as true original with {len(true_original_cluster_means_for_save)} cluster means")
    
    # Store edge history for undo functionality
    edge_history = []
    current_edges = None
    current_edge_mesh = None
    selected_edge_mesh = None  # Separate mesh for selected edges
    selected_edges = set()  # Track selected edges
    
    # --- Plane and edge selection state ---
    # Compute plane_height_limit as 1/factor of the largest bounding box dimension of cluster means
    bbox_min = np.min(cluster_means, axis=0)
    bbox_max = np.max(cluster_means, axis=0)
    bbox_dims = bbox_max - bbox_min
    plane_height_limit = np.max(bbox_dims) / plane_height_limit_factor
    plane_defined = {'done': False}
    plane_geom = {'n': None}  # store normal here
    edge_mask = np.zeros(0, bool)  # Will be initialized when edges are created
    selected_edge_mask = np.zeros(0, bool)  # Will be initialized when edges are created
    plane_edge_indices = []  # Store indices of edges that intersect the plane
    
    # Separate noise points from regular points for different sizing
    noise_mask = (labels == -1)
    regular_mask = ~noise_mask
    
    # Add cluster means with colors based on the average RGB of nearby points
    cluster_mean_colors = None  # Store colors for reuse after deletions
    if len(cluster_means) > 0:
        cluster_mean_colors_list = []
        for mean in cluster_means:
            # Find the closest points to this mean and get their average RGB
            distances = np.linalg.norm(points - mean, axis=1)
            closest_indices = np.argsort(distances)[:20]  # Get 20 closest points
            closest_rgb = rgb_values[closest_indices]
            avg_rgb = np.mean(closest_rgb, axis=0)
            cluster_mean_colors_list.append(avg_rgb)
        
        cluster_mean_colors = np.array(cluster_mean_colors_list)
        
        # OPTIMIZED: Use single PolyData with per-vertex colors instead of individual actors
        # This is MUCH faster to create and remove than 100+ individual actors
        point_cloud = pv.PolyData(cluster_means)
        point_cloud['colors'] = (cluster_mean_colors * 255).astype(np.uint8)
        plotter.add_mesh(point_cloud, scalars='colors', rgb=True, point_size=cluster_mean_point_size,
                        render_points_as_spheres=True, name='cluster_means', style='points')
    
    # Create and plot the Delaunay edges
    if len(cluster_means) > 3:  # Need at least 4 points for 3D Delaunay
        # CRITICAL: Check if we have saved edges to use (from loaded state)
        if base_edges is not None and len(base_edges) > 0:
            # Use saved edges instead of regenerating Delaunay
            valid_edges = base_edges.copy()
            current_edges = valid_edges.copy()
            original_edges = base_edges.copy()
            print(f"Using saved edges as current edges: {len(valid_edges)} edges")
        else:
            # No saved edges, generate fresh Delaunay edges
            vertices, edges, _ = create_delaunay_edges(cluster_means)
            valid_edges, _ = remove_outlier_edges(vertices, edges, config_manager.config)
            
            if len(valid_edges) > 0:
                # Store initial edges
                current_edges = valid_edges.copy()
                
                # Use provided base edges or current edges as original
                if original_edges is None:
                    original_edges = valid_edges.copy()  # Store original edges
                    print(f"Set original edges to current edges: {len(original_edges)} edges")
                else:
                    print(f"Using provided base edges as original: {len(original_edges)} edges")
        
        if len(valid_edges) > 0:
            # Set true original edges if not already set
            if true_original_edges_for_save is None:
                true_original_edges_for_save = valid_edges.copy()
                print(f"Set true original edges: {len(true_original_edges_for_save)} edges")
            
            edge_history.append(current_edges.copy())
            
            # Initialize edge masks
            edge_mask = np.zeros(len(valid_edges), bool)
            selected_edge_mask = np.zeros(len(valid_edges), bool)
            
            # Generate lines for the edges
            lines = np.hstack([np.full((valid_edges.shape[0], 1), 2), valid_edges])
            edge_mesh = pv.PolyData(cluster_means, lines=lines)
            current_edge_mesh = edge_mesh
            plotter.add_mesh(edge_mesh, color='red', line_width=edge_line_width, style='surface', name='edges')
    
    # --- Point selection state ---
    selected_points = set()  # Will store coordinate tuples (x, y, z) as unique identifiers

    def get_plane_from_camera(camera_position):
        """Return (origin, normal, up) for the vertical slicing plane."""
        pos, focal, up = map(np.asarray, camera_position)
        view_vec = focal - pos
        view_vec /= np.linalg.norm(view_vec)
        up_vec = up / np.linalg.norm(up)
        # plane normal = view × up  (vertical plane through screen centre)
        normal = np.cross(view_vec, up_vec)
        normal /= np.linalg.norm(normal)
        return pos, normal, up_vec

    def edges_intersect_plane(plotter, edges, verts, origin, normal, height_limit, gaze_origin, gaze_vector, tol=1e-9):
        """Boolean mask: True for edges that cross the plane and whose intersection point
        with the plane is within height_limit of the gaze vector line (perpendicular distance)."""
        origin = np.asarray(origin)
        gaze_origin = np.asarray(gaze_origin)
        gaze_vector = np.asarray(gaze_vector)
        # Normalize gaze_vector to ensure it's a unit vector
        gaze_norm = np.linalg.norm(gaze_vector)
        if gaze_norm > tol:
            gaze_vector = gaze_vector / gaze_norm
        
        p0 = verts[edges[:, 0]]
        p1 = verts[edges[:, 1]]
        d0 = (p0 - origin) @ normal      # signed distances
        d1 = (p1 - origin) @ normal
        
        # Check if edges cross the plane
        crosses_plane = np.logical_and(d0 * d1 < -tol, np.abs(d0 - d1) > tol)
        
        # Only process edges that cross the plane
        valid_edges = np.where(crosses_plane)[0]
        p0_valid = p0[valid_edges]
        p1_valid = p1[valid_edges]
        d0_valid = d0[valid_edges]
        d1_valid = d1[valid_edges]
        
        # Compute intersection point for each valid edge
        t = d0_valid / (d0_valid - d1_valid)  # shape (N,)
        intersection_points = p0_valid + (p1_valid - p0_valid) * t[:, None]  # shape (N, 3)
        
        # Compute perpendicular distance from each intersection point to the gaze vector line
        # The gaze vector line passes through gaze_origin in direction gaze_vector
        # Distance from point P to line through O with direction D: ||(P - O) - dot(P - O, D) * D||
        # Vectorized computation for all intersection points at once
        if len(intersection_points) > 0:
            # Vector from line origin to each point: shape (N, 3)
            vecs = intersection_points - gaze_origin
            
            # Project each vector onto line direction (gaze_vector is already normalized)
            # Shape: (N,)
            proj_lengths = vecs @ gaze_vector
            
            # Closest point on line for each intersection point: shape (N, 3)
            closest_points = gaze_origin + proj_lengths[:, None] * gaze_vector
            
            # Perpendicular distance from each point to line: shape (N,)
            distances = np.linalg.norm(intersection_points - closest_points, axis=1)
            
            # Filter by height_limit - strictly enforce the limit
            within_height = distances <= height_limit
        else:
            distances = np.array([])
            within_height = np.array([], dtype=bool)
        
        # Build final mask for all edges
        mask = np.zeros(edges.shape[0], dtype=bool)
        if len(valid_edges) > 0:
            mask[valid_edges] = within_height
        
        return mask

    def highlight_selected_points():
        # OPTIMIZED: Remove old highlights (now just one actor instead of many)
        try:
            plotter.remove_actor('selected_points_highlight')
        except Exception:
            pass
        
        # OPTIMIZED: Create a single actor for all selected points instead of one per point
        if selected_points:
            # Convert selected coordinates to array and find matching cluster means
            selected_coords = np.array(list(selected_points))
            selected_indices = []
            
            # Vectorized: compute all distances at once
            for coord in selected_coords:
                dists = np.linalg.norm(cluster_means - coord, axis=1)
                closest_idx = int(np.argmin(dists))
                if dists[closest_idx] < point_selection_threshold:
                    selected_indices.append(closest_idx)
            
            # Create single actor for all selected points
            if selected_indices:
                selected_positions = cluster_means[selected_indices]
                plotter.add_points(selected_positions, color='magenta', point_size=selected_point_size, 
                                 render_points_as_spheres=True, name='selected_points_highlight', 
                                 opacity=1.0, lighting=False, pickable=False)

    def find_closest_edge(point, current_edges, cluster_means):
        min_edge_dist = float('inf')
        closest_edge_idx = -1
        for i in range(len(current_edges)):
            edge = current_edges[i]
            v1, v2 = cluster_means[edge[0]], cluster_means[edge[1]]
            edge_vec = v2 - v1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0:
                continue
            edge_vec_normalized = edge_vec / edge_len
            point_vec = point - v1
            t = np.dot(point_vec, edge_vec_normalized)
            t = max(0, min(1, t))
            closest_point = v1 + t * edge_vec
            dist = np.linalg.norm(point - closest_point)
            if dist < min_edge_dist:
                min_edge_dist = dist
                closest_edge_idx = i
        return closest_edge_idx, min_edge_dist

    def on_pick_unified(point):
        """Unified picking callback that handles points only (edge selection disabled)"""
        if point is None or len(point) == 0:
            return
        
        # Point selection only
        if len(cluster_means) > 0:
            dists = np.linalg.norm(cluster_means - point, axis=1)
            min_dist_idx = int(np.argmin(dists))
            min_dist = dists[min_dist_idx]
            if min_dist < point_selection_threshold:  # Threshold for point selection
                point_coord = tuple(cluster_means[min_dist_idx])
                if point_coord in selected_points:
                    selected_points.remove(point_coord)
                else:
                    selected_points.add(point_coord)
                highlight_selected_points()
                point_color = cluster_mean_colors[min_dist_idx]
                print(f"Point selected: distance={min_dist:.3f} (threshold={point_selection_threshold}), color={point_color}")
            else:
                print(f"No point near click: closest distance={min_dist:.3f} (threshold={point_selection_threshold})")
        else:
            print("No cluster means available for selection")

    # Enable unified picking
    plotter.enable_point_picking(callback=on_pick_unified, show_message=True)
    
    def on_clear_selection():
        """Clear all selected points and edges"""
        nonlocal selected_points, selected_edge_mask, plane_edge_indices
        selected_points.clear()
        selected_edge_mask.fill(False)
        highlight_selected_points()
        
        # Remove cyan plane visualization
        try:
            plotter.remove_actor('cyan_edges')
            plotter.remove_actor('intersection_points')
            plotter.remove_actor('height_limit_sphere')
            plotter.remove_actor('selected_plane')
        except Exception:
            pass
        
        # Reset plane state
        plane_defined['done'] = False
        plane_edge_indices = []
        
        print("Selection cleared.")

    def on_rectangle_select():
        """Use rectangle selection to select multiple points"""
        nonlocal cluster_means, selected_points
        
        # Get camera position
        camera_pos = plotter.camera_position
        
        # Use rectangle selection
        selected_indices = select_points_rectangle_mode(cluster_means, camera_pos)
        
        if selected_indices:
            # Convert indices to coordinates for selected_points set
            for idx in selected_indices:
                point_coord = tuple(cluster_means[idx])
                selected_points.add(point_coord)
            
            # Update visualization
            highlight_selected_points()
            print(f"Added {len(selected_indices)} points to selection")
        else:
            print("No points selected")

    def on_delete_selected():
        nonlocal cluster_means, current_edges, current_edge_mesh, selected_points, selected_edge_mask, cluster_mean_colors
        
        if not selected_points and not np.any(selected_edge_mask):
            print('No points or edges selected to delete.')
            return
        
        deleted_points = len(selected_points)
        deleted_edges = np.sum(selected_edge_mask)
        
        # Process edge deletion FIRST (before point deletion changes edge indices)
        if np.any(selected_edge_mask) and current_edges is not None:
            plotter.remove_actor('edges')
            plotter.remove_actor('cyan_edges')
            
            # Keep only non-selected edges
            keep_edge_mask = ~selected_edge_mask
            valid_edges = current_edges[keep_edge_mask]
            
            if len(valid_edges) > 0:
                lines = np.hstack([np.full((valid_edges.shape[0], 1), 2), valid_edges])
                edge_mesh = pv.PolyData(cluster_means, lines=lines)
                plotter.add_mesh(edge_mesh, color='red', line_width=edge_line_width, style='surface', name='edges', reset_camera=False)
                current_edges = valid_edges.copy()
                current_edge_mesh = edge_mesh
                
                # Update edge masks - all edges are now unselected after deletion
                edge_mask = np.zeros(len(valid_edges), bool)
                selected_edge_mask = np.zeros(len(valid_edges), bool)
            else:
                current_edges = None
                current_edge_mesh = None
                edge_mask = np.zeros(0, bool)
                selected_edge_mask = np.zeros(0, bool)
        else:
            pass  # No edges to delete or no edges available
        
        # Remove selected cluster means
        if selected_points:
            # OPTIMIZED: Build keep_mask efficiently using vectorized operations
            keep_mask = np.ones(len(cluster_means), dtype=bool)
            
            # Convert selected_points to numpy array for vectorized distance calculation
            selected_coords = np.array(list(selected_points))
            
            # For each selected point, find the closest cluster mean and mark for deletion
            # Use broadcasting to compute all distances at once
            for coord in selected_coords:
                dists = np.linalg.norm(cluster_means - coord, axis=1)
                closest_idx = int(np.argmin(dists))
                keep_mask[closest_idx] = False
            
            cluster_means = cluster_means[keep_mask]
            
            # Filter cluster mean colors to match remaining points
            if cluster_mean_colors is not None:
                cluster_mean_colors = cluster_mean_colors[keep_mask]
            
            # Clear point selection
            selected_points.clear()
            try:
                plotter.remove_actor('selected_points_highlight')
            except Exception:
                pass
            
            # OPTIMIZED: Remove cluster means actor (now always a single actor)
            try:
                plotter.remove_actor('cluster_means')
            except Exception:
                pass
            
            # Redraw cluster means with preserved colors (if available)
            if len(cluster_means) > 0:
                if cluster_mean_colors is not None and len(cluster_mean_colors) == len(cluster_means):
                    # Use preserved colors
                    point_cloud = pv.PolyData(cluster_means)
                    point_cloud['colors'] = (cluster_mean_colors * 255).astype(np.uint8)
                    plotter.add_mesh(point_cloud, scalars='colors', rgb=True, point_size=cluster_mean_point_size,
                                    render_points_as_spheres=True, name='cluster_means', style='points')
                else:
                    # Fallback to red if colors not available
                    plotter.add_points(cluster_means, color='red', point_size=cluster_mean_point_size, 
                                      render_points_as_spheres=True, name='cluster_means')
            
            # OPTIMIZED: Vectorized edge filtering
            if current_edges is not None:
                plotter.remove_actor('edges')
                
                # Create mapping from old indices to new indices using vectorized operations
                old_to_new = np.full(len(keep_mask), -1, dtype=np.int32)
                old_to_new[keep_mask] = np.arange(np.sum(keep_mask))
                
                # Vectorized edge filtering: check if both vertices are kept
                v1_kept = old_to_new[current_edges[:, 0]]
                v2_kept = old_to_new[current_edges[:, 1]]
                valid_edge_mask = (v1_kept >= 0) & (v2_kept >= 0)
                
                if np.any(valid_edge_mask):
                    # Update edge indices to new vertex indices
                    valid_edges = np.column_stack([v1_kept[valid_edge_mask], v2_kept[valid_edge_mask]])
                    
                    lines = np.hstack([np.full((valid_edges.shape[0], 1), 2), valid_edges])
                    edge_mesh = pv.PolyData(cluster_means, lines=lines)
                    plotter.add_mesh(edge_mesh, color='red', line_width=edge_line_width, style='surface', name='edges', reset_camera=False)
                    current_edges = valid_edges.copy()
                    current_edge_mesh = edge_mesh
                    
                    # Update edge masks
                    edge_mask = np.zeros(len(valid_edges), bool)
                    selected_edge_mask = np.zeros(len(valid_edges), bool)
                else:
                    current_edges = None
                    current_edge_mesh = None
                    edge_mask = np.zeros(0, bool)
                    selected_edge_mask = np.zeros(0, bool)
        
        plotter.render()
        print(f'Deleted {deleted_points} points and {deleted_edges} edges. {len(cluster_means)} points remaining.')

    def on_define_plane():
        """Define plane based on camera position"""
        nonlocal edge_mask, selected_edge_mask, plane_edge_indices
        if current_edges is None or len(current_edges) == 0:
            print("No edges available to select.")
            return
        
        origin, normal, up_vec = get_plane_from_camera(plotter.camera_position)
        plane_geom['n'] = normal
        
        # edge mask with height limit - use cluster_means for vertex positions
        cam_pos = np.asarray(plotter.camera_position[0])
        cam_focal = np.asarray(plotter.camera_position[1])
        gaze_vec = cam_focal - cam_pos
        
        mask = edges_intersect_plane(plotter, current_edges, cluster_means, origin, normal, plane_height_limit, cam_focal, gaze_vec)
        
        if not mask.any():
            print("No edges selected. Try different camera view or press 'L'.")
            return
        
        # Ensure edge_mask and selected_edge_mask are the correct size
        edge_mask = np.zeros(len(current_edges), bool)
        selected_edge_mask = np.zeros(len(current_edges), bool)
        
        edge_mask[:] = mask
        selected_edge_mask.fill(False)  # Clear previous edge selection
        
        # Store indices of edges that intersect the plane
        plane_edge_indices = np.where(mask)[0].tolist()
        
        # Remove previous plane edge visualization
        try:
            plotter.remove_actor('cyan_edges')
            plotter.remove_actor('intersection_points')
            plotter.remove_actor('height_limit_sphere')
            plotter.remove_actor('selected_plane')
        except Exception:
            pass
        
        # Automatically select all intersecting edges when plane is defined
        # Use the mask that was already correctly computed by edges_intersect_plane
        selected_edge_mask[:] = mask
        
        # Highlight selected edges as cyan (using the correct mask from edges_intersect_plane)
        selected_edge_indices = np.where(mask)[0]
        if len(selected_edge_indices) > 0:
            cyan_lines = np.hstack([np.full((len(selected_edge_indices), 1), 2),
                                    current_edges[selected_edge_indices]])
            cyan_mesh = pv.PolyData(cluster_means, lines=cyan_lines)
            plotter.add_mesh(cyan_mesh, color='cyan', line_width=cyan_edge_line_width, name='cyan_edges')
        plane_defined['done'] = True

    def on_define_plane_simple():
        """Simplified plane selection - select all edges that cross the plane"""
        nonlocal edge_mask, selected_edge_mask, plane_edge_indices
        if current_edges is None or len(current_edges) == 0:
            print("No edges available to select.")
            return
        
        origin, normal, up_vec = get_plane_from_camera(plotter.camera_position)
        plane_geom['n'] = normal
        
        # Simple approach: just check which edges cross the plane
        p0 = cluster_means[current_edges[:, 0]]
        p1 = cluster_means[current_edges[:, 1]]
        d0 = (p0 - origin) @ normal
        d1 = (p1 - origin) @ normal
        
        # Edges cross the plane if the distances have opposite signs
        crosses_plane = (d0 * d1) < 0
        
        if not np.any(crosses_plane):
            print("No edges cross plane. Try different camera view.")
            return
        
        # Update masks
        edge_mask = np.zeros(len(current_edges), bool)
        selected_edge_mask = np.zeros(len(current_edges), bool)
        
        edge_mask[:] = crosses_plane
        selected_edge_mask[:] = crosses_plane
        plane_edge_indices = np.where(crosses_plane)[0].tolist()
        
        # Remove previous visualizations
        try:
            plotter.remove_actor('cyan_edges')
            plotter.remove_actor('intersection_points')
            plotter.remove_actor('height_limit_sphere')
            plotter.remove_actor('selected_plane')
        except Exception:
            pass
        
        # Highlight all crossing edges in cyan
        if np.any(crosses_plane):
            crossing_edges = current_edges[crosses_plane]
            cyan_lines = np.hstack([np.full((len(crossing_edges), 1), 2), crossing_edges])
            cyan_mesh = pv.PolyData(cluster_means, lines=cyan_lines)
            plotter.add_mesh(cyan_mesh, color='cyan', line_width=cyan_edge_line_width, name='cyan_edges')
        
        plane_defined['done'] = True
        
        print(f"Selected {np.sum(crosses_plane)} edges")

    def on_increase_height_limit():
        """Increase the height limit (make it more restrictive)"""
        nonlocal plane_height_limit_factor, plane_height_limit
        plane_height_limit_factor *= 0.5  # Make it smaller (more restrictive)
        # Recalculate the actual height limit
        bbox_min = np.min(cluster_means, axis=0)
        bbox_max = np.max(cluster_means, axis=0)
        bbox_dims = bbox_max - bbox_min
        plane_height_limit = np.max(bbox_dims) / plane_height_limit_factor
        print(f"Height limit: {plane_height_limit:.6f} (factor: {plane_height_limit_factor:.1f}, more restrictive)")

    def on_decrease_height_limit():
        """Decrease the height limit (make it less restrictive)"""
        nonlocal plane_height_limit_factor, plane_height_limit
        plane_height_limit_factor *= 2.0  # Make it larger (less restrictive)
        # Recalculate the actual height limit
        bbox_min = np.min(cluster_means, axis=0)
        bbox_max = np.max(cluster_means, axis=0)
        bbox_dims = bbox_max - bbox_min
        plane_height_limit = np.max(bbox_dims) / plane_height_limit_factor
        print(f"Height limit: {plane_height_limit:.6f} (factor: {plane_height_limit_factor:.1f}, less restrictive)")

    def on_save_state():
        """Save the current clustering state"""
        try:
            if experiment_name is not None and output_dir is not None:
                # CRITICAL: Use true_original for saving, not base state
                # This preserves the original clustering across multiple editing sessions
                state_filename = save_clustering_state(
                    cluster_means, current_edges, 
                    true_original_cluster_means_for_save, true_original_edges_for_save, 
                    experiment_name, output_dir
                )
            else:
                print("Cannot save state: experiment_name or output_dir not provided")
        except Exception as e:
            print(f"Error saving state: {e}")
            import traceback
            traceback.print_exc()

    # Add key events
    plotter.add_key_event('d', on_delete_selected)
    plotter.add_key_event('D', on_delete_selected)
    plotter.add_key_event('c', on_clear_selection)
    plotter.add_key_event('C', on_clear_selection)
    plotter.add_key_event('x', on_define_plane)
    plotter.add_key_event('X', on_define_plane)
    plotter.add_key_event('s', on_save_state)
    plotter.add_key_event('S', on_save_state)
    plotter.add_key_event('r', on_rectangle_select)
    plotter.add_key_event('R', on_rectangle_select)
    plotter.add_key_event('p', on_define_plane_simple)
    plotter.add_key_event('P', on_define_plane_simple)
    plotter.add_key_event('h', on_increase_height_limit)
    plotter.add_key_event('H', on_increase_height_limit)
    plotter.add_key_event('l', on_decrease_height_limit)
    plotter.add_key_event('L', on_decrease_height_limit)
    
    # Add space bar event to update camera position in config
    def on_update_camera_config():
        """Update the configuration with current camera position"""
        camera_pos = plotter.camera_position
        
        # Convert numpy arrays/types to lists for JSON serialization
        camera_pos_json = []
        for p in camera_pos:
            if hasattr(p, 'tolist'):
                camera_pos_json.append(p.tolist())
            else:
                camera_pos_json.append(list(p))
                
        print("\n" + "="*50)
        print("📷 UPDATING CAMERA POSITION")
        print("="*50)
        
        if config_manager is not None:
            config_manager.set('camera.position', camera_pos_json)
            config_manager.save_config()
            print(f"✅ Camera position updated in config: {config_manager.config_path}")
            print(f"New position: {camera_pos_json}")
        else:
            print("❌ Error: config_manager not available to save camera position")
        print("="*50)
    
    plotter.add_key_event('space', on_update_camera_config)
    
    # Set the camera position
    if camera_position:
        plotter.camera_position = camera_position
        print(f"Set camera position from configuration")
    else:
        print("No camera position found in configuration, using default")
    
    return plotter

def run_mesh_editor(config_manager):
    """
    Run the interactive mesh editor using the provided configuration manager.
    
    Args:
        config_manager: Configuration manager instance
    """
    config = config_manager.config
    
    try:
        # Load data based on experiment configuration
        dataset, data_path = load_experiment_data(config_manager.config)
        
        if dataset is not None:
            # Use the first time point for the example
            frame = dataset[0]          # (N, 6)
            xyz = frame[:, :3]
            rgb = np.clip(frame[:, 3:], 0.0, 1.0)
            
            # --- Color-then-spatial clustering ---
            # Clustering with color information
            color_labels, color_centroids = color_then_spatial_clustering(xyz, rgb, config_manager.config)
            color_cluster_means, color_cluster_points = calculate_cluster_means(xyz, color_labels)
            
            # Get experiment info
            experiment_name = config['experiment']['name']
            output_dir = config['experiment']['output_dir']
            
            # Check for existing clustering state
            existing_state_file = find_clustering_state_for_experiment(experiment_name, output_dir)
            
            # Only remove white cluster means if:
            # 1. The flag is enabled in config, AND
            # 2. No existing state file exists (i.e., this is the first time editing)
            # If a state file exists, the user has already manually curated the clustering
            if config_manager.get('clustering.remove_white_clusters', False) and not existing_state_file:
                color_cluster_means, color_cluster_points, kept_indices = remove_white_cluster_means(
                    color_cluster_means, color_cluster_points, xyz, rgb,
                    white_threshold=config_manager.get('clustering.white_threshold', 0.95)
                )
            
            # ALWAYS track the TRUE original clustering state (before any edits)
            true_original_cluster_means = color_cluster_means.copy()
            true_original_edges = None  # Will be set when edges are created
            
            # Track the "base" state for editing (either initial or loaded)
            base_cluster_means = color_cluster_means.copy()
            base_edges = None  # Track original edges from loaded state
            
            if existing_state_file:
                # Use utility function to load saved state
                saved_final_means, saved_final_edges, saved_original_means, saved_original_edges = load_saved_clustering_state(
                    experiment_name, output_dir
                )
                
                if saved_final_means is not None:
                    # Use the saved final state directly
                    color_cluster_means = saved_final_means
                    print(f"Loaded saved cluster means: {len(color_cluster_means)} cluster means")
                    
                    # Set base state for this editing session
                    base_cluster_means = color_cluster_means.copy()
                    base_edges = saved_final_edges.copy() if saved_final_edges is not None else None
                    print(f"Base state: {len(base_cluster_means)} cluster means, {len(base_edges) if base_edges is not None else 0} edges")
                    
                    # Set true original from saved state (for saving later)
                    if saved_original_means is not None:
                        true_original_cluster_means = saved_original_means
                        print(f"True original: {len(true_original_cluster_means)} cluster means")
                    
                    if saved_original_edges is not None:
                        true_original_edges = saved_original_edges
                        print(f"True original edges: {len(true_original_edges)} edges")
                else:
                    print("Could not load saved state, using original clustering")
            else:
                print(f"\nNo existing clustering state found for experiment: {experiment_name}")
                print("Starting with original clustering (will create new state when saved)")
            
            # Create interactive PyVista visualization
            plotter = create_interactive_colored_visualization(xyz, color_labels, color_cluster_means, rgb, color_centroids, 
                                                             f"Interactive Clustering - {experiment_name}", 
                                                             experiment_name=experiment_name, 
                                                             output_dir=output_dir, 
                                                             config_manager=config_manager,
                                                             base_cluster_means=base_cluster_means,
                                                             base_edges=base_edges,
                                                             true_original_cluster_means=true_original_cluster_means,
                                                             true_original_edges=true_original_edges)

            # Show instructions
            print("\n=== INTERACTIVE CONTROLS ===")
            print("• Click near cluster means: Select/deselect individual cluster means")
            print("• Press 'R' key: Rectangle selection mode (select multiple points at once)")
            print("• Selected cluster means: Highlighted in magenta with larger size")
            print("• Press 'X' key: Define plane based on camera position (complex, with height limit)")
            print("• Press 'P' key: Simple plane selection (all edges crossing the plane)")
            print("• Press 'H' key: Increase height limit (more restrictive)")
            print("• Press 'L' key: Decrease height limit (less restrictive)")
            print("• Press 'D' key: Remove all selected cluster means AND edges")
            print("• Press 'C' key: Clear current selection (both points and edges)")
            print("• Press 'S' key: Save current clustering state (for use in other scripts)")
            print("• Press SPACE: Save current camera position to the configuration file")
            print("• Mouse: Rotate, zoom, and pan the view")
            print("• Console: Shows selection feedback and distances")
            print("• Startup: Choose to load existing state or start fresh")
            print("• RECOMMENDED: Use 'R' for rectangle selection - much faster than individual clicking!")
            print("• PLANE SELECTION: Try 'P' first (simple), then 'X' if you need height filtering")
            print("• HEIGHT LIMIT: If 'X' selects no edges, try 'L' to make it less restrictive")
            print("==============================\n")
            
            # Show visualization
            print("Showing interactive visualization...")
            plotter.show()
        else:
            print("No dataset loaded.")
    except Exception as e:
        print(f"Error in mesh editor: {e}")
        import traceback
        traceback.print_exc() 