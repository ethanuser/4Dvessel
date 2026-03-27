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
    select_state_file
)
from utils.rectangle_select_points import select_points_with_rectangle

def load_experiment_data(config: dict):
    """Load data based on experiment configuration."""
    data_path = config['experiment']['data_path']
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    dataset = np.load(data_path)  # shape (V, N, 6)
    return dataset, data_path

def find_existing_clustering_state(experiment_name: str, output_dir: str) -> Optional[str]:
    """
    Find existing clustering state file for the given experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for the experiment
        
    Returns:
        Path to existing state file, or None if not found
    """
    # Look for clustering state in the experiment's processed directory
    state_filename = f"clustering_state_{experiment_name}.json"
    state_path = os.path.join(output_dir, state_filename)
    
    if os.path.exists(state_path):
        return state_path
    
    return None

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
        original_cluster_means: Original cluster means
        original_edges: Original edges
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
    print("\n=== RECTANGLE SELECTION MODE ===")
    print("Use rectangle selection to select multiple cluster means at once")
    print("This is more efficient than clicking individual points")
    print("=" * 50)
    
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
    plane_height_limit_factor = config_manager.get('visualization.plane_height_limit_factor', 80.0)
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
    if len(cluster_means) > 0:
        cluster_mean_colors = []
        for mean in cluster_means:
            # Find the closest points to this mean and get their average RGB
            distances = np.linalg.norm(points - mean, axis=1)
            closest_indices = np.argsort(distances)[:20]  # Get 20 closest points
            closest_rgb = rgb_values[closest_indices]
            avg_rgb = np.mean(closest_rgb, axis=0)
            cluster_mean_colors.append(avg_rgb)
        
        cluster_mean_colors = np.array(cluster_mean_colors)
        
        # Add cluster means with their corresponding average RGB colors
        if len(cluster_mean_colors) == 1:
            # Single color for all points
            plotter.add_points(cluster_means, color=cluster_mean_colors[0], point_size=cluster_mean_point_size, 
                              render_points_as_spheres=True, name='cluster_means')
        else:
            # Individual colors for each point - need to create separate point clouds
            for i, (mean, color) in enumerate(zip(cluster_means, cluster_mean_colors)):
                plotter.add_points(np.array([mean]), color=color, point_size=cluster_mean_point_size, 
                                  render_points_as_spheres=True, name=f'cluster_mean_{i}')
    
    # Create and plot the Delaunay edges
    if len(cluster_means) > 3:  # Need at least 4 points for 3D Delaunay
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
    selected_point_actors = {}  # actor handles for selected highlights

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

    def edges_intersect_plane(edges, verts, origin, normal, height_limit, gaze_origin, gaze_vector, tol=1e-9):
        """Boolean mask: True for edges that cross the plane and whose intersection point
        with the plane is within height_limit of the intersection of the camera gaze line."""
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
        
        print(f"  Edges crossing plane: {np.sum(crosses_plane)} out of {len(edges)}")
        
        # Only process edges that cross the plane
        valid_edges = np.where(crosses_plane)[0]
        p0_valid = p0[valid_edges]
        p1_valid = p1[valid_edges]
        d0_valid = d0[valid_edges]
        d1_valid = d1[valid_edges]
        
        # Debug: Show some example edges that cross the plane
        if len(valid_edges) > 0:
            print(f"  Example edges crossing plane (first 3):")
            for i in range(min(3, len(valid_edges))):
                edge_idx = valid_edges[i]
                print(f"    Edge {edge_idx}: {p0_valid[i]} -> {p1_valid[i]}")
                print(f"    Distances: d0={d0_valid[i]:.6f}, d1={d1_valid[i]:.6f}")
                print(f"    Edge length: {np.linalg.norm(p1_valid[i] - p0_valid[i]):.6f}")
        
        # Compute intersection point for each valid edge
        t = d0_valid / (d0_valid - d1_valid)  # shape (N,)
        intersection_points = p0_valid + (p1_valid - p0_valid) * t[:, None]  # shape (N, 3)
        
        print(f"  Computed {len(intersection_points)} intersection points")
        
        # Compute intersection of gaze line with plane
        denom = np.dot(gaze_vector, normal)
        if np.abs(denom) < tol:
            # Gaze is parallel to plane; use a different approach
            print(f"  WARNING: Gaze parallel to plane, using alternative approach")
            # Use the centroid of intersection points as the reference
            gaze_plane_point = np.mean(intersection_points, axis=0)
            print(f"  Using centroid of intersection points as reference: {gaze_plane_point}")
        else:
            t_gaze = np.dot((origin - gaze_origin), normal) / denom
            gaze_plane_point = gaze_origin + t_gaze * gaze_vector
            print(f"  Gaze-plane intersection point: {gaze_plane_point}")
        
        # Compute distance from each intersection point to gaze_plane_point
        distances = np.linalg.norm(intersection_points - gaze_plane_point, axis=1)
        within_height = distances <= height_limit
        
        print(f"  Distance statistics:")
        print(f"    Min distance: {np.min(distances):.6f}")
        print(f"    Max distance: {np.max(distances):.6f}")
        print(f"    Mean distance: {np.mean(distances):.6f}")
        print(f"    Height limit: {height_limit:.6f}")
        print(f"    Edges within height limit: {np.sum(within_height)}")
        
        # Show some example distances for debugging
        if len(distances) > 0:
            sorted_indices = np.argsort(distances)
            print(f"  Closest 5 distances: {distances[sorted_indices[:5]]}")
            print(f"  Farthest 5 distances: {distances[sorted_indices[-5:]]}")
            
            # If no edges pass the height limit, try adaptive approach
            if not np.any(within_height):
                print(f"  WARNING: No edges pass height limit. Trying adaptive approach...")
                # Use a more reasonable limit based on the distance distribution
                adaptive_limit = np.percentile(distances, 50)  # Use median distance
                within_height = distances <= adaptive_limit
                print(f"  Adaptive limit (median): {adaptive_limit:.6f}")
                print(f"  Edges with adaptive limit: {np.sum(within_height)}")
                
                # If still no edges, use a more generous limit
                if not np.any(within_height):
                    adaptive_limit = np.percentile(distances, 75)  # Use 75th percentile
                    within_height = distances <= adaptive_limit
                    print(f"  More generous limit (75th percentile): {adaptive_limit:.6f}")
                    print(f"  Edges with generous limit: {np.sum(within_height)}")
        
        # Build final mask for all edges
        mask = np.zeros(edges.shape[0], dtype=bool)
        mask[valid_edges] = within_height
        return mask

    def highlight_selected_points():
        # Remove all previous highlights
        for actor in selected_point_actors.values():
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        selected_point_actors.clear()
        # Add new highlights
        for coord in selected_points:
            # Find the cluster mean that matches this coordinate
            coord_array = np.array(coord)
            dists = np.linalg.norm(cluster_means - coord_array, axis=1)
            closest_idx = int(np.argmin(dists))
            # Only highlight if it's the same point (within small tolerance)
            if dists[closest_idx] < point_selection_threshold:  # Small tolerance for floating point comparison
                actor = plotter.add_points(cluster_means[closest_idx][None, :], color='magenta', point_size=selected_point_size, render_points_as_spheres=True, name=f'selected_point_{coord}', opacity=1.0, lighting=False, pickable=False)
                selected_point_actors[coord] = actor

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
                print(f"Point selected: distance={min_dist:.3f} (threshold={point_selection_threshold})")
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
        
        # Remove all selected edge highlights
        for i in range(len(current_edges) if current_edges is not None else 0):
            try:
                plotter.remove_actor(f"sel_edge_{i}")
            except Exception:
                pass
        
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
            print(f"Rectangle selection: {len(selected_indices)} points selected")
        else:
            print("No points selected with rectangle")

    def on_delete_selected():
        nonlocal cluster_means, current_edges, current_edge_mesh, selected_points, selected_edge_mask
        
        if not selected_points and not np.any(selected_edge_mask):
            print('No points or edges selected to delete.')
            return
        
        deleted_points = len(selected_points)
        deleted_edges = np.sum(selected_edge_mask)
        
        # Process edge deletion FIRST (before point deletion changes edge indices)
        if np.any(selected_edge_mask) and current_edges is not None:
            plotter.remove_actor('edges')
            plotter.remove_actor('cyan_edges')  # <-- Add this line
            
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
            
            # Remove all selected edge highlights
            for i in range(len(current_edges) if current_edges is not None else 0):
                try:
                    plotter.remove_actor(f"sel_edge_{i}")
                except Exception:
                    pass
                    
            # Also remove any remaining selected edge highlights from the old edge set
            for i in range(100):  # Remove any remaining selected edge highlights
                try:
                    plotter.remove_actor(f"sel_edge_{i}")
                except Exception:
                    pass
        else:
            pass  # No edges to delete or no edges available
        
        # Remove selected cluster means
        if selected_points:
            keep_mask = np.ones(len(cluster_means), dtype=bool)
            for coord in selected_points:
                # Find the index of the cluster mean that matches the coordinate
                dists = np.linalg.norm(cluster_means - np.array(coord), axis=1)
                closest_idx = int(np.argmin(dists))
                keep_mask[closest_idx] = False
            
            cluster_means = cluster_means[keep_mask]
            
            # Clear point selection
            selected_points.clear()
            for actor in selected_point_actors.values():
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass
            selected_point_actors.clear()
            
            # Remove all cluster mean actors and redraw with simple red color
            for i in range(100):
                try:
                    plotter.remove_actor(f'cluster_mean_{i}')
                except Exception:
                    pass
            try:
                plotter.remove_actor('cluster_means')
            except Exception:
                pass
            
            # Redraw cluster means with simple red color (no color calculation)
            if len(cluster_means) > 0:
                plotter.add_points(cluster_means, color='red', point_size=cluster_mean_point_size, 
                                  render_points_as_spheres=True, name='cluster_means')
            
            # Efficiently update edges: only remove edges connected to deleted points
            if current_edges is not None:
                plotter.remove_actor('edges')
                
                # Create mapping from old indices to new indices
                old_to_new = {}
                new_idx = 0
                for old_idx in range(len(keep_mask)):
                    if keep_mask[old_idx]:
                        old_to_new[old_idx] = new_idx
                        new_idx += 1
                
                # Filter edges: keep only edges where both vertices are not deleted
                valid_edges = []
                for edge in current_edges:
                    v1, v2 = edge[0], edge[1]
                    # Check if both vertices of this edge are still present
                    if v1 in old_to_new and v2 in old_to_new:
                        # Update edge indices to new vertex indices
                        new_edge = [old_to_new[v1], old_to_new[v2]]
                        valid_edges.append(new_edge)
                
                valid_edges = np.array(valid_edges)
                
                if len(valid_edges) > 0:
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
        
        # Debug: Print camera information
        print(f"\n=== PLANE SELECTION DEBUG ===")
        print(f"Camera position: {plotter.camera_position[0]}")
        print(f"Camera focal: {plotter.camera_position[1]}")
        print(f"Camera up: {plotter.camera_position[2]}")
        print(f"Total edges: {len(current_edges)}")
        print(f"Plane height limit: {plane_height_limit}")
        
        origin, normal, up_vec = get_plane_from_camera(plotter.camera_position)
        plane_geom['n'] = normal
        
        print(f"Plane origin: {origin}")
        print(f"Plane normal: {normal}")
        
        # edge mask with height limit - use cluster_means for vertex positions
        cam_pos = np.asarray(plotter.camera_position[0])
        cam_focal = np.asarray(plotter.camera_position[1])
        gaze_vec = cam_focal - cam_pos
        print(f"Gaze vector: {gaze_vec}")
        
        mask = edges_intersect_plane(current_edges, cluster_means, origin, normal, plane_height_limit, cam_focal, gaze_vec)
        print(f"Edges intersecting plane: {np.sum(mask)} out of {len(current_edges)}")
        
        if not mask.any():
            print("No edges intersect this plane within height limit.")
            print("Try adjusting your camera view or the height limit.")
            print("=" * 50)
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
        
        # Get ALL edges that intersect the plane (not just those within height_limit)
        all_intersecting_mask = np.zeros(len(current_edges), bool)
        p0 = cluster_means[current_edges[:, 0]]
        p1 = cluster_means[current_edges[:, 1]]
        d0 = (p0 - origin) @ normal
        d1 = (p1 - origin) @ normal
        all_intersecting_mask = np.logical_and(d0 * d1 < -1e-9, np.abs(d0 - d1) > 1e-9)
        
        # Compute and visualize intersection points
        valid_edges = np.where(all_intersecting_mask)[0]
        p0_valid = p0[valid_edges]
        p1_valid = p1[valid_edges]
        d0_valid = d0[valid_edges]
        d1_valid = d1[valid_edges]
        
        # Compute intersection points
        t = d0_valid / (d0_valid - d1_valid)
        intersection_points = p0_valid + (p1_valid - p0_valid) * t[:, None]
        
        # Compute gaze vector line on the plane
        # Since normal is perpendicular to gaze_vec by definition, we can always find intersection
        t_gaze = np.dot((origin - cam_focal), normal) / np.dot(gaze_vec, normal)
        gaze_plane_point = cam_focal + t_gaze * gaze_vec
        
        # Compute perpendicular distance from each intersection point to the gaze vector line
        # The gaze vector line on the plane is: gaze_plane_point + s * gaze_vector_projected
        # We need to project gaze_vector onto the plane
        gaze_projected = gaze_vec - np.dot(gaze_vec, normal) * normal
        gaze_projected = gaze_projected / np.linalg.norm(gaze_projected)
        
        print(f"  Gaze projected onto plane: {gaze_projected}")
        print(f"  Gaze projected magnitude: {np.linalg.norm(gaze_projected):.6f}")
        
        # For each intersection point, compute perpendicular distance to the gaze line
        perpendicular_distances = []
        for i, point in enumerate(intersection_points):
            # Vector from gaze_plane_point to intersection point
            vec_to_point = point - gaze_plane_point
            # Project this vector onto the gaze line direction
            proj_length = np.dot(vec_to_point, gaze_projected)
            # Closest point on the gaze line
            closest_point = gaze_plane_point + proj_length * gaze_projected
            # Perpendicular distance
            perp_dist = np.linalg.norm(point - closest_point)
            perpendicular_distances.append(perp_dist)
            
            # Debug first few points
            if i < 3:
                print(f"    Point {i}: {point}")
                print(f"    Vec to point: {vec_to_point}")
                print(f"    Proj length: {proj_length:.6f}")
                print(f"    Closest point: {closest_point}")
                print(f"    Perpendicular distance: {perp_dist:.6f}")
        
        perpendicular_distances = np.array(perpendicular_distances)
        
        # Select edges within height_limit of the gaze line
        within_band = perpendicular_distances <= plane_height_limit
        
        print(f"  Edges within height limit: {np.sum(within_band)} out of {len(valid_edges)}")
        if len(perpendicular_distances) > 0:
            print(f"  Min distance: {np.min(perpendicular_distances):.6f}")
            print(f"  Max distance: {np.max(perpendicular_distances):.6f}")
            print(f"  Mean distance: {np.mean(perpendicular_distances):.6f}")
            print(f"  Height limit: {plane_height_limit:.6f}")
            print(f"  Percentile 10%: {np.percentile(perpendicular_distances, 10):.6f}")
            print(f"  Percentile 25%: {np.percentile(perpendicular_distances, 25):.6f}")
            print(f"  Percentile 50%: {np.percentile(perpendicular_distances, 50):.6f}")
            
            # If no edges pass the height limit, try a more reasonable limit
            if not np.any(within_band):
                print("  WARNING: No edges pass height limit. Trying adaptive limit...")
                # Use 25th percentile as a more reasonable limit
                adaptive_limit = np.percentile(perpendicular_distances, 25)
                within_band = perpendicular_distances <= adaptive_limit
                print(f"  Adaptive limit: {adaptive_limit:.6f}")
                print(f"  Edges with adaptive limit: {np.sum(within_band)}")
        
        # Update the mask to only include edges within the band
        mask = np.zeros(len(current_edges), bool)
        mask[valid_edges[within_band]] = True
        
        # Update edge_mask and plane_edge_indices with the final mask
        edge_mask[:] = mask
        plane_edge_indices = np.where(mask)[0].tolist()
        
        # Automatically select all intersecting edges when plane is defined
        selected_edge_mask[:] = mask
        
        # Highlight edges within height limit as cyan
        cyan_edges = valid_edges[within_band]
        if len(cyan_edges) > 0:
            cyan_lines = np.hstack([np.full((len(cyan_edges), 1), 2),
                                    current_edges[cyan_edges]])
            cyan_mesh = pv.PolyData(cluster_means, lines=cyan_lines)
            plotter.add_mesh(cyan_mesh, color='cyan', line_width=cyan_edge_line_width, name='cyan_edges')
        plane_defined['done'] = True
        
        print(f"Plane defined: {mask.sum()} intersecting edges selected (cyan). Click edges to toggle selection.")
        print(f"Plane edge indices: {len(plane_edge_indices)}")
        print(f"Selected edges: {np.sum(selected_edge_mask)}")
        print("=" * 50)

    def on_define_plane_simple():
        """Simplified plane selection - select all edges that cross the plane"""
        nonlocal edge_mask, selected_edge_mask, plane_edge_indices
        if current_edges is None or len(current_edges) == 0:
            print("No edges available to select.")
            return
        
        print(f"\n=== SIMPLE PLANE SELECTION ===")
        print(f"Total edges: {len(current_edges)}")
        
        origin, normal, up_vec = get_plane_from_camera(plotter.camera_position)
        plane_geom['n'] = normal
        
        print(f"Plane origin: {origin}")
        print(f"Plane normal: {normal}")
        
        # Simple approach: just check which edges cross the plane
        p0 = cluster_means[current_edges[:, 0]]
        p1 = cluster_means[current_edges[:, 1]]
        d0 = (p0 - origin) @ normal
        d1 = (p1 - origin) @ normal
        
        # Edges cross the plane if the distances have opposite signs
        crosses_plane = (d0 * d1) < 0
        print(f"Edges crossing plane: {np.sum(crosses_plane)} out of {len(current_edges)}")
        
        if not np.any(crosses_plane):
            print("No edges cross this plane. Try adjusting your camera view.")
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
        
        print(f"Simple plane selection: {np.sum(crosses_plane)} edges selected (cyan)")
        print("=" * 50)

    def on_increase_height_limit():
        """Increase the height limit factor for plane selection"""
        nonlocal plane_height_limit_factor
        plane_height_limit_factor *= 0.5  # Make it smaller (more restrictive)
        print(f"Increased height limit factor to: {plane_height_limit_factor}")
        print("Press 'X' to try plane selection again with new limit")

    def on_decrease_height_limit():
        """Decrease the height limit factor for plane selection"""
        nonlocal plane_height_limit_factor
        plane_height_limit_factor *= 2.0  # Make it larger (less restrictive)
        print(f"Decreased height limit factor to: {plane_height_limit_factor}")
        print("Press 'X' to try plane selection again with new limit")

    def on_save_state():
        """Save the current clustering state"""
        try:
            if experiment_name is not None and output_dir is not None:
                state_filename = save_clustering_state(cluster_means, current_edges, true_original_cluster_means_for_save, true_original_edges_for_save, experiment_name, output_dir)
                print(f"State saved! Use this file in other scripts: {state_filename}")
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
    
    # Add space bar event to print camera position
    def on_print_camera():
        """Print the current camera position"""
        camera_pos = plotter.camera_position
        print("\n" + "="*50)
        print("📷 CURRENT CAMERA POSITION")
        print("="*50)
        print("Copy this to your config file:")
        print('"camera": {')
        print('  "position": [')
        print(f'    [{camera_pos[0][0]}, {camera_pos[0][1]}, {camera_pos[0][2]}],')
        print(f'    [{camera_pos[1][0]}, {camera_pos[1][1]}, {camera_pos[1][2]}],')
        print(f'    [{camera_pos[2][0]}, {camera_pos[2][1]}, {camera_pos[2][2]}]')
        print('  ]')
        print('}')
        print("="*50)
    
    plotter.add_key_event('space', on_print_camera)
    
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
            existing_state_file = find_existing_clustering_state(experiment_name, output_dir)
            
            # ALWAYS track the TRUE original clustering state (before any edits)
            true_original_cluster_means = color_cluster_means.copy()
            true_original_edges = None  # Will be set when edges are created
            
            # Track the "base" state for editing (either initial or loaded)
            base_cluster_means = color_cluster_means.copy()
            base_edges = None  # Track original edges from loaded state
            
            if existing_state_file:
                print(f"\nFound existing clustering state: {existing_state_file}")
                print("Loading existing state for editing...")
                
                try:
                    # Load and apply existing state
                    state_data = load_clustering_state(existing_state_file)
                    filtered_cluster_means, filtered_edges, kept_point_indices, kept_edge_indices = apply_clustering_state(
                        color_cluster_means, None, state_data
                    )
                    
                    if len(filtered_cluster_means) != len(color_cluster_means):
                        print(f"Loaded existing state: {len(color_cluster_means)} -> {len(filtered_cluster_means)} cluster means")
                        color_cluster_means = filtered_cluster_means
                        # Update cluster_points to match the filtered means
                        color_cluster_points = [color_cluster_points[i] for i in kept_point_indices]
                        # IMPORTANT: Update base state to be the loaded state (not initial clustering)
                        base_cluster_means = color_cluster_means.copy()
                        base_edges = filtered_edges.copy() if filtered_edges is not None else None
                        print(f"Updated base state to loaded state: {len(base_cluster_means)} cluster means, {len(base_edges) if base_edges is not None else 0} edges")
                        
                        # IMPORTANT: Set true original edges from the loaded state data
                        if 'original_edges' in state_data and state_data['original_edges'] is not None:
                            true_original_edges = np.array(state_data['original_edges'])
                            print(f"Set true original edges from loaded state: {len(true_original_edges)} edges")
                        else:
                            true_original_edges = None
                            print("No true original edges found in loaded state")
                    else:
                        print("No filtering applied (using original clustering)")
                        
                except Exception as e:
                    print(f"Error loading existing state: {e}")
                    print("Starting with original clustering")
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
            print("• Press SPACE: Print current camera position (for config file)")
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