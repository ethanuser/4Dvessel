"""
Clustering state utilities for loading and applying saved clustering states.
"""

import numpy as np
import tkinter as tk
from tkinter import filedialog
import json
from typing import Tuple, List, Optional, Dict
import os

def get_filtered_clustering_from_state(cluster_means: np.ndarray, experiment_name: str = None, output_dir: str = None) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Load and apply a saved clustering state to filter cluster means.
    
    Args:
        cluster_means: Original cluster means
        experiment_name: Name of the experiment (optional, for automatic state finding)
        output_dir: Output directory for the experiment (optional, for automatic state finding)
        
    Returns:
        Tuple of (filtered_cluster_means, filtered_edges, kept_point_indices, kept_edge_indices)
    """
    # If experiment info is provided, automatically find the state file
    if experiment_name and output_dir:
        state_filename = f"clustering_state_{experiment_name}.json"
        state_path = os.path.join(output_dir, state_filename)
        
        if os.path.exists(state_path):
            try:
                # Load state data
                with open(state_path, 'r') as f:
                    state_data = json.load(f)
                
                # Convert back to numpy arrays
                original_cluster_means = np.array(state_data['original_cluster_means'])
                final_cluster_means = np.array(state_data['final_cluster_means'])
                
                kept_point_indices = state_data['kept_point_indices']
                kept_edge_indices = state_data.get('kept_edge_indices', [])
                
                # Filter cluster means
                filtered_cluster_means = cluster_means[kept_point_indices]
                
                # Filter edges (if any)
                filtered_edges = None
                if state_data.get('original_edges') is not None and kept_edge_indices:
                    original_edges = np.array(state_data['original_edges'])
                    filtered_edges = original_edges[kept_edge_indices]
                
                print(f"Automatically loaded clustering state from: {state_path}")
                print(f"Original clusters: {len(cluster_means)}, Filtered clusters: {len(kept_point_indices)}")
                
                return filtered_cluster_means, filtered_edges, kept_point_indices, kept_edge_indices
                
            except Exception as e:
                print(f"Error loading state file: {e}")
                print("Using original clustering.")
                return cluster_means, None, list(range(len(cluster_means))), []
        else:
            print(f"No clustering state file found at: {state_path}")
            print("Using original clustering.")
            return cluster_means, None, list(range(len(cluster_means))), []
    
    # Fallback to manual selection if experiment info not provided
    # Prompt user to select state file
    root = tk.Tk()
    root.withdraw()
    state_file = filedialog.askopenfilename(
        title="Select clustering state file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    root.destroy()
    
    if not state_file:
        print("No state file selected. Using original clustering.")
        return cluster_means, None, list(range(len(cluster_means))), []
    
    try:
        # Load state data
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        
        # Convert back to numpy arrays
        original_cluster_means = np.array(state_data['original_cluster_means'])
        final_cluster_means = np.array(state_data['final_cluster_means'])
        
        kept_point_indices = state_data['kept_point_indices']
        kept_edge_indices = state_data.get('kept_edge_indices', [])
        
        # Filter cluster means
        filtered_cluster_means = cluster_means[kept_point_indices]
        
        # Filter edges (if any)
        filtered_edges = None
        if state_data.get('original_edges') is not None and kept_edge_indices:
            original_edges = np.array(state_data['original_edges'])
            filtered_edges = original_edges[kept_edge_indices]
        
        print(f"Loaded clustering state from: {state_file}")
        print(f"Original clusters: {len(cluster_means)}, Filtered clusters: {len(kept_point_indices)}")
        
        return filtered_cluster_means, filtered_edges, kept_point_indices, kept_edge_indices
        
    except Exception as e:
        print(f"Error loading state file: {e}")
        print("Using original clustering.")
        return cluster_means, None, list(range(len(cluster_means))), []

def select_state_file() -> Optional[str]:
    """
    Open a file dialog to select a clustering state file.
    
    Returns:
        Path to the selected state file, or None if no file was selected
    """
    root = tk.Tk()
    root.withdraw()
    state_file = filedialog.askopenfilename(
        title="Select clustering state file",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    root.destroy()
    return state_file if state_file else None

def load_clustering_state(state_filename: str) -> dict:
    """
    Load clustering state from a JSON file.
    
    Args:
        state_filename: Path to the state file
        
    Returns:
        State data dictionary
    """
    with open(state_filename, 'r') as f:
        state_data = json.load(f)
    
    # Convert back to numpy arrays
    state_data['original_cluster_means'] = np.array(state_data['original_cluster_means'])
    if state_data['original_edges'] is not None:
        state_data['original_edges'] = np.array(state_data['original_edges'])
    state_data['final_cluster_means'] = np.array(state_data['final_cluster_means'])
    if state_data['final_edges'] is not None:
        state_data['final_edges'] = np.array(state_data['final_edges'])
    
    return state_data

def apply_clustering_state(original_cluster_means: np.ndarray, original_edges: np.ndarray, 
                          state_data: dict) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Apply a saved clustering state to get the filtered cluster means and edges.
    
    Args:
        original_cluster_means: Original cluster means
        original_edges: Original edges
        state_data: State data dictionary
        
    Returns:
        Tuple of (filtered_cluster_means, filtered_edges, kept_point_indices, kept_edge_indices)
    """
    kept_point_indices = state_data['kept_point_indices']
    kept_edge_indices = state_data['kept_edge_indices']
    
    # Filter cluster means
    filtered_cluster_means = original_cluster_means[kept_point_indices]
    
    # Filter edges (if any)
    filtered_edges = None
    if original_edges is not None and kept_edge_indices:
        filtered_edges = original_edges[kept_edge_indices]
    
    return filtered_cluster_means, filtered_edges, kept_point_indices, kept_edge_indices

def find_clustering_state_for_experiment(experiment_name: str, output_dir: str) -> Optional[str]:
    """
    Find existing clustering state file for the given experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for the experiment
        
    Returns:
        Path to existing state file, or None if not found
    """
    # Look for clustering state in the experiment's processed directory
    # Try different possible filenames to handle cases where experiment_name might include .json
    # or where legacy files were saved as .json.json
    
    # 1. Base names to try
    base_names = [experiment_name]
    if experiment_name.endswith('.json'):
        base_names.append(experiment_name[:-5])
    
    # 2. Files to check for each base name
    for name in base_names:
        possible_filenames = [
            f"clustering_state_{name}.json",
            f"clustering_state_{name}.json.json"  # Legacy/double extension support
        ]
        
        for filename in possible_filenames:
            state_path = os.path.join(output_dir, filename)
            if os.path.exists(state_path):
                return state_path
    
    return None

def load_saved_clustering_state(experiment_name: str, output_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load saved clustering state and return the final cluster means and edges directly.
    
    This function loads the saved final_cluster_means and final_edges from the state file,
    which is the correct approach since clustering is non-deterministic and indices won't match.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for the experiment
        
    Returns:
        Tuple of (final_cluster_means, final_edges, original_cluster_means, original_edges)
        Returns (None, None, None, None) if no state found or error loading
    """
    state_path = find_clustering_state_for_experiment(experiment_name, output_dir)
    
    if state_path is None:
        print(f"No existing clustering state found for experiment: {experiment_name}")
        return None, None, None, None
    
    try:
        print(f"Found existing clustering state: {state_path}")
        print("Loading saved state...")
        
        # Load the saved state
        state_data = load_clustering_state(state_path)
        
        # Extract saved data
        saved_final_means = state_data.get('final_cluster_means')
        saved_final_edges = state_data.get('final_edges')
        saved_original_means = state_data.get('original_cluster_means')
        saved_original_edges = state_data.get('original_edges')
        
        if saved_final_means is None:
            print("Warning: No final_cluster_means in saved state")
            return None, None, None, None
        
        # Return as numpy arrays
        final_means = np.array(saved_final_means) if saved_final_means is not None else None
        final_edges = np.array(saved_final_edges) if saved_final_edges is not None else None
        original_means = np.array(saved_original_means) if saved_original_means is not None else None
        original_edges = np.array(saved_original_edges) if saved_original_edges is not None else None
        
        print(f"Successfully loaded saved clustering state: {len(final_means)} cluster means, {len(final_edges) if final_edges is not None else 0} edges")
        
        return final_means, final_edges, original_means, original_edges
        
    except Exception as e:
        print(f"Error loading clustering state: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def remap_labels_to_saved_means(xyz: np.ndarray, labels: np.ndarray, 
                                 original_cluster_means: np.ndarray, 
                                 saved_cluster_means: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Remap point labels to match saved cluster means.
    
    This function is needed because clustering is non-deterministic, so we need to match
    the newly computed clustering to the saved clustering by finding closest matches.
    
    Args:
        xyz: Point coordinates (N, 3)
        labels: Original cluster labels for each point (N,)
        original_cluster_means: Original cluster means from current clustering (M, 3)
        saved_cluster_means: Saved cluster means from state file (K, 3)
        
    Returns:
        Tuple of (updated_labels, updated_cluster_points)
    """
    print("Remapping labels to match saved cluster means...")
    
    # Create new labels array, starting with all points as noise
    updated_labels = np.full(len(xyz), -1, dtype=np.int32)
    
    # For each saved cluster mean, find all points that belong to it
    for new_idx in range(len(saved_cluster_means)):
        saved_mean = saved_cluster_means[new_idx]
        
        # Find the original cluster mean that's closest to this saved mean
        if len(original_cluster_means) > 0:
            dists_to_original = np.linalg.norm(original_cluster_means - saved_mean, axis=1)
            closest_original_idx = int(np.argmin(dists_to_original))
            
            # Assign all points from the original cluster to the new cluster
            updated_labels[labels == closest_original_idx] = new_idx
    
    # Create updated cluster_points list
    updated_cluster_points = []
    for new_idx in range(len(saved_cluster_means)):
        cluster_mask = updated_labels == new_idx
        cluster_pts = xyz[cluster_mask]
        if len(cluster_pts) > 0:
            updated_cluster_points.append(cluster_pts)
        else:
            # No points assigned to this cluster, use empty array
            updated_cluster_points.append(np.array([]).reshape(0, 3))
    
    print(f"Remapping complete:")
    print(f"  New unique labels: {np.unique(updated_labels[updated_labels != -1])}")
    print(f"  Cluster points: {len(updated_cluster_points)}")
    
    return updated_labels, updated_cluster_points 