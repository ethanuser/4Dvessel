"""
Core clustering functionality using centralized configuration.
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay
from typing import Tuple, List

def color_then_spatial_clustering(xyz: np.ndarray, rgb: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform color-then-spatial clustering with deterministic results.
    
    Args:
        xyz: XYZ coordinates of points
        rgb: RGB color values of points
        config: Configuration dictionary containing clustering parameters
        
    Returns:
        Tuple of (labels, color_centroids)
    """
    # Extract parameters from config
    clustering_config = config.get('clustering', {})
    n_color_clusters = clustering_config.get('n_color_clusters')
    spatial_eps = clustering_config.get('spatial_eps')
    spatial_min_samples = clustering_config.get('min_cluster_points')
    random_seed = clustering_config.get('random_seed')
    
    print(f"Color clustering: {n_color_clusters} clusters")
    print(f"Spatial clustering: eps={spatial_eps}, min_samples={spatial_min_samples}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # 1. Color-based clustering (KMeans)
    kmeans = KMeans(n_clusters=n_color_clusters, random_state=random_seed)
    color_labels = kmeans.fit_predict(rgb)
    
    # Store the color centroids for later use
    color_centroids = kmeans.cluster_centers_
    print(f"Initial color clusters: {len(np.unique(color_labels))}")
    
    # 2. For each color cluster, spatial refinement
    final_labels = -np.ones(len(xyz), dtype=int)
    label_counter = 0
    
    # Sort color IDs to ensure deterministic processing order
    color_ids = np.unique(color_labels)
    color_ids = np.sort(color_ids)
    
    for color_id in color_ids:
        mask = (color_labels == color_id)
        if np.sum(mask) < spatial_min_samples:
            print(f"Skipping color cluster {color_id}: only {np.sum(mask)} points")
            continue  # skip tiny color clusters
        
        # Get points for this color cluster
        color_points = xyz[mask]
        color_indices = np.where(mask)[0]
        
        # Sort points to ensure deterministic DBSCAN results
        # Sort by x, then y, then z coordinates
        sort_indices = np.lexsort(color_points.T[::-1])
        sorted_points = color_points[sort_indices]
        sorted_indices = color_indices[sort_indices]
        
        # Apply DBSCAN to sorted points
        spatial_db = DBSCAN(eps=spatial_eps, min_samples=spatial_min_samples)
        spatial_labels = spatial_db.fit_predict(sorted_points)
        
        # Assign unique global labels
        for s in np.unique(spatial_labels):
            if s == -1:
                continue  # skip noise
            # Map back to original indices
            spatial_mask = (spatial_labels == s)
            original_indices = sorted_indices[spatial_mask]
            final_labels[original_indices] = label_counter
            label_counter += 1
    
    print(f"Final spatial clusters: {label_counter}")
    return final_labels, color_centroids

def calculate_cluster_means(points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Calculate cluster means and cluster points.
    
    Args:
        points: Input points
        labels: Cluster labels
        
    Returns:
        Tuple of (cluster_means, cluster_points)
    """
    unique_labels = set(labels)
    cluster_means = []
    cluster_points = []
    
    for k in unique_labels:
        if k != -1:  # Exclude noise points
            class_member_mask = (labels == k)
            cluster_mean = np.mean(points[class_member_mask], axis=0)
            cluster_means.append(cluster_mean)
            cluster_points.append(points[class_member_mask])
    
    return np.array(cluster_means), cluster_points

def create_delaunay_edges(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Delaunay]:
    """
    Create Delaunay edges with deterministic ordering.
    
    Args:
        points: Input points
        
    Returns:
        Tuple of (vertices, edges, triangulation)
    """
    tri = Delaunay(points)
    vertices = points
    edges = []
    
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edge = [simplex[i], simplex[j]]
                edges.append(edge)
    
    # Sort edges to ensure deterministic ordering
    edges = np.array(edges)
    # Sort each edge so smaller index comes first
    edges = np.sort(edges, axis=1)
    # Sort all edges lexicographically
    edges = edges[np.lexsort(edges.T[::-1])]
    
    print("Edges shape:", np.shape(edges), "Triangles shape:", np.shape(tri.simplices))
    return vertices, edges, tri

def remove_outlier_edges(vertices: np.ndarray, edges: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outlier edges based on length threshold.
    
    Args:
        vertices: Vertex positions
        edges: Edge indices
        config: Configuration dictionary
        
    Returns:
        Tuple of (valid_edges, initial_lengths)
    """
    clustering_config = config.get('clustering', {})
    threshold = clustering_config.get('edge_outlier_threshold', 0.25)
    
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    mean_length = np.mean(edge_lengths)
    std_length = np.std(edge_lengths)
    threshold_length = mean_length + threshold * std_length
    valid_indices = edge_lengths <= threshold_length
    valid_edges = edges[valid_indices]
    initial_lengths = edge_lengths[valid_indices]
    
    return valid_edges, initial_lengths 