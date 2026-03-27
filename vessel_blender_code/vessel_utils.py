from scipy.spatial import Delaunay
import sys
import matplotlib.pyplot as plt
import numpy as np

# Constants
COLORMAP = 'viridis' # 'jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
YOUNG_MODULUS_SILICON = 1.15e6  # 1.15 MPa in Pa
MAX_STRESS_PA = 0.15e6  # 2 MPa in Pa (default max stress for colorbars)
MAX_DISPLACEMENT_MM = 5.0  # 5 mm (default max displacement for colorbars)

# Calibration parameters for unit conversion (internal units -> mm)
CHESS_SQUARE_LENGTH_M = 0.035
CALIB_SCALING_FACTOR = 600.0/7.0
CORRECTION_FACTOR = 1.0 
UNIT_TO_MM = (CHESS_SQUARE_LENGTH_M / CALIB_SCALING_FACTOR) * 1000.0 * CORRECTION_FACTOR

def calculate_cluster_means(points: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Calculate cluster means from DBSCAN labels.
    
    Args:
        points: Array of point coordinates (N, 3)
        labels: Array of cluster labels from DBSCAN (N,)
        
    Returns:
        Tuple of (cluster_means, cluster_points_list)
        - cluster_means: Array of cluster mean positions (M, 3)
        - cluster_points_list: List of arrays, each containing points in a cluster
    """
    unique_labels = np.unique(labels)
    cluster_means = []
    cluster_points_list = []
    
    # Exclude noise points (label == -1)
    for label in unique_labels:
        if label == -1:
            continue
        
        cluster_mask = labels == label
        cluster_points = points[cluster_mask]
        
        if len(cluster_points) > 0:
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_means.append(cluster_mean)
            cluster_points_list.append(cluster_points)
    
    if len(cluster_means) == 0:
        return np.array([]).reshape(0, 3), []
    
    return np.array(cluster_means), cluster_points_list


def create_delaunay_edges(cluster_means: np.ndarray) -> tuple:
    """
    Create Delaunay triangulation edges from cluster means.
    
    Args:
        cluster_means: Array of cluster mean positions (N, 3)
        
    Returns:
        Tuple of (vertices, edges, simplices)
        - vertices: Same as cluster_means (for consistency with reference code)
        - edges: Array of edge indices (M, 2)
        - simplices: Array of simplex indices from Delaunay triangulation
    """
    if len(cluster_means) < 4:
        # Need at least 4 points for 3D Delaunay
        return cluster_means, np.array([]).reshape(0, 2), np.array([])
    
    # Perform 3D Delaunay triangulation
    tri = Delaunay(cluster_means)
    
    # Extract edges from simplices (tetrahedra in 3D)
    edges = set()
    for simplex in tri.simplices:
        # Each simplex has 4 vertices, create edges between all pairs
        for i in range(4):
            for j in range(i + 1, 4):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
    
    edges_array = np.array(list(edges))
    
    return cluster_means, edges_array, tri.simplices


def remove_outlier_edges(vertices: np.ndarray, edges: np.ndarray, 
                        outlier_threshold: float) -> tuple:
    """
    Remove outlier edges based on edge length.
    
    Args:
        vertices: Array of vertex positions (N, 3)
        edges: Array of edge indices (M, 2)
        outlier_threshold: Multiplier of median edge length to consider an edge an outlier
        
    Returns:
        Tuple of (valid_edges, removed_count)
        - valid_edges: Array of valid edge indices after outlier removal
        - removed_count: Number of edges removed
    """
    if len(edges) == 0:
        return edges, 0
    
    # Calculate edge lengths
    edge_lengths = []
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        length = np.linalg.norm(v2 - v1)
        edge_lengths.append(length)
    
    edge_lengths = np.array(edge_lengths)
    
    # Calculate median edge length
    median_length = np.median(edge_lengths)
    threshold_length = median_length * outlier_threshold
    
    # Filter edges
    valid_mask = edge_lengths <= threshold_length
    valid_edges = edges[valid_mask]
    removed_count = np.sum(~valid_mask)
    
    return valid_edges, removed_count


def extract_frame_data(coords, frame_numbers, times, frame_idx):
    """Extract data for a specific frame"""
    # Find all vertices that belong to this frame
    frame_mask = frame_numbers == frame_idx
    frame_coords = coords[frame_mask]
    frame_time = times[frame_idx] if frame_idx < len(times) else None
    return frame_coords, frame_time


def print_progress_bar(current, total, bar_length=40):
    """Print a simple text-based progress bar"""
    percent = float(current) / total
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write(f'\rLoading frames: [{hashes}{spaces}] {current}/{total} ({percent*100:.1f}%)')
    sys.stdout.flush()


def get_all_frames_data(coords, frame_numbers, times):
    """Get data organized by frames"""
    unique_frames = np.unique(frame_numbers)
    frames_data = {}
    total_frames = len(unique_frames)
    

    for idx, frame_idx in enumerate(unique_frames, 1):
        frame_coords, frame_time = extract_frame_data(coords, frame_numbers, times, frame_idx)
        frames_data[frame_idx] = {
            'coords': frame_coords,
            'time': frame_time,
            'num_vertices': len(frame_coords)
        }
        print_progress_bar(idx, total_frames)
    
    print()  # New line after progress bar
    
    return frames_data


def compute_edge_stress(
    cluster_positions: np.ndarray,
    edges: np.ndarray,
    initial_lengths: np.ndarray,
    young_modulus: float,
    poisson_ratio: float = 0.5
) -> np.ndarray:
    """
    Compute stress for each edge using a Neo-Hookean model.
    
    Args:
        cluster_positions: Cluster positions for one frame (K, 3)
        edges: Edge indices (E, 2)
        initial_lengths: Initial edge lengths (E,)
        young_modulus: Young's modulus in Pa
        poisson_ratio: Poisson's ratio (default 0.5 for incompressible)
        
    Returns:
        Array of stress values (E,) in Pa
    """
    if len(edges) == 0 or len(cluster_positions) == 0:
        return np.array([])
    
    # Calculate current edge lengths
    current_lengths = np.linalg.norm(
        cluster_positions[edges[:, 0]] - cluster_positions[edges[:, 1]], axis=1
    )
    
    # lam = current_length / initial_length
    # Avoid division by zero if initial_length is 0
    safe_initial = np.where(initial_lengths > 0, initial_lengths, 1.0)
    lam = current_lengths / safe_initial
    
    # mu = Shear modulus = E / (2 * (1 + nu))
    mu = young_modulus / (2 * (1 + poisson_ratio))
    
    # Neo-Hookean model from tools/compare_linear_vs_neohookean.py
    # sigma_neo = mu * (lam**2 - 1.0 / lam)
    stress = mu * (lam**2 - 1.0 / lam)
    
    # Handle the cases where initial_length was 0
    stress[initial_lengths <= 0] = 0
    
    return stress


def get_colormap():
    """
    Returns the viridis colormap.
    """
    return plt.get_cmap(COLORMAP)


def get_colors(values: np.ndarray, colormap, vmin: float = 0.0, vmax: float = None, use_abs: bool = False) -> np.ndarray:
    """
    Map values to RGB colors using a colormap.
    
    Args:
        values: Data values (N,)
        colormap: Matplotlib colormap object
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization (defaults to max(values))
        use_abs: Whether to use absolute values
        
    Returns:
        Array of RGB colors (N, 3)
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
        
    map_values = np.abs(values) if use_abs else values
    
    if vmax is None:
        vmax = np.max(map_values) + 1e-9
        
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    normalized_vals = norm(map_values)
    colors = colormap(normalized_vals)
    
    # Extract RGB channels only (ignore Alpha if present)
    return colors[:, 0:3]