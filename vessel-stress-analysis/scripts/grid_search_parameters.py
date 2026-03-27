# Use this to tune hyperparameters. 

import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay, distance
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
import time
import os
import datetime
import json

# =============================================================================
# USER CONFIGURATION SECTION - Modify these parameters as needed
# =============================================================================

# Material properties
YOUNG_MODULUS_SILICONE = 1.15e6  # Young's modulus for silicone in Pascals (Pa)

# Grid search parameters for regular DBSCAN (3D data)
EPS_VALUES = [0.05, 0.04, 0.03, 0.06]  # Expanded DBSCAN epsilon values
MIN_SAMPLES_VALUES = [2, 3, 4]    # Expanded DBSCAN min_samples values

# Grid search parameters for color clustering (6D data)
N_COLOR_CLUSTERS_VALUES = [2, 4, 6, 8]  # Expanded color clusters for KMeans

# Alternative clustering methods (set to True to enable)
USE_PURE_SPATIAL_DBSCAN = True    # Try pure spatial DBSCAN on 6D data (ignores color)
USE_PURE_COLOR_KMEANS = True       # Try pure color KMeans clustering
USE_COLOR_THEN_SPATIAL = True      # Use the original color-then-spatial approach

# Data preprocessing options
RGB_NORMALIZATION = 'clip'  # Options: 'clip', 'minmax', 'zscore', 'none'
RGB_CLIP_RANGE = (0.0, 1.0)  # Range for RGB clipping

# Clustering parameters (defaults used in functions)
DEFAULT_SPATIAL_EPS = 0.04
DEFAULT_SPATIAL_MIN_SAMPLES = 2
DEFAULT_N_COLOR_CLUSTERS = 5

# Edge filtering parameters
EDGE_FILTER_THRESHOLD = 0.25  # Threshold for removing outlier edges

# Stress visualization parameters
STRESS_NORM_MIN = -0.001e9  # Minimum stress for color normalization
STRESS_NORM_MAX = 0.001e9   # Maximum stress for color normalization

# =============================================================================
# END USER CONFIGURATION SECTION
# =============================================================================


def load_config_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    filepath = filedialog.askopenfilename(title="Select a config file", filetypes=[("JSON files", "*.json"), ("all files", "*.*")])
    root.destroy()  # Close the GUI
    if filepath:
        with open(filepath, 'r') as f:
            config = json.load(f)
        print("Loaded config file:", filepath)
        return config
    print("No config file selected.")
    return None

def load_data_from_config(config):
    """Load data using the config file"""
    data_path = config['experiment']['data_path']
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return None, None, None
    
    original_dataset = np.load(data_path)
    print("Loaded data shape:", original_dataset.shape)
    
    # Store original shape to detect if we had color information
    original_shape = original_dataset.shape
    
    # Handle both 3D and 6D data (with color information)
    if len(original_dataset.shape) == 3:
        if original_dataset.shape[2] == 6:
            # Extract only XYZ coordinates (first 3 dimensions)
            print("Detected 6D data with color information. Extracting XYZ coordinates only.")
            dataset = original_dataset[:, :, :3]
            print("Extracted XYZ data shape:", dataset.shape)
        elif original_dataset.shape[2] == 3:
            print("Detected 3D data. Using as is.")
            dataset = original_dataset
        else:
            print(f"Warning: Unexpected data shape {original_dataset.shape}. Using first 3 dimensions.")
            dataset = original_dataset[:, :, :3]
    else:
        print(f"Warning: Unexpected data shape {original_dataset.shape}. Attempting to use as is.")
        dataset = original_dataset
    
    return dataset, original_dataset, original_shape

def get_camera_position_from_config(config):
    """Extract camera position from config"""
    camera_config = config.get('camera', {})
    camera_position = camera_config.get('position')
    if camera_position:
        print("Using camera position from config")
        return camera_position
    else:
        print("No camera position found in config, using default")
        return [(-0.31676350915969265, -17.727903759830316, -5.178437900637502),
                (0.23947606203271332, 0.26414248443613014, -1.2407439169630974),
                (-0.9948541431039188, 0.050037906334993124, -0.08809904584374163)]

def color_then_spatial_clustering(xyz, rgb, n_color_clusters=DEFAULT_N_COLOR_CLUSTERS, spatial_eps=DEFAULT_SPATIAL_EPS, spatial_min_samples=DEFAULT_SPATIAL_MIN_SAMPLES):
    """Color-then-spatial clustering approach"""
    # 1. Color-based clustering (KMeans for simplicity, can use DBSCAN if you prefer)
    kmeans = KMeans(n_clusters=n_color_clusters, random_state=42)
    color_labels = kmeans.fit_predict(rgb)
    
    # Store the color centroids for later use
    color_centroids = kmeans.cluster_centers_
    
    # 2. For each color cluster, spatial refinement
    final_labels = -np.ones(len(xyz), dtype=int)
    label_counter = 0
    for color_id in np.unique(color_labels):
        mask = (color_labels == color_id)
        if np.sum(mask) < spatial_min_samples:
            continue  # skip tiny color clusters
        spatial_db = DBSCAN(eps=spatial_eps, min_samples=spatial_min_samples)
        spatial_labels = spatial_db.fit_predict(xyz[mask])
        # Assign unique global labels
        for s in np.unique(spatial_labels):
            if s == -1:
                continue  # skip noise
            final_labels[np.where(mask)[0][spatial_labels == s]] = label_counter
            label_counter += 1
    
    return final_labels, color_centroids

def preprocess_rgb_data(rgb, normalization='clip', clip_range=(0.0, 1.0)):
    """Preprocess RGB data with different normalization options"""
    if normalization == 'clip':
        return np.clip(rgb, clip_range[0], clip_range[1])
    elif normalization == 'minmax':
        # Min-max normalization to [0, 1]
        rgb_min = np.min(rgb, axis=0)
        rgb_max = np.max(rgb, axis=0)
        rgb_range = rgb_max - rgb_min
        rgb_range[rgb_range == 0] = 1  # Avoid division by zero
        return (rgb - rgb_min) / rgb_range
    elif normalization == 'zscore':
        # Z-score normalization
        rgb_mean = np.mean(rgb, axis=0)
        rgb_std = np.std(rgb, axis=0)
        rgb_std[rgb_std == 0] = 1  # Avoid division by zero
        return (rgb - rgb_mean) / rgb_std
    elif normalization == 'none':
        return rgb
    else:
        print(f"Warning: Unknown normalization '{normalization}'. Using clip.")
        return np.clip(rgb, clip_range[0], clip_range[1])

def pure_spatial_dbscan_clustering(xyz, eps, min_samples):
    """Pure spatial DBSCAN clustering (ignores color)"""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    return clustering.labels_, None

def pure_color_kmeans_clustering(xyz, rgb, n_clusters):
    """Pure color KMeans clustering (ignores spatial proximity)"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(rgb)
    return labels, kmeans.cluster_centers_

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def plot_3d_clusters_and_means(points, labels, cluster_means):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black used for noise.

        class_member_mask = (labels == k)
        xyz = points[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], s=10, alpha=0.1)

    # Plot mean locations of each cluster
    ax.scatter(cluster_means[:, 0], cluster_means[:, 1], cluster_means[:, 2], c='red', s=50, marker='x', label='Cluster Means')

    # Set figure labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Clusters with Mean Locations')
    set_axes_equal(ax)
    plt.legend()
    plt.show()

def calculate_cluster_means(points, labels):
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

def create_delaunay_edges(points):
    tri = Delaunay(points)
    vertices = points
    edges = []
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edge = [simplex[i], simplex[j]]
                edges.append(edge)
    edges = np.array(edges)
    print("Edges shape:", np.shape(edges), "Triangles shape:", np.shape(tri.simplices))
    return vertices, edges, tri

def remove_outlier_edges(vertices, edges, threshold=EDGE_FILTER_THRESHOLD):
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    mean_length = np.mean(edge_lengths)
    std_length = np.std(edge_lengths)
    threshold_length = mean_length + threshold * std_length
    valid_indices = edge_lengths <= threshold_length
    valid_edges = edges[valid_indices]
    initial_lengths = edge_lengths[valid_indices]
    return valid_edges, initial_lengths

def generate_color_map(deformation, colormap):
    norm = plt.Normalize(vmin=STRESS_NORM_MIN, vmax=STRESS_NORM_MAX, clip=True)  # Fixed normalization
    normalized_deformation = norm(deformation)
    colors = colormap(normalized_deformation)
    colors = colors[:, 0:3]
    return colors

def visualize_deformation(labels, cluster_means, cluster_points, dataset, camera_position, save_render=False, save_path="", param_str=""):
    plotter = pv.Plotter(off_screen=True)  # Enable off-screen rendering

    # Create and plot the Delaunay edges
    vertices, edges, _ = create_delaunay_edges(cluster_means)
    valid_edges, initial_lengths = remove_outlier_edges(vertices, edges)

    # Generate lines for the edges
    lines = np.hstack([np.full((valid_edges.shape[0], 1), 2), valid_edges])
    edge_mesh = pv.PolyData(vertices, lines=lines)

    plotter.add_mesh(edge_mesh, color='blue', line_width=1, style='surface')
    plotter.add_points(cluster_means, color='red', point_size=5, render_points_as_spheres=True)

    # Set camera position from config
    plotter.camera_position = camera_position

    # Compute stress for the first frame only
    j = 0  # Process only the first frame
    new_points = dataset[j].astype(np.float64)
    
    new_cluster_means = []
    for k in np.unique(labels):
        if k != -1:
            cluster_mask = labels == k
            new_cluster_mean = np.mean(new_points[cluster_mask], axis=0)
            new_cluster_means.append(new_cluster_mean)
    new_cluster_means = np.array(new_cluster_means)

    # Calculate new lengths and deformation
    new_lengths = np.linalg.norm(new_cluster_means[valid_edges[:, 0]] - new_cluster_means[valid_edges[:, 1]], axis=1)
    deformation = (new_lengths - initial_lengths)
    strain = deformation / initial_lengths
    stress = YOUNG_MODULUS_SILICONE * strain

    # Generate color map
    colors_list = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cmap_name = 'stress_strain_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=1024)
    colors_rgba = generate_color_map(stress, cm)

    plotter.clear()
    edge_mesh.points = new_cluster_means
    plotter.add_mesh(edge_mesh, rgb=True, scalars=colors_rgba, line_width=2.0, opacity=0.25)
    plotter.add_text(f'Time {j}', font_size=12)

    if save_render:
        plotter.show(auto_close=False)  # Render the scene before capturing
        screenshot_path = os.path.join(save_path, f'rendered_image_{param_str}.png')
        plotter.screenshot(screenshot_path)
        print(f"Saved screenshot: {screenshot_path}")

    plotter.close()



def main():
    config = load_config_file()
    if config is None:
        return

    dataset, original_dataset, original_shape = load_data_from_config(config)
    if dataset is None:
        return

    # Get camera position from config
    camera_position = get_camera_position_from_config(config)

    points = dataset[0]  # Use only the first frame for clustering
    
    # Check if we have color information (6D data) using original shape
    has_color_info = original_shape[2] == 6
    
    if has_color_info:
        # Extract XYZ and RGB from the appropriate datasets
        xyz = points  # Already extracted XYZ
        original_points = original_dataset[0]  # Get original points with RGB
        rgb_raw = original_points[:, 3:]  # Extract RGB from original
        rgb = preprocess_rgb_data(rgb_raw, RGB_NORMALIZATION, RGB_CLIP_RANGE)
        print("Using color clustering approach with 6D data")
        print(f"RGB preprocessing: {RGB_NORMALIZATION}")
    else:
        # Use only XYZ coordinates for regular DBSCAN
        xyz = points
        rgb = None
        print("Using regular DBSCAN clustering with 3D data")

    # Use grid search parameters from configuration section
    
    # Create a single dataset folder with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_folder = f"rendered_data_{timestamp}"
    os.makedirs(dataset_folder, exist_ok=True)
    
    if has_color_info:
        # Multiple clustering approaches for 6D data
        clustering_methods = []
        
        if USE_PURE_SPATIAL_DBSCAN:
            clustering_methods.append(('pure_spatial', 'Pure Spatial DBSCAN'))
        if USE_PURE_COLOR_KMEANS:
            clustering_methods.append(('pure_color', 'Pure Color KMeans'))
        if USE_COLOR_THEN_SPATIAL:
            clustering_methods.append(('color_spatial', 'Color-then-Spatial'))
        
        for method_name, method_desc in clustering_methods:
            print(f"\nTesting {method_desc}...")
            
            if method_name == 'pure_spatial':
                # Pure spatial DBSCAN grid search
                for eps in EPS_VALUES:
                    for min_samples in MIN_SAMPLES_VALUES:
                        param_str = f"pure_spatial_eps_{eps}_min_{min_samples}"
                        
                        labels, _ = pure_spatial_dbscan_clustering(xyz, eps, min_samples)
                        cluster_means, cluster_points = calculate_cluster_means(xyz, labels)
                        
                        visualize_deformation(labels, cluster_means, cluster_points, dataset, camera_position,
                                           save_render=True, save_path=dataset_folder, param_str=param_str)
            
            elif method_name == 'pure_color':
                # Pure color KMeans grid search
                for n_clusters in N_COLOR_CLUSTERS_VALUES:
                    param_str = f"pure_color_clusters_{n_clusters}"
                    
                    labels, _ = pure_color_kmeans_clustering(xyz, rgb, n_clusters)
                    cluster_means, cluster_points = calculate_cluster_means(xyz, labels)
                    
                    visualize_deformation(labels, cluster_means, cluster_points, dataset, camera_position,
                                       save_render=True, save_path=dataset_folder, param_str=param_str)
            
            elif method_name == 'color_spatial':
                # Color-then-spatial clustering grid search
                for n_color_clusters in N_COLOR_CLUSTERS_VALUES:
                    for spatial_eps in EPS_VALUES:
                        for min_samples in MIN_SAMPLES_VALUES:
                            param_str = f"color_{n_color_clusters}_spatial_eps_{spatial_eps}_min_{min_samples}"

                            labels, color_centroids = color_then_spatial_clustering(
                                xyz, rgb, n_color_clusters, spatial_eps, min_samples
                            )

                            cluster_means, cluster_points = calculate_cluster_means(xyz, labels)

                            visualize_deformation(labels, cluster_means, cluster_points, dataset, camera_position,
                                               save_render=True, save_path=dataset_folder, param_str=param_str)
    else:
        # Regular DBSCAN grid search for 3D data
        for eps in EPS_VALUES:
            for min_samples in MIN_SAMPLES_VALUES:
                # Generate a filename identifier for this combination
                param_str = f"eps_{eps}_min_{min_samples}"

                # Apply DBSCAN
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
                labels = clustering.labels_

                # Calculate cluster means
                cluster_means, cluster_points = calculate_cluster_means(xyz, labels)

                # Render only the first frame, save with param_str in the filename
                visualize_deformation(labels, cluster_means, cluster_points, dataset, camera_position,
                                   save_render=True, save_path=dataset_folder, param_str=param_str)

    print(f"All results saved in {dataset_folder}")

if __name__ == "__main__":
    main()