"""
Force analysis module using centralized configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from typing import List
from scipy.spatial import Voronoi, ConvexHull
import pyvista as pv
import time
from enum import Enum

from core.clustering import (
    color_then_spatial_clustering,
    calculate_cluster_means,
    create_delaunay_edges,
    remove_outlier_edges
)
from utils.clustering_state_utils import get_filtered_clustering_from_state
import utils.vessel_utils as vu

# ── Area calculation methods ──────────────────────────────────────────────
class AreaCalculationMethod(Enum):
    """Enum for different area calculation methods."""
    DEFAULT = "default"
    VORONOI = "voronoi"
    CONVEX_HULL = "convex_hull"

# ── Physical-scale constants ──────────────────────────────────────────────
CHESS_TOTAL_LENGTH_M    = 6.195 / 100        # length across 8 squares (metres)
CHESS_SQUARES_ALONG_X   = 8              # number of squares measured
CHESS_SQUARE_LENGTH_M   = CHESS_TOTAL_LENGTH_M / CHESS_SQUARES_ALONG_X  # ~0.774375 cm per square
CALIB_SCALING_FACTOR    = 0.667125       # same factor used in calibration objp (units per square)
CORRECTION_FACTOR       = 6.195 / 5.053 # ratio of the length of the chessboard in real life to the length of the chessboard we've calculated
UNIT_TO_MM              = (CHESS_SQUARE_LENGTH_M / CALIB_SCALING_FACTOR) * 1000 * CORRECTION_FACTOR  # Convert to mm

print("Scaling Factor: ", str(UNIT_TO_MM))
#  → multiply any 4DGS distance by UNIT_TO_MM to get millimetres

def find_clustering_state_for_experiment(experiment_name: str, output_dir: str) -> str:
    """
    Find clustering state file for the given experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for the experiment
        
    Returns:
        Path to the clustering state file
    """
    # Look for clustering state in the experiment's processed directory
    state_filename = f"clustering_state_{experiment_name}.json"
    state_path = os.path.join(output_dir, state_filename)
    
    return state_path

def get_plane_from_camera(camera_position):
    """Return (origin, normal, up) for the vertical slicing plane.
    camera_position is (pos, focal, up)."""
    pos, focal, up = map(np.asarray, camera_position)
    view_vec = focal - pos
    view_vec /= np.linalg.norm(view_vec)
    up_vec = up / np.linalg.norm(up)
    # plane normal = view × up  (vertical plane through screen centre)
    normal = np.cross(view_vec, up_vec)
    normal /= np.linalg.norm(normal)
    return pos, normal, up_vec

def edges_intersect_plane(edges, verts, origin, normal, tol=1e-9):
    """Boolean mask: True for edges that cross the plane."""
    p0 = verts[edges[:, 0]]
    p1 = verts[edges[:, 1]]
    d0 = (p0 - origin) @ normal      # signed distances
    d1 = (p1 - origin) @ normal
    return np.logical_and(d0 * d1 < -tol, np.abs(d0 - d1) > tol)

def compute_axial_force(stress, dirs, normal, areas=None, method=AreaCalculationMethod.DEFAULT):
    """Return scalar F (sum over selected edges).
    
    Args:
        stress: Stress values for each edge
        dirs: Edge direction vectors
        normal: Plane normal vector
        areas: Area values for each edge (optional)
        method: Area calculation method to use
    """
    proj = (dirs @ normal)           # dot-product
    
    if method == AreaCalculationMethod.DEFAULT:
        # Use default area if not provided (in mm²)
        default_area = 1e-8 * (UNIT_TO_MM ** 2)
        return float(np.sum(proj * stress * default_area))
    elif method == AreaCalculationMethod.VORONOI:
        if areas is not None:
            return float(np.sum(proj * stress * areas))
        else:
            # Fallback to default area
            default_area = 1e-8 * (UNIT_TO_MM ** 2)
            return float(np.sum(proj * stress * default_area))
    elif method == AreaCalculationMethod.CONVEX_HULL:
        # For convex hull method, stress should be a single averaged value
        if np.isscalar(stress):
            return float(stress * areas)  # areas should be the convex hull area
        else:
            # Fallback to default area
            default_area = 1e-8 * (UNIT_TO_MM ** 2)
            return float(np.sum(proj * stress * default_area))
    else:
        raise ValueError(f"Unknown area calculation method: {method}")

def calculate_force_through_plane(cluster_means, valid_edges, stress, camera_position, area_method=AreaCalculationMethod.VORONOI):
    """
    Calculate axial force through a plane defined by camera position.
    
    Args:
        cluster_means: Current cluster mean positions
        valid_edges: Valid edge connections
        stress: Stress values for each edge
        camera_position: Camera position to define the plane
        area_method: Method for calculating areas (default: VORONOI)
        
    Returns:
        Tuple of (axial_force, force_info) where force_info is None for non-convex hull methods
    """
    # Define plane from camera position
    origin, normal, up_vec = get_plane_from_camera(camera_position)
    
    # Find edges that intersect the plane
    edge_mask = edges_intersect_plane(valid_edges, cluster_means, origin, normal)
    
    if not edge_mask.any():
        print("No edges intersect the plane defined by camera position.")
        return 0.0, None
    
    # Calculate edge directions
    dirs = (cluster_means[valid_edges[:, 1]] - cluster_means[valid_edges[:, 0]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    
    if area_method == AreaCalculationMethod.DEFAULT:
        # Use default area calculation
        default_area = 1e-8 * (UNIT_TO_MM ** 2)
        F_t = compute_axial_force(stress[edge_mask], dirs[edge_mask], normal, 
                                 np.full(np.sum(edge_mask), default_area), 
                                 method=AreaCalculationMethod.DEFAULT)
        return F_t, None
    
    elif area_method == AreaCalculationMethod.CONVEX_HULL:
        # Calculate areas using convex hull method
        intersecting_edges = valid_edges[edge_mask]
        mids = (cluster_means[intersecting_edges[:, 0]] + cluster_means[intersecting_edges[:, 1]]) * 0.5
        
        # Local 2-D basis on plane
        v1 = np.cross(normal, up_vec)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        pts2d = np.column_stack([(mids - origin) @ v1, (mids - origin) @ v2])
        
        try:
            # Calculate convex hull area
            hull = ConvexHull(pts2d)
            hull_vertices = hull.vertices
            hull_points = pts2d[hull_vertices]
            x, y = hull_points[:, 0], hull_points[:, 1]
            hull_area = 0.5 * abs((x * np.roll(y, -1) - y * np.roll(x, -1)).sum())
            # Convert to real-world units (mm²)
            hull_area = hull_area * (UNIT_TO_MM ** 2)
            
            # Average the stresses for intersecting edges
            intersecting_stresses = stress[edge_mask]
            avg_stress = np.mean(intersecting_stresses)
            
            # Average the projections for intersecting edges
            intersecting_dirs = dirs[edge_mask]
            avg_projection = np.mean(intersecting_dirs @ normal)
            
            # Calculate force: average_stress * average_projection * hull_area
            F_t = float(avg_stress * avg_projection * hull_area)
            
            # Return additional information for convex hull method
            return F_t, {
                'hull_area_mm2': hull_area,
                'avg_stress_pa': avg_stress,
                'avg_projection': avg_projection,
                'num_intersecting_edges': np.sum(edge_mask)
            }
            
        except Exception as e:
            print(f"Convex hull calculation failed: {e}. Using default area.")
            # Fallback to default area calculation with proper units
            default_area = 1e-8 * (UNIT_TO_MM ** 2)  # Convert default area to mm²
            avg_stress = np.mean(stress[edge_mask])
            avg_projection = np.mean(dirs[edge_mask] @ normal)
            F_t = float(avg_stress * avg_projection * default_area)
            
            return F_t, {
                'hull_area_mm2': default_area,
                'avg_stress_pa': avg_stress,
                'avg_projection': avg_projection,
                'num_intersecting_edges': np.sum(edge_mask),
                'error': str(e)
            }
    
    elif area_method == AreaCalculationMethod.VORONOI:
        # Calculate areas using Voronoi tessellation
        intersecting_edges = valid_edges[edge_mask]
        mids = (cluster_means[intersecting_edges[:, 0]] + cluster_means[intersecting_edges[:, 1]]) * 0.5
        
        # Local 2-D basis on plane
        v1 = np.cross(normal, up_vec)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        pts2d = np.column_stack([(mids - origin) @ v1, (mids - origin) @ v2])
        
        # Build Voronoi and compute cell areas
        try:
            vor = Voronoi(pts2d)
            areas = np.zeros(len(pts2d), dtype=float)
            total_area = 0.0
            
            for i, region_index in enumerate(vor.point_region):
                region = vor.regions[region_index]
                if region is None or -1 in region or len(region) == 0:
                    areas[i] = np.nan
                else:
                    poly = vor.vertices[region]
                    x, y = poly[:, 0], poly[:, 1]
                    # Shoelace formula
                    area_i = 0.5 * abs((x * np.roll(y, -1) - y * np.roll(x, -1)).sum())
                    # Convert 2D area to real-world units (mm²)
                    area_i = area_i * (UNIT_TO_MM ** 2)
                    areas[i] = area_i
                    total_area += area_i
            
            # Fallback: any NaNs fill with leftover area equally
            nan_mask = np.isnan(areas)
            if nan_mask.any():
                try:
                    hull = ConvexHull(pts2d)
                    leftover = (hull.volume - total_area / (UNIT_TO_MM ** 2)) * (UNIT_TO_MM ** 2)
                    areas[nan_mask] = leftover / nan_mask.sum()
                except:
                    # If ConvexHull fails, use equal distribution
                    areas[nan_mask] = total_area / len(areas)
            
            # Create full-length array for all valid edges
            full_areas = np.zeros(len(valid_edges), float)
            full_areas[edge_mask] = areas
            
            # Compute axial force
            F_t = compute_axial_force(stress[edge_mask], dirs[edge_mask], normal, areas, 
                                     method=AreaCalculationMethod.VORONOI)
            
        except Exception as e:
            print(f"Voronoi calculation failed: {e}. Using default area.")
            # Fallback to default area calculation with proper units
            default_area = 1e-8 * (UNIT_TO_MM ** 2)  # Convert default area to mm²
            F_t = compute_axial_force(stress[edge_mask], dirs[edge_mask], normal, 
                                     np.full(np.sum(edge_mask), default_area),
                                     method=AreaCalculationMethod.DEFAULT)
        
        return F_t, None
    
    else:
        raise ValueError(f"Unknown area calculation method: {area_method}")

def interactive_plane_selection(cluster_means, valid_edges, camera_position):
    """
    Interactive plane selection using PyVista.
    
    Args:
        cluster_means: Initial cluster mean positions
        valid_edges: Valid edge connections
        camera_position: Initial camera position
        
    Returns:
        Selected camera position for force analysis
    """
    plotter = pv.Plotter()

    # Create mesh for visualization
    vertices = cluster_means
    lines = np.hstack([np.full((valid_edges.shape[0], 1), 2), valid_edges])
    edge_mesh = pv.PolyData(vertices, lines=lines)

    # Add mesh and points
    plotter.add_mesh(edge_mesh, color='blue', line_width=1, style='surface')
    plotter.add_points(cluster_means, color='red', point_size=5, render_points_as_spheres=True)

    # Set initial camera position
    plotter.camera_position = camera_position

    # State variables
    plane_defined = {'done': False}
    plane_geom = {'n': None}
    edge_mask = np.zeros(len(valid_edges), bool)
    selected_mask = np.zeros(len(valid_edges), bool)
    resume_flag = {'go': False}

    def _define_plane():
        origin, normal, up_vec = get_plane_from_camera(plotter.camera_position)
        plane_geom['n'] = normal
        
        # Find edges that intersect the plane
        mask = edges_intersect_plane(valid_edges, vertices, origin, normal)
        if not mask.any():
            print("No edges intersect this plane.")
            return
        edge_mask[:] = mask
        
        # Visualize intersecting edges in cyan
        cyan_lines = np.hstack([np.full((np.count_nonzero(mask), 1), 2), valid_edges[mask]])
        plane_edges = pv.PolyData(vertices, lines=cyan_lines)
        plotter.add_mesh(plane_edges, color='cyan', line_width=4, name='plane_edges')
        
        # Create and visualize the plane
        # Calculate plane bounds based on data extent
        x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
        
        # Create a large plane that covers the data
        data_extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
        plane_size = data_extent * 1.5
        
        # Create plane geometry
        plane = pv.Plane(center=origin, direction=normal, i_size=plane_size, j_size=plane_size)
        plotter.add_mesh(plane, color='green', opacity=0.4, name='plane_visualization')
        
        plane_defined['done'] = True
        
        print(f"Plane defined: {mask.sum()} intersecting edges (cyan).")
        print("Green transparent plane shows the analysis plane.")
        print("Click on cyan edges to select/deselect them for force analysis.")
        print("Press ENTER when done selecting edges.")

    def _pick_edge(mesh, *args):
        if not plane_defined['done']:
            return
        cell_id = args[0]
        if cell_id is None or cell_id >= len(valid_edges) or not edge_mask[cell_id]:
            return
        
        # Toggle selection
        if plotter.mouse_button == 'left':
            selected_mask[cell_id] = True
            plotter.add_mesh(mesh.extract_cells(cell_id), color='yellow',
                            line_width=6, name=f"sel_{cell_id}")
        elif plotter.mouse_button == 'right':
            selected_mask[cell_id] = False
            plotter.remove_actor(f"sel_{cell_id}")

    def _resume():
        if not plane_defined['done']:
            print("Define the plane first with key P.")
            return
        if not selected_mask.any():
            # If user never clicked, pre-select all intersecting edges
            selected_mask[:] = edge_mask
            print("No edges manually chosen – using all intersecting edges.")
        plotter.remove_actor('plane_edges')  # Clean up
        plotter.remove_actor('plane_visualization')  # Remove plane visualization
        resume_flag['go'] = True
        plotter.close()

    # Add key events
    plotter.add_key_event('p', _define_plane)
    plotter.add_key_event('Return', _resume)
    
    # Enable cell picking
    plotter.enable_cell_picking(callback=_pick_edge, through=True, show=True,
                                style='wire', use_mesh=True, multiple=True)

    print("=== INTERACTIVE PLANE SELECTION ===")
    print("Press 'P' to define plane based on current camera position")
    print("Click on cyan edges to select/deselect them")
    print("Press ENTER when done selecting edges")

    # Show the plotter and wait for user interaction
    plotter.show(interactive_update=True)
    print("Visualization initiated.")

    # Wait for user to complete plane selection
    while not resume_flag['go']:
        plotter.update()
        time.sleep(0.05)

    # Return the final camera position
    return plotter.camera_position

def plot_force_over_time(force_series: List[float], experiment_name: str, output_dir: str, total_experiment_time: float):
    """
    Plot force over time and save to experiment's plots directory.
    
    Args:
        force_series: List of force values for each time point
        experiment_name: Name of the experiment for the plot title
        output_dir: Output directory for the experiment
        total_experiment_time: Total experiment time in seconds
    """
    # Create time array in seconds
    num_time_points = len(force_series)
    time_points = np.linspace(0, total_experiment_time, num_time_points)
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot the force as a function of time
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, force_series, marker='o', linestyle='-', color='red', linewidth=2, markersize=6)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Axial Force (N)', fontsize=12)
    plt.title(f'Axial Force Over Time - {experiment_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot as SVG and PNG
    plot_filename = f'axial_force_{experiment_name}.svg'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    
    # Also save as PNG
    png_filename = f'axial_force_{experiment_name}.png'
    png_path = os.path.join(plots_dir, png_filename)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Force plot saved as '{plot_path}' and '{png_path}'")

def run_force_analysis(config_manager):
    """
    Run force analysis using the provided configuration manager.
    
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
        print("The force plot requires the total experiment time to be specified.")
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
    
    # Debug: Print bounding box dimensions
    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    xyz_dims = xyz_max - xyz_min
    longest_dim_idx = np.argmax(xyz_dims)
    longest_dim_length = xyz_dims[longest_dim_idx]
    longest_dim_length_mm = longest_dim_length * UNIT_TO_MM
    print(f"=== BOUNDING BOX DEBUG ===")
    print(f"XYZ min: {xyz_min}")
    print(f"XYZ max: {xyz_max}")
    print(f"XYZ dimensions: {xyz_dims}")
    print(f"Longest dimension: {['X', 'Y', 'Z'][longest_dim_idx]} = {longest_dim_length:.6f} units")
    print(f"Longest dimension in mm: {longest_dim_length_mm:.3f} mm")
    print("="*30)
    
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
    
    # Apply clustering state (required)
    print("\nApplying clustering state...")
    experiment_name = config_manager.config['experiment']['name']
    output_dir = config_manager.config['experiment']['output_dir']
    filtered_cluster_means, filtered_edges, kept_point_indices, kept_edge_indices = get_filtered_clustering_from_state(
        cluster_means, experiment_name=experiment_name, output_dir=output_dir
    )
    
    if len(filtered_cluster_means) != len(cluster_means):
        print(f"Applied filtering: {len(cluster_means)} -> {len(filtered_cluster_means)} cluster means")
        
        # CRITICAL FIX: Update labels to match the filtered cluster means
        # Create a mapping from old cluster indices to new cluster indices
        old_to_new_label = {}
        for new_idx, old_idx in enumerate(kept_point_indices):
            old_to_new_label[old_idx] = new_idx
        
        # Update labels to use the new cluster indices
        updated_labels = np.copy(labels)
        for old_label in np.unique(labels):
            if old_label != -1 and old_label in old_to_new_label:
                new_label = old_to_new_label[old_label]
                updated_labels[labels == old_label] = new_label
            elif old_label != -1 and old_label not in old_to_new_label:
                # This cluster was removed, mark as noise
                updated_labels[labels == old_label] = -1
        
        # Update the variables
        cluster_means = filtered_cluster_means
        labels = updated_labels
        cluster_points = [cluster_points[i] for i in kept_point_indices]
        
        print(f"Updated labels to match filtered cluster means")
        print(f"New unique labels: {np.unique(labels)}")
        print(f"New cluster means shape: {cluster_means.shape}")
    else:
        print("No filtering applied (using original clustering)")
    
    # Get initial camera position from config
    camera_position = config_manager.config.get('camera', {}).get('position')
    if not camera_position:
        print("Warning: No camera position found in config. Using default camera position.")
        camera_position = [(-0.31676350915969265, -17.727903759830316, -5.178437900637502),
                          (0.23947606203271332, 0.26414248443613014, -1.2407439169630974),
                          (-0.9948541431039188, 0.050037906334993124, -0.08809904584374163)]
    
    # Get material properties
    young_modulus = config_manager.get('material_properties.young_modulus_silicone', vu.YOUNG_MODULUS_SILICON)
    
    # Create and plot the Delaunay edges
    vertices, edges, _ = create_delaunay_edges(cluster_means)
    valid_edges, initial_lengths = remove_outlier_edges(vertices, edges, config_manager.config)
    
    print(f"Created mesh with {len(valid_edges)} valid edges")
    
    # Interactive plane selection
    final_camera_position = interactive_plane_selection(cluster_means, valid_edges, camera_position)
    
    # Area calculation method selection
    print("\n=== AREA CALCULATION METHOD SELECTION ===")
    print("Available methods:")
    print("1. DEFAULT - Use fixed default area for all edges")
    print("2. VORONOI - Use Voronoi tessellation to calculate individual edge areas")
    print("3. CONVEX_HULL - Use convex hull area with averaged stress and projection")
    
    method_choice = input("Select area calculation method (1/2/3, default=2): ").strip()
    
    if method_choice == "1":
        area_method = AreaCalculationMethod.DEFAULT
        print("Using DEFAULT area calculation method")
    elif method_choice == "3":
        area_method = AreaCalculationMethod.CONVEX_HULL
        print("Using CONVEX_HULL area calculation method")
    else:
        area_method = AreaCalculationMethod.VORONOI
        print("Using VORONOI area calculation method (default)")
    
    # Calculate force for each time frame
    force_series = []
    force_info_series = []  # Store additional info for convex hull method
    print("\nCalculating axial forces...")
    
    total_frames = len(dataset) - 1
    for j in range(1, len(dataset)):
        # Progress bar
        progress = int((j / total_frames) * 50)
        bar = "█" * progress + "░" * (50 - progress)
        print(f"\r[{bar}] {j}/{total_frames}", end="", flush=True)
        
        # Get new frame data
        new_frame = dataset[j].astype(np.float64)
        new_points = new_frame[:, :3]  # Only use first 3 columns (X, Y, Z)
        
        # Update cluster means while maintaining cluster ownership
        new_cluster_means = []
        unique_labels_sorted = np.sort(np.unique(labels))
        
        for k in unique_labels_sorted:
            if k != -1:
                cluster_mask = labels == k
                new_cluster_mean = np.mean(new_points[cluster_mask], axis=0)
                new_cluster_means.append(new_cluster_mean)
        
        new_cluster_means = np.array(new_cluster_means)
        
        # Calculate new lengths and deformation
        new_lengths = np.linalg.norm(new_cluster_means[valid_edges[:, 0]] - new_cluster_means[valid_edges[:, 1]], axis=1)
        deformation = (new_lengths - initial_lengths)
        strain = deformation / initial_lengths  # Calculate strain
        stress = young_modulus * strain  # Calculate stress
        
        # Calculate axial force through the plane
        if area_method == AreaCalculationMethod.CONVEX_HULL:
            axial_force, force_info = calculate_force_through_plane(new_cluster_means, valid_edges, stress, final_camera_position, area_method)
            force_series.append(axial_force)
            force_info_series.append(force_info)
            
            # Output detailed information for convex hull method
            print(f"\nTime point {j}:")
            print(f"  Hull area: {force_info['hull_area_mm2']:.6f} mm²")
            print(f"  Average stress: {force_info['avg_stress_pa']:.6f} Pa")
            print(f"  Average projection: {force_info['avg_projection']:.6f}")
            print(f"  Number of intersecting edges: {force_info['num_intersecting_edges']}")
            print(f"  Calculated force: {axial_force:.6f} N")
            if 'error' in force_info:
                print(f"  Error: {force_info['error']}")
        else:
            axial_force, _ = calculate_force_through_plane(new_cluster_means, valid_edges, stress, final_camera_position, area_method)
            force_series.append(axial_force)
    
    print()  # New line after progress bar
    print(f"Force calculation complete!")
    
    # Save force values
    output_file = config_manager.config['analysis']['force']['output_file']
    output_path = config_manager.get_output_path('force', output_file)
    np.save(output_path, np.array(force_series))
    print(f'Force calculations saved to {output_path}')
    
    # Save additional force info for convex hull method
    if area_method == AreaCalculationMethod.CONVEX_HULL and force_info_series:
        force_info_output_path = output_path.replace('.npy', '_info.npy')
        np.save(force_info_output_path, np.array(force_info_series, dtype=object))
        print(f'Force info saved to {force_info_output_path}')
    
    # Plot force over time
    plot_force_over_time(force_series, experiment_name, output_dir, total_experiment_time) 