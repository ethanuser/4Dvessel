from pathlib import Path
import numpy as np
import pyvista as pv
import sys
import os
import datetime
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
import vessel_utils as vu

# ============================================================================
# GRID SEARCH HYPERPARAMETERS
# ============================================================================
# DBSCAN clustering parameters to test
EPS_VALUES = [0.04,0.045,0.035]  # Maximum distance between samples in the same neighborhood
MIN_SAMPLES_VALUES = [2,3,4]  # Minimum number of samples in a neighborhood to form a cluster

# Delaunay edge filtering parameters to test
OUTLIER_EDGE_THRESHOLD_VALUES = [2]  # Multiplier of median edge length to consider an edge an outlier

# Frame selection
FRAME_TO_RENDER = 0  # Which frame to use for rendering (0 = first frame)

# Rendering settings
IMAGE_RESOLUTION = (1920, 1080)  # (width, height) for saved images
BACKGROUND_COLOR = 'white'
CLUSTER_POINT_SIZE = 10  # Size of cluster mean points
CLUSTER_POINT_COLOR = 'red'  # Color of cluster mean points
EDGE_LINE_WIDTH = 2  # Width of Delaunay edges
EDGE_COLOR = 'blue'  # Color of Delaunay edges

# Path to the single efficient file
# base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/blender-data")
base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/synthetic_translate/blender-data")
# base_path = str(Path.home() / "Documents/Blender/blender-data")
single_file_path = f'{base_path}/mesh_data_single.npz'



# ============================================================================
# CLUSTERING AND MESHING FUNCTIONS
# ============================================================================
# Core functions moved to vessel_utils.py


def render_and_save_mesh(cluster_means: np.ndarray, edges: np.ndarray,
                        save_path: str, param_str: str,
                        eps: float, min_samples: int, outlier_threshold: float,
                        bounds: list = None) -> None:
    """
    Render the mesh and save as an image.
    
    Args:
        cluster_means: Array of cluster mean positions (N, 3)
        edges: Array of edge indices (M, 2)
        save_path: Directory to save the image
        param_str: String identifier for this parameter combination
        eps: DBSCAN eps parameter value
        min_samples: DBSCAN min_samples parameter value
        outlier_threshold: Edge filtering threshold value
        bounds: Optional bounding box for camera setup
    """
    # Create off-screen plotter
    plotter = pv.Plotter(off_screen=True, window_size=IMAGE_RESOLUTION)
    plotter.set_background(BACKGROUND_COLOR)
    
    # Add cluster means
    if len(cluster_means) > 0:
        plotter.add_points(
            cluster_means,
            render_points_as_spheres=True,
            point_size=CLUSTER_POINT_SIZE,
            color=CLUSTER_POINT_COLOR,
            name='cluster_means'
        )
    
    # Add Delaunay edges
    if len(edges) > 0 and len(cluster_means) > 0:
        lines = np.hstack([np.full((edges.shape[0], 1), 2), edges])
        edge_mesh = pv.PolyData(cluster_means, lines=lines)
        plotter.add_mesh(
            edge_mesh,
            color=EDGE_COLOR,
            line_width=EDGE_LINE_WIDTH,
            name='delaunay_edges'
        )
    
    # Set camera to fit the data
    if bounds is not None:
        plotter.camera_position = 'xy'
        plotter.reset_camera(bounds=bounds)
    else:
        plotter.reset_camera()
    
    # Add text with parameter information
    info_text = (
        f"eps={eps}, min_samples={min_samples}, threshold={outlier_threshold}\n"
        f"Clusters: {len(cluster_means)}, Edges: {len(edges)}"
    )
    plotter.add_text(info_text, font_size=10, position='upper_left')
    
    # Save screenshot
    screenshot_path = os.path.join(save_path, f'mesh_{param_str}.png')
    plotter.screenshot(screenshot_path)
    print(f"  Saved: {screenshot_path}")
    
    plotter.close()


def process_frame_with_params(coords: np.ndarray, eps: float, min_samples: int,
                             outlier_threshold: float, bounds: list,
                             save_path: str) -> dict:
    """
    Process a single frame with given hyperparameters.
    
    Args:
        coords: Point coordinates for the frame (N, 3)
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        outlier_threshold: Edge filtering threshold
        bounds: Bounding box for camera setup
        save_path: Directory to save rendered images
        
    Returns:
        Dictionary with processing results
    """
    # Create parameter string for filename
    param_str = f"eps_{eps}_min_{min_samples}_thresh_{outlier_threshold}"
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    
    # Calculate cluster means
    cluster_means, cluster_points_list = vu.calculate_cluster_means(coords, labels)
    
    # Create Delaunay edges
    if len(cluster_means) >= 4:
        vertices, edges, simplices = vu.create_delaunay_edges(cluster_means)
        # Remove outlier edges
        valid_edges, removed_count = vu.remove_outlier_edges(vertices, edges, outlier_threshold)
    else:
        valid_edges = np.array([]).reshape(0, 2)
        removed_count = 0
    
    # Render and save
    render_and_save_mesh(cluster_means, valid_edges, save_path, param_str, 
                        eps, min_samples, outlier_threshold, bounds)
    
    return {
        'eps': eps,
        'min_samples': min_samples,
        'outlier_threshold': outlier_threshold,
        'num_clusters': len(cluster_means),
        'num_edges': len(valid_edges),
        'removed_edges': removed_count,
        'param_str': param_str
    }





def main():
    """Main function to run grid search"""
    try:
        # Load the single file
        print(f"Loading numpy file: {single_file_path}")
        if not os.path.exists(single_file_path):
            print(f"ERROR: File not found: {single_file_path}")
            return
        
        mesh_data = np.load(single_file_path)
        
        # Extract the arrays
        coords = mesh_data['coords']
        frame_numbers = mesh_data['frame_numbers']
        times = mesh_data['times']
        
        
        # Extract data organized by frames
        frames_data = vu.get_all_frames_data(coords, frame_numbers, times)
        
        # Select frame to render
        frame_indices = sorted(frames_data.keys())
        if FRAME_TO_RENDER >= len(frame_indices):
            print(f"ERROR: Frame {FRAME_TO_RENDER} not available. Using frame 0.")
            frame_to_use = frame_indices[0]
        else:
            frame_to_use = frame_indices[FRAME_TO_RENDER]
        
        frame_data = frames_data[frame_to_use]
        frame_coords = frame_data['coords']
        
        print(f"\nUsing frame {frame_to_use} for rendering")
        print(f"Frame points: {len(frame_coords):,}")
        
        # Calculate bounding box for camera setup
        print("\nCalculating bounding box...")
        bounds = [
            frame_coords[:, 0].min(), frame_coords[:, 0].max(),  # x range
            frame_coords[:, 1].min(), frame_coords[:, 1].max(),  # y range
            frame_coords[:, 2].min(), frame_coords[:, 2].max()   # z range
        ]
        print(f"Bounds: X[{bounds[0]:.2f}, {bounds[1]:.2f}], "
              f"Y[{bounds[2]:.2f}, {bounds[3]:.2f}], "
              f"Z[{bounds[4]:.2f}, {bounds[5]:.2f}]")
        
        # Create output directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"mesh_grid_search_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving images to: {output_dir}")
        
        # Grid search
        total_combinations = len(EPS_VALUES) * len(MIN_SAMPLES_VALUES) * len(OUTLIER_EDGE_THRESHOLD_VALUES)
        current_combination = 0
        
        print(f"\nStarting grid search over {total_combinations} combinations...")
        print(f"EPS values: {EPS_VALUES}")
        print(f"MIN_SAMPLES values: {MIN_SAMPLES_VALUES}")
        print(f"OUTLIER_THRESHOLD values: {OUTLIER_EDGE_THRESHOLD_VALUES}")
        print("="*70)
        
        results = []
        
        for eps in EPS_VALUES:
            for min_samples in MIN_SAMPLES_VALUES:
                for outlier_threshold in OUTLIER_EDGE_THRESHOLD_VALUES:
                    current_combination += 1
                    print(f"\n[{current_combination}/{total_combinations}] "
                          f"eps={eps}, min_samples={min_samples}, threshold={outlier_threshold}")
                    
                    result = process_frame_with_params(
                        frame_coords, eps, min_samples, outlier_threshold,
                        bounds, output_dir
                    )
                    results.append(result)
        
        # Save summary results
        summary_path = os.path.join(output_dir, 'grid_search_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GRID SEARCH SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Frame used: {frame_to_use}\n")
            f.write(f"Total points in frame: {len(frame_coords):,}\n")
            f.write(f"Total combinations tested: {total_combinations}\n\n")
            f.write("="*70 + "\n")
            f.write("RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for result in results:
                f.write(f"eps={result['eps']}, min_samples={result['min_samples']}, "
                       f"threshold={result['outlier_threshold']}\n")
                f.write(f"  Clusters: {result['num_clusters']}, "
                       f"Edges: {result['num_edges']}, "
                       f"Removed edges: {result['removed_edges']}\n")
                f.write(f"  Image: mesh_{result['param_str']}.png\n\n")
        
        print(f"\n" + "="*70)
        print("GRID SEARCH COMPLETE")
        print("="*70)
        print(f"All images saved to: {output_dir}")
        print(f"Summary saved to: {summary_path}")
        print(f"Total combinations: {total_combinations}")
        
        # Print best results (most clusters, most edges, etc.)
        if results:
            most_clusters = max(results, key=lambda x: x['num_clusters'])
            most_edges = max(results, key=lambda x: x['num_edges'])
            
            print("\nBest results:")
            print(f"  Most clusters: {most_clusters['num_clusters']} "
                  f"(eps={most_clusters['eps']}, min_samples={most_clusters['min_samples']}, "
                  f"threshold={most_clusters['outlier_threshold']})")
            print(f"  Most edges: {most_edges['num_edges']} "
                  f"(eps={most_edges['eps']}, min_samples={most_edges['min_samples']}, "
                  f"threshold={most_edges['outlier_threshold']})")
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
