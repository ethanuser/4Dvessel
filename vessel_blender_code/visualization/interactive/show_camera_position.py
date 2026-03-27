#!/usr/bin/env python3
"""
Interactive visualization tool to capture camera positions and pick points.

This script:
1. Loads data from a numpy file (cluster positions or mesh data)
2. Displays an interactive 3D visualization
3. When space bar is pressed, prints the current camera position and orientation
4. When a point is clicked, prints the position of the clicked point
5. Output can be copied directly into config files

Usage:
    python show_camera_position.py [npy_path]

Arguments:
    npy_path: Path to the .npy file with cluster positions or mesh data
"""

import sys
import os
from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import vessel_utils as vu
from utils.file_utils import resolve_path, get_project_root
from utils.data_utils import load_clustered_numpy, clustered_numpy_to_frames_data

# ============================================================================
# CONFIGURATION
# ============================================================================
# Default path to numpy file (cluster mesh timeseries)
DEFAULT_NPY_FILE = r"cluster_data/cluster_mesh_timeseries_lattice_strength_0_25.npy"

# Visualization constants
CLUSTER_POINT_SIZE = 15
CLUSTER_POINT_COLOR = "red"
HIGHLIGHT_COLOR = "magenta"
HIGHLIGHT_POINT_SIZE = 25  # Larger size for highlighted point
FRAME_INDEX = 0  # Which frame to display (0 = first frame)


def load_data(npy_path: str, project_root: Path):
    """
    Load data from numpy file.
    
    Args:
        npy_path: Path to numpy file
        project_root: Project root directory
        
    Returns:
        Tuple of (points, frame_index) where points is (N, 3) array
    """
    resolved_path = resolve_path(npy_path, project_root)
    
    # Try loading as clustered numpy first
    try:
        data = load_clustered_numpy(str(resolved_path), project_root)
        cluster_positions = data["cluster_positions"]
        num_frames = cluster_positions.shape[0]
        
        if FRAME_INDEX >= num_frames:
            print(f"Warning: Frame index {FRAME_INDEX} >= {num_frames}, using frame 0")
            frame_idx = 0
        else:
            frame_idx = FRAME_INDEX
        
        points = cluster_positions[frame_idx]  # (K, 3)
        print(f"Loaded cluster positions: {points.shape[0]} points from frame {frame_idx}/{num_frames-1}")
        return points, frame_idx
    except (FileNotFoundError, ValueError, KeyError):
        # Not a clustered numpy file, try other formats
        pass
    
    # Try loading as npz or raw numpy
    try:
        data = np.load(str(resolved_path), allow_pickle=True)
        
        # Check if it's a dict (npz format)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = {key: data[key] for key in data.keys()}
        
        if isinstance(data, dict):
            # Check if it's mesh data format (from npz)
            if "coords" in data:
                coords = data["coords"]
                frame_numbers = data["frame_numbers"]
                
                # Get first frame
                frame_mask = frame_numbers == 0
                points = coords[frame_mask]
                print(f"Loaded mesh data: {points.shape[0]} points from frame 0")
                return points, 0
        
        # Try as raw numpy array
        if isinstance(data, np.ndarray):
            if len(data.shape) == 2 and data.shape[1] == 3:
                points = data
                print(f"Loaded raw numpy array: {points.shape[0]} points")
                return points, 0
            else:
                raise ValueError(f"Unexpected array shape: {data.shape}")
    except Exception as e:
        raise ValueError(f"Could not load data from {npy_path}: {e}")


def create_interactive_visualization(points: np.ndarray, selection_radius: float = None):
    """
    Create an interactive PyVista visualization with camera position capture.
    
    Args:
        points: Array of points to visualize (N, 3)
        selection_radius: Radius for selecting neighboring points. If None, uses default.
    """
    plotter = pv.Plotter(title="Camera Position Capture - Press SPACE to print camera position")
    plotter.set_background("white")
    
    # Add points
    if len(points) > 0:
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(
            point_cloud,
            color=CLUSTER_POINT_COLOR,
            point_size=CLUSTER_POINT_SIZE,
            render_points_as_spheres=True,
            name="points"
        )
    else:
        print("Warning: No points to display")
        return
    
    # Add axes
    plotter.add_axes()
    
    # Set camera to fit the data
    plotter.camera_position = "xy"
    plotter.reset_camera()
    
    # Track the current highlight actors
    current_highlight_actors = []
    
    # Calculate appropriate marker size based on point cloud scale
    if len(points) > 0:
        bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
        scene_scale = np.linalg.norm(bbox_size)
        marker_radius = scene_scale * 0.02  # 2% of bounding box diagonal
        
        # Set default selection radius if not provided
        if selection_radius is None:
            selection_radius = scene_scale * 0.1  # 10% of bounding box default
            print(f"Using auto-calculated selection radius: {selection_radius:.4f}")
    else:
        marker_radius = 1.0
        if selection_radius is None:
            selection_radius = 5.0

    print(f"Selection Radius: {selection_radius:.4f}")
    
    def clear_highlights():
        """Remove all highlight actors"""
        nonlocal current_highlight_actors
        for actor in current_highlight_actors:
            plotter.remove_actor(actor)
        current_highlight_actors = []

    def highlight_region(center_point: np.ndarray, nearby_points: np.ndarray):
        """Highlight a region of points and show selection sphere"""
        clear_highlights()
        
        # 1. Highlight the center point (the one closest to click)
        center_sphere = pv.Sphere(radius=marker_radius, center=center_point)
        actor_center = plotter.add_mesh(
            center_sphere,
            color=HIGHLIGHT_COLOR,
            name="highlight_center",
            pickable=False
        )
        current_highlight_actors.append(actor_center)

        # 2. Highlight all points within radius
        if len(nearby_points) > 0:
            nearby_poly = pv.PolyData(nearby_points)
            actor_points = plotter.add_mesh(
                nearby_poly,
                color=HIGHLIGHT_COLOR,
                point_size=HIGHLIGHT_POINT_SIZE,
                render_points_as_spheres=True,
                name="highlight_points",
                pickable=False
            )
            current_highlight_actors.append(actor_points)
        
        # 3. Show the selection radius as a transparent sphere
        radius_sphere = pv.Sphere(radius=selection_radius, center=center_point)
        actor_radius = plotter.add_mesh(
            radius_sphere,
            color=HIGHLIGHT_COLOR,
            style="wireframe",
            opacity=0.5,
            name="highlight_radius",
            pickable=False
        )
        current_highlight_actors.append(actor_radius)
        
        # Also maybe a transparent surface for the sphere
        actor_radius_surf = plotter.add_mesh(
            radius_sphere,
            color=HIGHLIGHT_COLOR,
            opacity=0.1,
            name="highlight_radius_surf",
            pickable=False
        )
        current_highlight_actors.append(actor_radius_surf)
    
    def on_print_camera():
        """Print the current camera position when space bar is pressed"""
        camera_pos = plotter.camera_position
        print("\n" + "="*70)
        print("📷 CURRENT CAMERA POSITION")
        print("="*70)
        print("Copy this to your config file:")
        print('"camera": {')
        print('  "position": [')
        print(f'    [{camera_pos[0][0]:.6f}, {camera_pos[0][1]:.6f}, {camera_pos[0][2]:.6f}],')
        print(f'    [{camera_pos[1][0]:.6f}, {camera_pos[1][1]:.6f}, {camera_pos[1][2]:.6f}],')
        print(f'    [{camera_pos[2][0]:.6f}, {camera_pos[2][1]:.6f}, {camera_pos[2][2]:.6f}]')
        print('  ]')
        print('}')
        print("="*70)
        print()
    
    def on_point_picked(picked_point):
        """Handle point picking - select region around clicked point"""
        try:
            # Handle different formats PyVista might return
            if picked_point is None:
                return
            
            # Convert to numpy array if needed
            if not isinstance(picked_point, np.ndarray):
                picked_point = np.array(picked_point)
            
            # Ensure it's a 1D array of 3 elements
            picked_point = picked_point.flatten()
            if len(picked_point) < 3:
                return
            
            picked_point = picked_point[:3].reshape(1, -1)
            
            # Find the nearest point in the point cloud to get exact center
            distances = cdist(picked_point, points)
            nearest_idx = np.argmin(distances)
            nearest_point = points[nearest_idx]
            min_distance = distances[0, nearest_idx]
            
            # Find all points within selection_radius of the nearest_point
            # We recalculate distances from the TRUE point, not the mouse click location
            dists_from_center = cdist(nearest_point.reshape(1, 3), points).flatten()
            mask_nearby = dists_from_center <= selection_radius
            nearby_points = points[mask_nearby]
            nearby_indices = np.where(mask_nearby)[0]
            
            # Highlight the region
            highlight_region(nearest_point, nearby_points)
            
            print(f"📍 SELECTED: center_idx={nearest_idx}, pos=[{nearest_point[0]:.2f}, {nearest_point[1]:.2f}, {nearest_point[2]:.2f}], radius={selection_radius:.2f}, count={len(nearby_points)}")

        except Exception as e:
            print(f"Error processing picked point: {e}")
            import traceback
            traceback.print_exc()
    
    # Add space bar event
    plotter.add_key_event('space', on_print_camera)
    
    # Enable point picking
    plotter.enable_point_picking(callback=on_point_picked, show_message=False)
    
    # Instructions
    instructions = (
        "Controls:\n"
        "  SPACE: Print current camera position\n"
        "  Click: Select region around point\n"
        "  Mouse: Rotate/zoom/pan view\n"
        "  Close window to exit\n"
        "\n"
        f"Selection Radius: {selection_radius:.4f}\n"
        "Camera position will be printed in config format\n"
        "when you press SPACE."
    )
    plotter.add_text(instructions, font_size=10, position="upper_right")
    
    print("\n" + "="*70)
    print("INTERACTIVE CAMERA POSITION CAPTURE & REGION SELECTION")
    print("="*70)
    print("• Use mouse to rotate, zoom, and pan the view")
    print("• Press SPACE to print the current camera position")
    print("• Click on any point to select surrounding region")
    print(f"• Selection Radius: {selection_radius:.4f}")
    print("• Close the window to exit")
    print("="*70 + "\n")
    
    # Show visualization
    plotter.show()


def main():
    """
    Main entry point for camera position capture tool.
    """
    # Determine project root
    project_root = get_project_root(__file__)
    
    # Parse command-line arguments
    npy_path_str = DEFAULT_NPY_FILE
    selection_radius = None
    
    if len(sys.argv) >= 2:
        npy_path_str = sys.argv[1]
    
    if len(sys.argv) >= 3:
        try:
            selection_radius = float(sys.argv[2])
        except ValueError:
            print(f"⚠️  Invalid radius value: {sys.argv[2]}, using default.")
            
    if not npy_path_str:
        print("=" * 70)
        print("⚠️  No numpy file specified.")
        print("Edit DEFAULT_NPY_FILE in show_camera_position.py")
        print("or pass the path as a command-line argument:")
        print("  python show_camera_position.py <npy_path> [radius]")
        print("=" * 70)
        return
    
    # Resolve path
    resolved_path = resolve_path(npy_path_str, project_root)
    
    if not resolved_path.exists():
        print(f"⚠️  ERROR: File not found: {npy_path_str}")
        print(f"   Resolved path: {resolved_path}")
        return
    
    try:
        print(f"\nLoading data from: {resolved_path}")
        points, frame_idx = load_data(npy_path_str, project_root)
        
        if len(points) == 0:
            print("⚠️  ERROR: No points found in data file")
            return
        
        # Create and show interactive visualization
        create_interactive_visualization(points, selection_radius)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n⚠️  ERROR:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

