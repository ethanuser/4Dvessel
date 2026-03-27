from pathlib import Path
import numpy as np
import sys
import os

# Path to the single efficient file
# base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/blender-data")
base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/synthetic_translate/blender-data")
# base_path = str(Path.home() / "Documents/Blender/blender-data")
single_file_path = f'{base_path}/mesh_data_single.npz'


def calculate_bounding_box(coords):
    """
    Calculate the bounding box (min/max x, y, z) for all points.
    
    Args:
        coords: Array of point coordinates (N, 3)
        
    Returns:
        Dictionary with min/max values for each axis
    """
    if len(coords) == 0:
        return None
    
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    
    return {
        'x': {'min': x_min, 'max': x_max, 'range': x_max - x_min},
        'y': {'min': y_min, 'max': y_max, 'range': y_max - y_min},
        'z': {'min': z_min, 'max': z_max, 'range': z_max - z_min}
    }


def print_bounding_box(bounds, total_points):
    """Print bounding box information in a formatted way"""
    print("\n" + "="*70)
    print("BOUNDING BOX INFORMATION")
    print("="*70)
    print(f"Total number of points: {total_points:,}")
    print()
    print("X-axis:")
    print(f"  Min: {bounds['x']['min']:.6f}")
    print(f"  Max: {bounds['x']['max']:.6f}")
    print(f"  Range: {bounds['x']['range']:.6f}")
    print()
    print("Y-axis:")
    print(f"  Min: {bounds['y']['min']:.6f}")
    print(f"  Max: {bounds['y']['max']:.6f}")
    print(f"  Range: {bounds['y']['range']:.6f}")
    print()
    print("Z-axis:")
    print(f"  Min: {bounds['z']['min']:.6f}")
    print(f"  Max: {bounds['z']['max']:.6f}")
    print(f"  Range: {bounds['z']['range']:.6f}")
    print()
    print("Center point:")
    center_x = (bounds['x']['min'] + bounds['x']['max']) / 2
    center_y = (bounds['y']['min'] + bounds['y']['max']) / 2
    center_z = (bounds['z']['min'] + bounds['z']['max']) / 2
    print(f"  ({center_x:.6f}, {center_y:.6f}, {center_z:.6f})")
    print()
    print("Bounding box dimensions:")
    print(f"  Width (X):  {bounds['x']['range']:.6f}")
    print(f"  Height (Y): {bounds['y']['range']:.6f}")
    print(f"  Depth (Z):  {bounds['z']['range']:.6f}")
    print("="*70)


try:
    # Load the single file
    print(f"Loading numpy file: {single_file_path}")
    if not os.path.exists(single_file_path):
        print(f"ERROR: File not found: {single_file_path}")
        sys.exit(1)
    
    mesh_data = np.load(single_file_path)
    
    # Extract the arrays
    coords = mesh_data['coords']
    frame_numbers = mesh_data['frame_numbers']
    times = mesh_data['times']
    
    # Print initial metrics
    total_points = len(coords)
    unique_frames = np.unique(frame_numbers)
    num_frames = len(unique_frames)
    
    print("\n" + "="*70)
    print("FILE METRICS")
    print("="*70)
    print(f"Total number of points: {total_points:,}")
    print(f"Number of frames: {num_frames}")
    print(f"File size: {coords.nbytes / (1024*1024):.2f} MB (coordinates only)")
    print("="*70)
    
    # Calculate bounding box for all points
    print("\nCalculating bounding box for all points...")
    bounds = calculate_bounding_box(coords)
    
    if bounds is None:
        print("ERROR: No points found in the file.")
        sys.exit(1)
    
    # Print bounding box information
    print_bounding_box(bounds, total_points)
    
    # Optional: Calculate per-frame bounding boxes
    print("\n" + "="*70)
    print("PER-FRAME BOUNDING BOX ANALYSIS")
    print("="*70)
    
    frame_bounds = {}
    for frame_idx in unique_frames:
        frame_mask = frame_numbers == frame_idx
        frame_coords = coords[frame_mask]
        frame_bounds[frame_idx] = calculate_bounding_box(frame_coords)
    
    # Find frames with largest/smallest ranges
    x_ranges = {idx: b['x']['range'] for idx, b in frame_bounds.items()}
    y_ranges = {idx: b['y']['range'] for idx, b in frame_bounds.items()}
    z_ranges = {idx: b['z']['range'] for idx, b in frame_bounds.items()}
    
    max_x_frame = max(x_ranges, key=x_ranges.get)
    max_y_frame = max(y_ranges, key=y_ranges.get)
    max_z_frame = max(z_ranges, key=z_ranges.get)
    
    print(f"Frame with largest X range: Frame {max_x_frame} (range: {x_ranges[max_x_frame]:.6f})")
    print(f"Frame with largest Y range: Frame {max_y_frame} (range: {y_ranges[max_y_frame]:.6f})")
    print(f"Frame with largest Z range: Frame {max_z_frame} (range: {z_ranges[max_z_frame]:.6f})")
    print("="*70)
    
except FileNotFoundError as e:
    print(f"ERROR: File not found: {e}")
    sys.exit(1)
except KeyError as e:
    print(f"ERROR: Missing key in numpy file: {e}")
    print("Expected keys: 'coords', 'frame_numbers', 'times'")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
