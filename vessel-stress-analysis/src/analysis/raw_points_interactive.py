"""
Interactive raw point cloud visualization module.
Displays all raw points frame-by-frame without any clustering or meshing.
Fast and lightweight - just point rendering with RGB colors.
"""

import numpy as np
import pyvista as pv
import os
from typing import Optional

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================
POINT_SIZE = 4.0  # Size of rendered points


def create_raw_points_visualization(
    dataset: np.ndarray,
    experiment_name: str = "Raw Points",
    times: Optional[np.ndarray] = None,
) -> None:
    """
    Create an interactive 3D visualization showing raw point clouds frame-by-frame.
    
    Args:
        dataset: NumPy array of shape (T, N, 6) where T=time frames, N=points, 6=[X,Y,Z,R,G,B]
                 or shape (T, N, 3) for XYZ only
        experiment_name: Name for the visualization window
        times: Optional array of time values for each frame (T,)
    """
    # Validate dataset shape
    if len(dataset.shape) == 2:
        # Single frame - add time dimension
        dataset = dataset[np.newaxis, :, :]
    
    if len(dataset.shape) != 3:
        print(f"ERROR: Expected dataset shape (T, N, 6) or (T, N, 3), got {dataset.shape}")
        return
    
    num_frames, num_points_per_frame, num_features = dataset.shape
    
    if num_features not in [3, 6]:
        print(f"ERROR: Expected 3 (XYZ) or 6 (XYZRGB) features, got {num_features}")
        return
    
    has_rgb = num_features == 6
    
    print(f"Loaded dataset: {num_frames} frames, ~{num_points_per_frame} points per frame")
    print(f"RGB data: {'Yes' if has_rgb else 'No'}")
    
    # Extract frames data
    frames_data = []
    for t in range(num_frames):
        frame = dataset[t]
        coords = frame[:, :3]  # XYZ
        
        if has_rgb:
            rgb = np.clip(frame[:, 3:6], 0.0, 1.0)  # RGB, clipped to [0,1]
        else:
            rgb = None
        
        frames_data.append({
            'coords': coords,
            'rgb': rgb,
            'num_points': len(coords)
        })
    
    # Calculate bounding box across all frames
    all_coords = np.vstack([fd['coords'] for fd in frames_data])
    bounds = [
        float(all_coords[:, 0].min()),
        float(all_coords[:, 0].max()),
        float(all_coords[:, 1].min()),
        float(all_coords[:, 1].max()),
        float(all_coords[:, 2].min()),
        float(all_coords[:, 2].max()),
    ]
    
    # Pre-create PolyData objects for each frame (for fast switching)
    print("Preparing point cloud data...")
    frame_polydata = {}
    
    for t in range(num_frames):
        fd = frames_data[t]
        coords = fd['coords']
        
        if len(coords) > 0:
            poly = pv.PolyData(coords)
            
            # Add RGB colors if available
            if fd['rgb'] is not None:
                poly['colors'] = fd['rgb']
            
            frame_polydata[t] = poly
        else:
            frame_polydata[t] = None
    
    print(f"Prepared {len(frame_polydata)} frames")
    
    # Create PyVista plotter
    plotter = pv.Plotter(title=f"Raw Point Cloud Visualization - {experiment_name}")
    plotter.set_background('white')
    
    current_points_actor = None
    current_text_actor = None
    current_frame_idx = 0
    
    def update_frame_display(frame_idx: int) -> None:
        """Update the display to show a specific frame's raw points."""
        nonlocal current_points_actor, current_text_actor, current_frame_idx
        
        if frame_idx < 0 or frame_idx >= num_frames:
            return
        
        current_frame_idx = frame_idx
        fd = frames_data[frame_idx]
        poly = frame_polydata[frame_idx]
        
        # Store old actor for removal
        old_points_actor = current_points_actor
        
        # Add new points
        if poly is not None:
            if fd['rgb'] is not None:
                # Use RGB colors from data
                current_points_actor = plotter.add_mesh(
                    poly,
                    scalars='colors',
                    rgb=True,
                    render_points_as_spheres=True,
                    point_size=POINT_SIZE,
                    name=f'raw_points_{frame_idx}'
                )
            else:
                # No RGB - use default color
                current_points_actor = plotter.add_mesh(
                    poly,
                    render_points_as_spheres=True,
                    point_size=POINT_SIZE,
                    color='blue',
                    name=f'raw_points_{frame_idx}'
                )
        else:
            current_points_actor = None
        
        # Remove old actor
        if old_points_actor is not None:
            try:
                plotter.remove_actor(old_points_actor)
            except Exception:
                pass
        
        # Update text HUD
        if current_text_actor is not None:
            try:
                plotter.remove_actor(current_text_actor)
            except Exception:
                pass
        
        time_val = (
            float(times[frame_idx])
            if times is not None and len(times) == num_frames
            else float(frame_idx)
        )
        
        current_text_actor = plotter.add_text(
            f"Frame: {frame_idx}/{num_frames - 1} | "
            f"Time: {time_val:.4f} | "
            f"Points: {fd['num_points']:,}",
            font_size=12,
            position='upper_left'
        )
        
        plotter.render()
    
    def slider_callback(value: float) -> None:
        """Callback for slider widget"""
        frame_idx = int(value)
        update_frame_display(frame_idx)
    
    def navigate_frame(direction: str) -> None:
        """Navigate to next or previous frame"""
        nonlocal current_frame_idx
        
        if direction == 'next':
            if current_frame_idx < num_frames - 1:
                new_idx = current_frame_idx + 1
                update_frame_display(new_idx)
                if len(plotter.slider_widgets) > 0:
                    plotter.slider_widgets[0].GetRepresentation().SetValue(new_idx)
        elif direction == 'prev':
            if current_frame_idx > 0:
                new_idx = current_frame_idx - 1
                update_frame_display(new_idx)
                if len(plotter.slider_widgets) > 0:
                    plotter.slider_widgets[0].GetRepresentation().SetValue(new_idx)
        elif direction == 'first':
            update_frame_display(0)
            if len(plotter.slider_widgets) > 0:
                plotter.slider_widgets[0].GetRepresentation().SetValue(0)
        elif direction == 'last':
            last_idx = num_frames - 1
            update_frame_display(last_idx)
            if len(plotter.slider_widgets) > 0:
                plotter.slider_widgets[0].GetRepresentation().SetValue(last_idx)
    
    # Add coordinate axes
    plotter.add_axes()
    
    # Add slider widget for frame navigation
    plotter.add_slider_widget(
        callback=slider_callback,
        rng=(0, num_frames - 1),
        value=0,
        title="Frame",
        pointa=(0.1, 0.02),
        pointb=(0.9, 0.02),
        style='modern',
        tube_width=0.005,
        slider_width=0.02,
        interaction_event='always'
    )
    
    # Initialize with first frame
    update_frame_display(0)
    
    # Set camera to fit all points
    plotter.camera_position = 'xy'
    plotter.reset_camera(bounds=bounds)
    
    # Add instructions text
    instructions = (
        "Controls:\n"
        "  Slider (bottom): Drag to navigate frames\n"
        "  Arrow Keys: Left/Right to navigate frames\n"
        "  A/D keys: Previous/Next frame\n"
        "  Mouse: Rotate/zoom/pan view\n"
        "  Close window to exit\n"
        "\n"
        "Visualization:\n"
        "  Raw point cloud with RGB colors (if available)"
    )
    plotter.add_text(instructions, font_size=9, position='lower_left')
    
    # Register keyboard events
    def on_next_frame():
        navigate_frame('next')
    
    def on_prev_frame():
        navigate_frame('prev')
    
    plotter.add_key_event('Right', on_next_frame)
    plotter.add_key_event('Left', on_prev_frame)
    plotter.add_key_event('d', on_next_frame)
    plotter.add_key_event('D', on_next_frame)
    plotter.add_key_event('a', on_prev_frame)
    plotter.add_key_event('A', on_prev_frame)
    
    print("\nInteractive visualization ready!")
    print("Use the slider or arrow keys (Left/Right) to navigate frames")
    print("Close the window to exit\n")
    
    # Show the interactive plotter
    plotter.show()


def run_raw_points_interactive(data_path: str, experiment_name: Optional[str] = None) -> None:
    """
    Load raw point cloud data and run interactive visualization.
    
    Args:
        data_path: Path to .npy file containing point cloud data
                   Shape should be (T, N, 6) for [X,Y,Z,R,G,B] or (T, N, 3) for [X,Y,Z]
        experiment_name: Optional experiment name (defaults to filename)
    """
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        return
    
    if experiment_name is None:
        experiment_name = os.path.splitext(os.path.basename(data_path))[0]
    
    print(f"\n=== RAW POINT CLOUD VISUALIZATION: {experiment_name} ===")
    print(f"Loading data from: {data_path}")
    
    dataset = np.load(data_path)
    print(f"Loaded data shape: {dataset.shape}")
    
    # Try to infer time array from dataset metadata or use frame indices
    times = None
    if len(dataset.shape) == 3:
        # Default: assume unit time steps (can be overridden if needed)
        times = np.arange(dataset.shape[0], dtype=float)
    
    # Launch visualization
    create_raw_points_visualization(dataset, experiment_name, times)

