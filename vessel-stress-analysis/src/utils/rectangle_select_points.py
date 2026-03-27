"""
Rectangle selection utility for PyVista point clouds.
Provides functions to select points in 3D space using 2D rectangle selection.
"""

import pyvista as pv
import numpy as np
import time
from typing import List, Tuple


def select_points_with_rectangle(points: np.ndarray, camera_position=None, 
                                point_size: int = 5, point_color: str = 'red',
                                selected_color: str = 'green', selected_size: int = 8) -> List[int]:
    """
    Interactive rectangle selection for 3D points using PyVista.
    
    Args:
        points: Array of 3D points (N, 3) - X, Y, Z coordinates
        camera_position: Initial camera position (optional)
        point_size: Size of the points to display
        point_color: Color of unselected points
        selected_color: Color of selected points
        selected_size: Size of selected points
        
    Returns:
        List of selected point indices
    """
    
    # Create plotter and add points
    plotter = pv.Plotter()
    points_mesh = pv.PolyData(points)
    points_mesh['point_ids'] = np.arange(len(points))
    plotter.add_points(points_mesh, color=point_color, point_size=point_size, render_points_as_spheres=True)
    
    # Set camera position if provided
    if camera_position:
        plotter.camera_position = camera_position
    
    # State variables
    selected_indices = []
    selection_complete = {'done': False}
    
    def _world_to_screen_coordinates(point, camera, window_size):
        """Convert 3D world coordinates to 2D screen coordinates."""
        try:
            # Get camera parameters
            cam_pos = np.array(camera.position)
            focal_point = np.array(camera.focal_point)
            view_up = np.array(camera.up)
            
            # Calculate view direction
            view_dir = focal_point - cam_pos
            view_dir = view_dir / np.linalg.norm(view_dir)
            
            # Calculate right vector
            right = np.cross(view_dir, view_up)
            right = right / np.linalg.norm(right)
            
            # Calculate up vector (orthogonal to view_dir and right)
            up = np.cross(right, view_dir)
            up = up / np.linalg.norm(up)
            
            # Transform point to camera space
            point_vec = np.array(point) - cam_pos
            
            # Project onto camera axes
            x_cam = np.dot(point_vec, right)
            y_cam = np.dot(point_vec, up)
            z_cam = np.dot(point_vec, view_dir)
            
            # Simple perspective projection (assuming z_cam > 0)
            if z_cam > 0.001:  # Avoid division by zero
                # Get field of view and aspect ratio
                fov = camera.view_angle * np.pi / 180.0  # Convert to radians
                aspect = window_size[0] / window_size[1]
                
                # Project to normalized device coordinates
                ndc_x = x_cam / (z_cam * np.tan(fov / 2.0) * aspect)
                ndc_y = y_cam / (z_cam * np.tan(fov / 2.0))
                
                # Convert to screen coordinates
                screen_x = (ndc_x + 1.0) * 0.5 * window_size[0]
                screen_y = (ndc_y + 1.0) * 0.5 * window_size[1]
                
                return screen_x, screen_y
            else:
                return None, None
                
        except Exception as e:
            print(f"Error converting coordinates: {e}")
            return None, None
    
    def _rectangle_selected(rectangle):
        """Callback for rectangle selection"""
        nonlocal selected_indices
        selected_indices = []
        
        if hasattr(rectangle, 'viewport'):
            viewport = rectangle.viewport
            x1, y1, x2, y2 = viewport
            
            # Get camera and window information
            camera = plotter.camera
            window_size = plotter.window_size
            
            # Convert each point to screen coordinates and check if it's in the rectangle
            for i, point in enumerate(points):
                screen_x, screen_y = _world_to_screen_coordinates(point, camera, window_size)
                
                if screen_x is not None and screen_y is not None:
                    # Check if within selection rectangle
                    if (min(x1, x2) <= screen_x <= max(x1, x2) and 
                        min(y1, y2) <= screen_y <= max(y1, y2)):
                        selected_indices.append(i)
            
            print(f"Selected {len(selected_indices)} points")
            
            # Highlight selected points visually
            if selected_indices:
                # Clear any previous highlighting
                plotter.clear()
                
                # Add all points in original color
                plotter.add_points(points_mesh, color=point_color, point_size=point_size, render_points_as_spheres=True)
                
                # Add selected points in highlighted color
                selected_points = points[selected_indices]
                selected_mesh = pv.PolyData(selected_points)
                plotter.add_points(selected_mesh, color=selected_color, point_size=selected_size, render_points_as_spheres=True)
    
    def _confirm():
        """Called when user presses Enter/Return"""
        selection_complete['done'] = True
        plotter.close()
    
    def _reset():
        """Called when user presses 'r' to reset selection"""
        nonlocal selected_indices
        selected_indices = []
        print("Selection reset")
        
        # Clear and redraw all points in original color
        plotter.clear()
        plotter.add_points(points_mesh, color=point_color, point_size=point_size, render_points_as_spheres=True)
    
    # Enable rectangle selection
    try:
        plotter.enable_rectangle_picking(callback=_rectangle_selected, show_message=True)
    except Exception as e:
        print(f"Error enabling rectangle selection: {e}")
        return []
    
    # Add instruction text
    instruction = plotter.add_text(
        "Click and drag to select. ENTER to confirm, 'r' to reset.",
        position='lower_right', font_size=12
    )
    
    # Add key events
    plotter.add_key_event('Return', _confirm)
    plotter.add_key_event('r', _reset)
    
    # Show the plotter and wait for user interaction
    plotter.show(interactive_update=True)
    
    # Wait for user to complete selection
    while not selection_complete['done']:
        plotter.update()
        time.sleep(0.05)
    
    return selected_indices


def test_rectangle_selection():
    """Test function to demonstrate rectangle selection with a sample point cloud."""
    # Create a 3D point cloud with random distribution
    np.random.seed(42)  # For reproducible results
    
    # Generate random points in 3D space
    n_points = 200
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.random.uniform(-3, 3, n_points)
    
    # Create some clusters and patterns
    # Add a few clusters
    cluster1_x = np.random.normal(-2, 0.5, 30)
    cluster1_y = np.random.normal(-2, 0.5, 30)
    cluster1_z = np.random.normal(0, 0.3, 30)
    
    cluster2_x = np.random.normal(2, 0.7, 25)
    cluster2_y = np.random.normal(2, 0.7, 25)
    cluster2_z = np.random.normal(1, 0.4, 25)
    
    # Add some points along a curve
    curve_t = np.linspace(0, 2*np.pi, 40)
    curve_x = 3 * np.cos(curve_t)
    curve_y = 3 * np.sin(curve_t)
    curve_z = np.sin(2*curve_t)
    
    # Combine all points
    x = np.concatenate([x, cluster1_x, cluster2_x, curve_x])
    y = np.concatenate([y, cluster1_y, cluster2_y, curve_y])
    z = np.concatenate([z, cluster1_z, cluster2_z, curve_z])
    
    # Create the final point array
    points = np.column_stack([x, y, z])
    
    print(f"Testing with {len(points)} points")
    
    # Test the rectangle selection
    selected_indices = select_points_with_rectangle(points)
    print(f"Selected {len(selected_indices)} points")
    
    return selected_indices


if __name__ == "__main__":
    test_rectangle_selection()