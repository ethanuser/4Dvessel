from pathlib import Path
import numpy as np
import pyvista as pv
import time
import sys
import os

# Animation configuration constants
FPS = 5  # Frames per second (animation speed)
NUM_CYCLES = 1  # Number of times to cycle through all frames
SAVE_VIDEO = True  # Set to True to save animation as video file

# Path to the single efficient file
# base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/blender-data")
base_path = str(Path.home() / "Documents/Blender/Vessel-Renders/synthetic_translate/blender-data")
# base_path = str(Path.home() / "Documents/Blender/blender-data")
single_file_path = f'{base_path}/mesh_data_single.npz'


def create_interactive_visualization(frames_data):
    """Create an animated visualization that cycles through frames"""
    
    # Get all frame indices
    frame_indices = sorted(frames_data.keys())
    
    # Calculate frame delay based on FPS
    frame_delay = 1.0 / FPS
    
    # CRITICAL: Calculate bounding box across ALL frames to set fixed camera
    # This ensures translation is visible (camera doesn't auto-fit each frame)
    print("Calculating bounding box across all frames...")
    all_coords = np.vstack([frames_data[idx]['coords'] for idx in frame_indices])
    bounds = [
        all_coords[:, 0].min(), all_coords[:, 0].max(),  # x range
        all_coords[:, 1].min(), all_coords[:, 1].max(),  # y range
        all_coords[:, 2].min(), all_coords[:, 2].max()   # z range
    ]
    print(f"Overall bounds: X[{bounds[0]:.2f}, {bounds[1]:.2f}], "
          f"Y[{bounds[2]:.2f}, {bounds[3]:.2f}], "
          f"Z[{bounds[4]:.2f}, {bounds[5]:.2f}]")
    
    # Create a PyVista plotter
    plotter = pv.Plotter(title=f"Animated Frame Visualization - {FPS} FPS")
    plotter.set_background('white')
    
    # Set up video recording if enabled
    video_path = None
    if SAVE_VIDEO:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        video_filename = f"vessel_animation_{FPS}fps_{NUM_CYCLES}cycles.mp4"
        video_path = os.path.join(script_dir, video_filename)
        print(f"\nRecording video to: {video_path}")
        plotter.open_movie(video_path, framerate=FPS)
    
    # Store the current point cloud actor for easy removal
    current_actor = None
    
    def update_frame(frame_idx):
        nonlocal current_actor
        
        frame_data = frames_data[frame_idx]
        
        # Remove previous actor if it exists
        if current_actor is not None:
            plotter.remove_actor(current_actor)
        
        # Add new points for current frame
        if frame_data['coords'].shape[0] > 0:
            point_cloud = pv.PolyData(frame_data['coords'])
            current_actor = plotter.add_points(point_cloud, 
                                             render_points_as_spheres=True,
                                             point_size=3,
                                             color='blue',
                                             opacity=0.8)
        
        # DO NOT reset camera - keep fixed view to see translation
    
    # Initialize with first frame
    update_frame(frame_indices[0])
    
    # Set camera to fit ALL frames (only once, at the start)
    # This ensures translation is visible throughout the animation
    plotter.camera_position = 'xy'
    plotter.reset_camera(bounds=bounds)
    
    # Show the plotter (non-interactive if saving video, interactive otherwise)
    if SAVE_VIDEO:
        plotter.show(auto_close=False, interactive=False)
    else:
        plotter.show(interactive_update=True)
    
    # Animation loop
    total_frames_to_record = len(frame_indices) * NUM_CYCLES
    frame_count = 0
    
    for cycle in range(NUM_CYCLES):
        for frame_idx in frame_indices:
            # Update to this frame
            update_frame(frame_idx)
            
            # Update the plotter
            plotter.update()
            
            # Write frame to video if recording
            if SAVE_VIDEO:
                plotter.write_frame()
                frame_count += 1
                if frame_count % 10 == 0 or frame_count == total_frames_to_record:
                    print(f"  Recorded {frame_count}/{total_frames_to_record} frames...", end='\r')
            
            # Wait based on FPS (only if not recording video)
            if not SAVE_VIDEO:
                time.sleep(frame_delay)
    
    # Close the plotter
    if SAVE_VIDEO:
        print(f"\nVideo recording complete! Saved to: {video_path}")
    plotter.close()

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
    
    print(f"\nProcessing {total_frames} frames...")
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

try:
    # Load the single file
    print(f"Loading numpy file: {single_file_path}")
    mesh_data = np.load(single_file_path)
    
    # Extract the arrays
    coords = mesh_data['coords']
    frame_numbers = mesh_data['frame_numbers']
    times = mesh_data['times']
    
    # Print initial metrics
    total_points = len(coords)
    unique_frames = np.unique(frame_numbers)
    num_frames = len(unique_frames)
    
    print("\n" + "="*60)
    print("FILE METRICS")
    print("="*60)
    print(f"Total number of points: {total_points:,}")
    print(f"Number of frames: {num_frames}")
    print(f"File size: {coords.nbytes / (1024*1024):.2f} MB (coordinates only)")
    print("="*60)
    
    # Extract data organized by frames
    frames_data = get_all_frames_data(coords, frame_numbers, times)
    
    # Print summary after processing
    total_vertices_processed = sum(frame['num_vertices'] for frame in frames_data.values())
    print(f"\nProcessing complete!")
    print(f"Total vertices processed: {total_vertices_processed:,}")
    print(f"Average vertices per frame: {total_vertices_processed // num_frames:,}")
    print()
    
    # Create interactive visualization
    create_interactive_visualization(frames_data)
    
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Error processing Single-File Efficient Storage: {e}") 