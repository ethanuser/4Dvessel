from pathlib import Path
import getpass
import os
import json
import math
import platform
from datetime import datetime
import bpy
import numpy as np
from mathutils import Matrix, Vector, Euler

# Parameters
num_frames = 160  # Number of frames to capture per camera
EXPORT_MESH_DATA = True  # If True, save and export mesh data. If False, skip mesh data extraction
SKIP_RENDERS = True  # If True, skip all rendering operations (only extract mesh data if enabled)

# Rigid motion controls
RIGID_MODE = True  # True = rigid translation/rotation only
RIGID_TRANSLATION = (1, 0, 0)  # Total translation over the full sequence (Blender units, XYZ)
RIGID_ROT_DEG = (0, 0, 0)  # Total rotation over the full sequence (degrees, XYZ Euler)
RIGID_SPACE = 'WORLD'  # 'WORLD' or 'LOCAL' - space for applying transformations
SETUP_ONLY = False  # If True, only sets keyframes and returns (for viewport preview without rendering)

# Rendering optimization settings
EEVEE_SAMPLES = 1  # Number of samples for Eevee (lower = faster, higher = better quality)

# Object names
chessboard_name = 'chessboard_6x8'
vessel_name = 'moderate tortuosity ICA'
text_name_1 = "Text.001"
text_name_2 = "Text.002"

# Detect operating system
is_windows = platform.system() == 'Windows'
is_mac = platform.system() == 'Darwin'

# Generate timestamp-based folder name (format: YYYY-MM-DD_HH-MM-SS)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Resolve camera parameters path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
json_file_path = os.path.abspath(os.path.join(script_dir, "..", "..", "Preprocess", "Camera Calibration", "calibrations", "camera_parameters.json"))

# Set paths based on operating system
if is_windows:
    username = getpass.getuser()
    base_output_dir = f"C:/Users/{username}/Documents/Blender/Vessel-Renders/{timestamp}/"
    output_dirs = [f"{base_output_dir}{name}/" for name in ["train_chessboard", "train_vessel"]]
    json_file_paths = [f"{base_output_dir}transforms_{name}.json" for name in ["train_chessboard", "train_vessel"]]
    blender_data_dir = f"{base_output_dir}blender-data/"
elif is_mac:
    base_output_dir = str(Path.home() / f"Documents/Blender/Vessel-Renders/{timestamp}/")
    output_dirs = [f"{base_output_dir}{name}/" for name in ["train_chessboard", "train_vessel"]]
    json_file_paths = [f"{base_output_dir}transforms_{name}.json" for name in ["train_chessboard", "train_vessel"]]
    blender_data_dir = f"{base_output_dir}blender-data/"
else:
    raise OSError(f"Unsupported operating system: {platform.system()}")

# Load camera parameters from JSON file (only needed if rendering)
if not SKIP_RENDERS:
    with open(json_file_path, 'r') as f:
        camera_params = json.load(f)

# Create blender-data directory for numpy files
if not os.path.exists(blender_data_dir):
    os.makedirs(blender_data_dir)

# Set up log file for all output (replaces print statements)
log_file_path = os.path.join(blender_data_dir, 'script_log.txt')
log_file = open(log_file_path, 'w')

def log_write(message):
    """Write message to log file (replaces print statements)"""
    log_file.write(str(message) + '\n')
    log_file.flush()  # Ensure it's written immediately

# Rigid motion helper functions
def store_original_transform(obj):
    """Store the original transform of an object for later restoration."""
    return (obj.matrix_world.copy(), obj.location.copy(), obj.rotation_euler.copy())

def restore_original_transform(obj, original):
    """Restore an object to its original transform."""
    mw, loc, rot = original
    obj.matrix_world = mw.copy()
    obj.location = loc.copy()
    obj.rotation_euler = rot.copy()

def clear_object_animation(obj):
    """Clear all animation data from an object."""
    if obj.animation_data:
        obj.animation_data_clear()
    # Also clear any leftover NLA tracks if present
    if obj.animation_data and obj.animation_data.nla_tracks:
        for tr in obj.animation_data.nla_tracks:
            obj.animation_data.nla_tracks.remove(tr)

def clear_all_scene_animations():
    """Wipe all animation data from the entire scene to ensure a clean slate."""
    for obj in bpy.data.objects:
        clear_object_animation(obj)
    # Remove all actions from the data block
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
    log_write("Cleared all scene animations and actions.")

def mute_all_modifiers(obj):
    """Disable all modifiers on an object to prevent any non-rigid deformation."""
    for mod in obj.modifiers:
        mod.show_viewport = False
        mod.show_render = False
    log_write(f"Muted ({len(obj.modifiers)}) modifiers on {obj.name} for rigid export.")

def setup_rigid_keyframes(obj, num_frames, translation_xyz, rot_deg_xyz, space='WORLD'):
    """Creates viewport-visible rigid motion by keyframing object transforms.
    
    Args:
        obj: The object to animate
        num_frames: Number of frames in the animation
        translation_xyz: Tuple of (x, y, z) translation values
        rot_deg_xyz: Tuple of (x, y, z) rotation values in degrees
        space: 'WORLD' or 'LOCAL' - space for applying transformations
    """
    clear_object_animation(obj)

    # Ensure rotation mode is Euler for predictable keyframes
    obj.rotation_mode = 'XYZ'

    # Record start pose
    start_loc = obj.location.copy()
    start_rot = obj.rotation_euler.copy()

    # End pose
    dloc = Vector(translation_xyz)
    drot = Euler(tuple(math.radians(x) for x in rot_deg_xyz), 'XYZ')

    if space.upper() == 'WORLD':
        end_loc = start_loc + dloc
        end_rot = Euler((start_rot.x + drot.x, start_rot.y + drot.y, start_rot.z + drot.z), 'XYZ')
    else:
        # Local-space translation: translate along the object's local axes
        end_loc = start_loc + (obj.matrix_world.to_3x3() @ dloc)
        end_rot = Euler((start_rot.x + drot.x, start_rot.y + drot.y, start_rot.z + drot.z), 'XYZ')

    # Keyframe at frame 0
    bpy.context.scene.frame_set(0)
    obj.location = start_loc
    obj.rotation_euler = start_rot
    obj.keyframe_insert(data_path="location", frame=0)
    obj.keyframe_insert(data_path="rotation_euler", frame=0)

    # Keyframe at last frame
    last = max(0, num_frames - 1)
    bpy.context.scene.frame_set(last)
    obj.location = end_loc
    obj.rotation_euler = end_rot
    obj.keyframe_insert(data_path="location", frame=last)
    obj.keyframe_insert(data_path="rotation_euler", frame=last)

    # Make motion linear (optional but usually desired for validation)
    if obj.animation_data and obj.animation_data.action:
        for fcu in obj.animation_data.action.fcurves:
            for kp in fcu.keyframe_points:
                kp.interpolation = 'LINEAR'

# Function to set object visibility
def set_object_visibility(object_name, visibility):
    obj = bpy.data.objects.get(object_name)
    if obj:
        obj.hide_viewport = not visibility
        obj.hide_render = not visibility
    else:
        raise ValueError(f"Object named '{object_name}' not found")

# Function to get mesh vertices and edges with evaluated modifiers
def get_mesh_data(obj):
    """Extract vertices and edges from a mesh object with evaluated modifiers and animation"""
    if obj.type != 'MESH':
        return None, None
    
    # CRITICAL: The object's matrix_world property should already include animation transforms
    # after frame_set() and view_layer.update(). We need to ensure the depsgraph reflects this.
    
    # Update view layer to ensure animation is evaluated
    bpy.context.view_layer.update()
    
    # Get the object's current transform matrix (includes animation)
    # This is the transform that should be applied to the mesh
    obj_matrix_world = obj.matrix_world.copy()
    
    # Create a depsgraph to evaluate the object (for modifiers, but animation is already in matrix_world)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    
    # Get the evaluated mesh data (local coordinates, no transform)
    mesh = evaluated_obj.data
    
    # Get vertices (world coordinates) by applying the object's world matrix
    # This matrix includes the animation transform from keyframes
    vertices = []
    for vertex in mesh.vertices:
        # Apply the object's world transform (includes animation)
        world_coord = obj_matrix_world @ vertex.co
        vertices.append([world_coord.x, world_coord.y, world_coord.z])
    
    # Get edges from the evaluated mesh
    edges = []
    for edge in mesh.edges:
        edges.append([edge.vertices[0], edge.vertices[1]])
    
    return np.array(vertices), np.array(edges)

# Function to get mesh statistics
def get_mesh_statistics(obj):
    """Get statistics about the mesh"""
    if obj.type != 'MESH':
        return None
    
    mesh = obj.data
    num_vertices = len(mesh.vertices)
    num_edges = len(mesh.edges)
    num_faces = len(mesh.polygons)
    
    return {
        'vertices': num_vertices,
        'edges': num_edges,
        'faces': num_faces
    }

# Function to get memory usage (simplified, no logging)
def get_memory_mb(arr):
    """Get memory usage in MB for an array"""
    if arr is None:
        return 0
    return arr.nbytes / (1024*1024)

# Set up rendering (only if not skipping renders)
if not SKIP_RENDERS:
    # Clear existing cameras before adding new ones
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj)

    # Create cameras once and store them
    cameras = []
    for camera_key, camera_data in camera_params.items():
        if camera_key.startswith("Camera"):
            # Create camera using data-level API (doesn't require operator context)
            camera_data_block = bpy.data.cameras.new(name=camera_key)
            camera_data_block.angle = 0.69  # Set the horizontal FOV in radians
            # Create camera object
            camera_obj = bpy.data.objects.new(name=camera_key, object_data=camera_data_block)
            # Link to scene
            bpy.context.scene.collection.objects.link(camera_obj)
            # Set transform matrix
            transform_matrix = Matrix(camera_data['transform_matrix'])
            camera_obj.matrix_world = transform_matrix
            cameras.append(camera_obj)

    # Set up rendering settings
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Enable transparency
    bpy.context.scene.render.resolution_x = 1360
    bpy.context.scene.render.resolution_y = 1360
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.film_transparent = True  # Transparent background

    # Optimize rendering engine for speed - using Eevee Next (real-time renderer - much faster)
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    # Eevee Next settings (may have different API, so we'll try to set what's available)
    try:
        if hasattr(bpy.context.scene, 'eevee'):
            bpy.context.scene.eevee.taa_render_samples = EEVEE_SAMPLES
            bpy.context.scene.eevee.use_bloom = False  # Disable bloom for speed
            bpy.context.scene.eevee.use_ssr = False  # Disable screen space reflections for speed
            bpy.context.scene.eevee.use_ssr_refraction = False
            bpy.context.scene.eevee.use_soft_shadows = False  # Disable soft shadows for speed
            bpy.context.scene.eevee.use_volumetric_shadows = False
            bpy.context.scene.eevee.use_gtao = False  # Disable ambient occlusion for speed
            bpy.context.scene.eevee.use_motion_blur = False  # Disable motion blur for speed
    except AttributeError:
        # Eevee Next may have different settings structure, continue with defaults
        pass

    # Disable unnecessary features for speed
    bpy.context.scene.render.use_motion_blur = False
    bpy.context.scene.render.use_simplify = False
else:
    # Initialize cameras as empty list if skipping renders
    cameras = []

# Function to perform the rendering process
def render_scene(output_dir, json_file_path, object_to_unhide, object_to_hide, frames_per_camera=1):
    """Render scene with specified objects visible/hidden and save images and camera transforms
    
    Args:
        output_dir: Directory to save rendered images
        json_file_path: Path to save JSON file with camera transforms
        object_to_unhide: Object to make visible
        object_to_hide: Object to hide
        frames_per_camera: Number of frames to render per camera (1 for chessboard, num_frames for vessel)
    """
    # Hide and unhide appropriate objects
    set_object_visibility(object_to_unhide, True)
    set_object_visibility(object_to_hide, False)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the imported object
    imported_object = bpy.data.objects.get(object_to_unhide)
    if not imported_object:
        raise ValueError(f"Object named '{object_to_unhide}' not found")

    # CRITICAL: Clear all animation data from the scene to prevent interference
    clear_all_scene_animations()
    
    # CRITICAL: Disable all modifiers on the object for rigid rendering/extraction
    mute_all_modifiers(imported_object)

    # Store original transform for restoration at the end
    original = store_original_transform(imported_object)

    # Set up rigid motion if processing the vessel AND rendering multiple frames
    if (object_to_unhide == vessel_name) and (frames_per_camera > 1) and RIGID_MODE:
        setup_rigid_keyframes(imported_object, frames_per_camera, RIGID_TRANSLATION, RIGID_ROT_DEG, RIGID_SPACE)
        
        # If SETUP_ONLY is True, just set up keyframes and return
        if SETUP_ONLY:
            restore_original_transform(imported_object, original)
            log_write("Rigid motion keyframes set up. Scrub timeline to preview.")
            return

    # Prepare JSON data
    json_data = {
        "camera_angle_x": cameras[0].data.angle if cameras else 0,
        "frames": []
    }

    # Capture frames
    time_counter = 0
    total_frames = frames_per_camera * len(cameras)
    time_values = [i / (total_frames - 1) if total_frames > 1 else 0 for i in range(total_frames)]

    # For chessboard (single frame), set frame to 0 and don't loop
    frames_to_render = frames_per_camera if frames_per_camera > 1 else 1
    
    for frame in range(frames_to_render):
        if frames_per_camera > 1:
            bpy.context.scene.frame_set(frame)
        else:
            # For single frame (chessboard), use frame 0
            bpy.context.scene.frame_set(0)


        for idx, camera in enumerate(cameras):
            bpy.context.scene.camera = camera
            frame_name = f"camera_{idx:02d}_frame_{time_counter:04d}"
            file_path = os.path.join(output_dir, frame_name + ".png")
            bpy.context.scene.render.filepath = file_path
            bpy.ops.render.render(write_still=True)

            # Get camera properties and add to JSON data
            transform_matrix = camera.matrix_world
            transform_matrix_list = [[transform_matrix[j][i] for i in range(4)] for j in range(4)]
            
            # Determine the correct file path prefix based on folder name
            folder_name = os.path.basename(output_dir.rstrip('/'))
            frame_data = {
                "file_path": f"./{folder_name}/{frame_name}",
                "rotation": 0.0,
                "time": time_values[time_counter] if time_counter < len(time_values) else 0,
                "transform_matrix": transform_matrix_list
            }
            json_data["frames"].append(frame_data)

            time_counter += 1

    # Save JSON data to file
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    # Restore original transform
    restore_original_transform(imported_object, original)

    log_write(f"Rendering complete. Images saved to {output_dir}")
    log_write(f"JSON data saved to {json_file_path}")

# Function to perform the data extraction process
def extract_mesh_data(object_to_unhide, object_to_hide):
    # Hide and unhide appropriate objects
    set_object_visibility(object_to_unhide, True)
    set_object_visibility(object_to_hide, False)

    # Select the imported object
    imported_object = bpy.data.objects.get(object_to_unhide)

    if not imported_object:
        raise ValueError(f"Object named '{object_to_unhide}' not found")

    # CRITICAL: Clear all animation data from the scene to prevent interference
    clear_all_scene_animations()
    
    # CRITICAL: Disable all modifiers on the vessel for rigid transform export
    mute_all_modifiers(imported_object)

    # Get initial mesh statistics
    mesh_stats = get_mesh_statistics(imported_object)
    if mesh_stats:
        log_write(f"Mesh Statistics for {object_to_unhide}:")
        log_write(f"  Vertices: {mesh_stats['vertices']}")
        log_write(f"  Edges: {mesh_stats['edges']}")
        log_write(f"  Faces: {mesh_stats['faces']}")

    # Set up rigid motion if processing the vessel
    if (object_to_unhide == vessel_name) and RIGID_MODE:
        setup_rigid_keyframes(imported_object, num_frames, RIGID_TRANSLATION, RIGID_ROT_DEG, RIGID_SPACE)

    # Capture frames
    time_counter = 0
    total_frames = num_frames
    time_values = [i / (total_frames - 1) if total_frames > 1 else 0 for i in range(total_frames)]

    # Store metrics for this frame
    frame_metrics = []
    
    # Arrays to store efficient single-file data
    all_vertices_coords = []
    all_frame_numbers = []
    all_times = []
    
    # Extract mesh data for all frames
    log_write(f"\nExtracting mesh data for {num_frames} frames...")

    for frame in range(num_frames):
        # Set the frame - this should update the animation
        bpy.context.scene.frame_set(frame)
        
        # CRITICAL: Update view layer to ensure animation is evaluated
        bpy.context.view_layer.update()
        
        # Progress indicator every 10 frames
        if frame % 10 == 0 or frame == num_frames - 1:
            log_write(f"Processing frame {frame}/{num_frames - 1}...")

        # Get mesh data for this frame (with evaluated modifiers and animation)
        vertices, edges = get_mesh_data(imported_object)
        
        if vertices is not None:
            # Convert to float32 for memory efficiency
            vertices_float32 = vertices.astype(np.float32)
            
            # Store coordinates, frame numbers, and times
            all_vertices_coords.append(vertices_float32)
            all_frame_numbers.extend([frame] * len(vertices))  # Each vertex belongs to this frame
            all_times.append(time_values[time_counter])
            
            # Calculate metrics for this frame (for summary)
            vertices_memory = get_memory_mb(vertices)
            edges_memory = get_memory_mb(edges)
            vertices_memory_float32 = get_memory_mb(vertices_float32)
            
            frame_metric = {
                'frame': frame,
                'time': time_values[time_counter],
                'num_vertices': len(vertices),
                'num_edges': len(edges) if edges is not None else 0,
                'vertices_shape': vertices.shape,
                'edges_shape': edges.shape if edges is not None else (0, 0),
                'vertices_memory_mb': vertices_memory,
                'edges_memory_mb': edges_memory,
                'vertices_float32_memory_mb': vertices_memory_float32
            }
            frame_metrics.append(frame_metric)

        time_counter += 1

    # Save single efficient file
    log_write(f"\nSaving mesh data to file...")
    
    if all_vertices_coords:
        # Create single data structure
        coords_array = np.vstack(all_vertices_coords)
        frame_numbers_array = np.array(all_frame_numbers, dtype=np.int32)
        times_array = np.array(all_times, dtype=np.float32)
        
        # Capture edges and initial lengths from the last evaluated frame's edges 
        # (assuming constant topology for the rigid transform)
        # We'll use the first frame's data for initial lengths
        first_frame_vertices = all_vertices_coords[0]
        _, edges_array = get_mesh_data(imported_object) # Topology is constant
        
        # Calculate initial lengths from frame 0
        if edges_array is not None and len(edges_array) > 0:
            v0 = first_frame_vertices[edges_array[:, 0]]
            v1 = first_frame_vertices[edges_array[:, 1]]
            initial_lengths_array = np.linalg.norm(v0 - v1, axis=1)
        else:
            initial_lengths_array = np.array([])

        # Create single file with all data
        single_file_path = os.path.join(blender_data_dir, 'mesh_data_single.npz')
        np.savez_compressed(single_file_path, 
                           coords=coords_array,
                           frame_numbers=frame_numbers_array,
                           times=times_array,
                           edges=edges_array if edges_array is not None else np.array([]),
                           initial_lengths=initial_lengths_array)
        
        # Calculate memory usage
        coords_memory = coords_array.nbytes / (1024*1024)
        frame_numbers_memory = frame_numbers_array.nbytes / (1024*1024)
        times_memory = times_array.nbytes / (1024*1024)
        edges_memory = edges_array.nbytes / (1024*1024) if edges_array is not None else 0
        total_memory = coords_memory + frame_numbers_memory + times_memory + edges_memory
        
        log_write(f"Saved: {single_file_path}")
        log_write(f"  Total memory: {total_memory:.2f} MB ({coords_memory:.2f} MB coords, {edges_memory:.2f} MB edges)")
        log_write(f"  Shape: {coords_array.shape[0]:,} vertices, {len(edges_array) if edges_array is not None else 0:,} edges")

    # Save detailed metrics to text file
    metrics_file_path = os.path.join(blender_data_dir, f'single_file_analysis_{object_to_unhide.replace(" ", "_")}.txt')
    with open(metrics_file_path, 'w') as f:
        f.write(f"Single-File Efficient Storage Analysis for: {object_to_unhide}\n")
        f.write("=" * 60 + "\n\n")
        
        if mesh_stats:
            f.write(f"Initial Mesh Statistics:\n")
            f.write(f"  Vertices: {mesh_stats['vertices']}\n")
            f.write(f"  Edges: {mesh_stats['edges']}\n")
            f.write(f"  Faces: {mesh_stats['faces']}\n\n")
        
        # Calculate projected memory for full dataset
        if frame_metrics:
            f.write(f"Projected Memory for Full Dataset (160 frames):\n")
            f.write("-" * 50 + "\n")
            avg_vertices_memory = sum(m['vertices_float32_memory_mb'] for m in frame_metrics) / len(frame_metrics)
            f.write(f"  Float32 vertices (160 frames): {avg_vertices_memory * 160:.2f} MB = {avg_vertices_memory * 160 / 1024:.2f} GB\n")
    
    # Restore original transform
    restore_original_transform(imported_object, original)
    
    log_write(f"Detailed analysis saved to {metrics_file_path}")

# Try to toggle console to make sure it's visible (if available)
# NOTE: To see print statements in Blender:
#   - Windows: Window > Toggle System Console (or run Blender from command prompt)
#   - Mac: Run Blender from Terminal, or Window > Toggle System Console
#   - Linux: Run Blender from terminal, or Window > Toggle System Console
#   - Alternatively: Switch to Scripting workspace and check the console panel
try:
    if hasattr(bpy.ops.wm, 'console_toggle'):
        bpy.ops.wm.console_toggle()
except (AttributeError, RuntimeError):
    # Console toggle not available in this context, skip it
    pass

# Extract mesh data for the vessel (if enabled)
if EXPORT_MESH_DATA:
    extract_mesh_data(vessel_name, chessboard_name)

# Render scenes (if not skipped)
if not SKIP_RENDERS:
    # Render the chessboard first (1 image per camera)
    set_object_visibility(text_name_1, True)
    set_object_visibility(text_name_2, True)
    render_scene(output_dirs[0], json_file_paths[0], chessboard_name, vessel_name, frames_per_camera=1)

    # Render the vessel second using the same cameras (num_frames images per camera with rigid motion)
    set_object_visibility(text_name_1, False)
    set_object_visibility(text_name_2, False)
    render_scene(output_dirs[1], json_file_paths[1], vessel_name, chessboard_name, frames_per_camera=num_frames)
else:
    log_write("Skipping renders (SKIP_RENDERS = True)")

# Close log file at the end
log_write(f"Log file saved to: {log_file_path}")
log_file.close()