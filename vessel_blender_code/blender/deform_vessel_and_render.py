from pathlib import Path
import getpass
import os
import json
import math
import platform
from datetime import datetime
import bpy
import numpy as np
from mathutils import Matrix

# Parameters
num_frames = 160  # Number of frames to capture per camera
EXPORT_MESH_DATA = True  # If True, save and export mesh data. If False, skip mesh data extraction
SKIP_RENDERS = True  # If True, skip all rendering operations (only extract mesh data if enabled)
USE_OLD_DEFORMS = False  # If True, use old displacement/twist deformations. If False, use lattice deformations

# Lattice deformation strength levels - script will run once for each value
# Each run will deform from 0 to the specified maximum strength
LATTICE_DEFORM_STRENGTHS = [0.25, 0.5, 0.75, 1.0, 1.25]  # List of maximum deformation strengths to test

# Rendering optimization settings
EEVEE_SAMPLES = 1  # Number of samples for Eevee (lower = faster, higher = better quality)
# MAIN SPEED OPTIMIZATION: SUBDIVISION_RENDER_LEVELS = 0
# Setting this to 0 disables the SubdivisionSurface modifier during rendering, which was the main bottleneck.
# SubdivisionSurface multiplies geometry (each face becomes 4 faces at level 1, 16 at level 2, etc.),
# dramatically increasing polygon count and render time. Disabling it can make renders 10-100x faster.
# The viewport still uses subdivision for preview (levels = 1), but renders use the original geometry.
SUBDIVISION_RENDER_LEVELS = 0  # Subdivision levels for rendering (0 = disabled, 1+ = enabled). Lower = faster rendering

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
    base_output_dir_template = f"C:/Users/{username}/Documents/Blender/Vessel-Renders/{timestamp}/"
elif is_mac:
    base_output_dir_template = str(Path.home() / f"Documents/Blender/Vessel-Renders/{timestamp}/")
else:
    raise OSError(f"Unsupported operating system: {platform.system()}")

# Load camera parameters from JSON file (only needed if rendering)
if not SKIP_RENDERS:
    with open(json_file_path, 'r') as f:
        camera_params = json.load(f)

# Object names
chessboard_name = 'chessboard_6x8'
vessel_name = 'moderate tortuosity ICA'
text_name_1 = "Text.001"
text_name_2 = "Text.002"

# Function to set object visibility
def set_object_visibility(object_name, visibility):
    obj = bpy.data.objects.get(object_name)
    if obj:
        obj.hide_viewport = not visibility
        obj.hide_render = not visibility
    else:
        raise ValueError(f"Object named '{object_name}' not found")

# Function to clear object animation
def clear_object_animation(obj):
    """Clear all animation data from an object to prevent unwanted transforms."""
    obj.animation_data_clear()
    # Also clear any leftover NLA tracks if present
    if obj.animation_data and obj.animation_data.nla_tracks:
        for tr in obj.animation_data.nla_tracks:
            obj.animation_data.nla_tracks.remove(tr)

# Function to get mesh vertices and edges with evaluated modifiers
def get_mesh_data(obj):
    """Extract vertices and edges from a mesh object with evaluated modifiers"""
    if obj.type != 'MESH':
        return None, None
    
    # CRITICAL: Update view layer to ensure any frame changes are evaluated
    bpy.context.view_layer.update()
    
    # Get the object's current transform matrix (should be static if animation is cleared)
    obj_matrix_world = obj.matrix_world.copy()
    
    # Create a depsgraph to evaluate the object with all modifiers
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    
    # Get the evaluated mesh data (local coordinates)
    mesh = evaluated_obj.data
    
    # Get vertices (world coordinates) by applying the object's world matrix
    # Use obj_matrix_world to ensure we're using the object's actual transform, not evaluated transform
    vertices = []
    for vertex in mesh.vertices:
        # Apply the object's world transform
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

# Function to analyze numpy array details
def analyze_array_details(arr, name):
    """Analyze and print detailed information about a numpy array"""
    if arr is None:
        print(f"{name}: None")
        return 0
    
    print(f"\n{name} Analysis:")
    print(f"  Shape: {arr.shape}")
    print(f"  Data type: {arr.dtype}")
    print(f"  Item size: {arr.itemsize} bytes")
    print(f"  Total elements: {arr.size}")
    print(f"  Memory usage: {arr.nbytes / (1024*1024):.2f} MB")
    print(f"  Memory per element: {arr.nbytes / arr.size:.2f} bytes")
    
    # Show first few elements
    if arr.size > 0:
        print(f"  First 3 elements: {arr[:3] if len(arr.shape) == 1 else arr[:3].tolist()}")
        print(f"  Last 3 elements: {arr[-3:] if len(arr.shape) == 1 else arr[-3:].tolist()}")
    
    return arr.nbytes / (1024*1024)  # Return memory in MB

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
    bpy.context.scene.render.resolution_x = 1300
    bpy.context.scene.render.resolution_y = 1300
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
def render_scene(output_dir, json_file_path, object_to_unhide, object_to_hide, frames_per_camera=1, max_deform_strength=0.5):
    """Render scene with specified objects visible/hidden and save images and camera transforms
    
    Args:
        output_dir: Directory to save rendered images
        json_file_path: Path to save JSON file with camera transforms
        object_to_unhide: Object to make visible
        object_to_hide: Object to hide
        frames_per_camera: Number of frames to render per camera (1 for chessboard, num_frames for vessel)
        max_deform_strength: Maximum lattice deformation strength (0 to this value over frames)
    """
    # Hide and unhide appropriate objects
    set_object_visibility(object_to_unhide, True)
    set_object_visibility(object_to_hide, False)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the imported object to apply deformations if needed
    imported_object = bpy.data.objects.get(object_to_unhide)
    if not imported_object:
        raise ValueError(f"Object named '{object_to_unhide}' not found")

    # Apply deformations only if processing the vessel AND rendering multiple frames
    apply_deformations = (object_to_unhide == vessel_name) and (frames_per_camera > 1)

    if apply_deformations:
        # Optimize subdivision modifier for rendering speed
        # SubdivisionSurface dramatically increases geometry and render time
        if 'SubdivisionSurface' in imported_object.modifiers:
            subsurf_modifier = imported_object.modifiers['SubdivisionSurface']
        else:
            subsurf_modifier = imported_object.modifiers.new(name='SubdivisionSurface', type='SUBSURF')
            subsurf_modifier.levels = 1  # Viewport level (for preview)
        # Set render levels (0 = disabled for fastest rendering, 1+ = enabled)
        # Setting to 0 can make renders 10-100x faster depending on mesh complexity
        subsurf_modifier.render_levels = SUBDIVISION_RENDER_LEVELS
        
        # Disable subdivision modifier entirely if render_levels is 0 (for maximum speed)
        if SUBDIVISION_RENDER_LEVELS == 0:
            subsurf_modifier.show_render = False
        else:
            subsurf_modifier.show_render = True

        if USE_OLD_DEFORMS:
            # Ensure displacement modifier exists
            if 'Displace' in imported_object.modifiers:
                displace_modifier = imported_object.modifiers['Displace']
            else:
                displace_modifier = imported_object.modifiers.new(name='Displace', type='DISPLACE')
                texture = bpy.data.textures.new(name='DisplaceTexture', type='CLOUDS')
                displace_modifier.texture = texture
                displace_modifier.strength = 0.5  # Initial strength

            # Ensure twist modifier exists
            if 'SimpleDeform_TWIST' in imported_object.modifiers:
                twist_modifier = imported_object.modifiers['SimpleDeform_TWIST']
            else:
                twist_modifier = imported_object.modifiers.new(name='SimpleDeform_TWIST', type='SIMPLE_DEFORM')
                twist_modifier.deform_method = 'TWIST'
        else:
            # Ensure SimpleDeform_LATTICE modifier exists
            if 'SimpleDeform_LATTICE' in imported_object.modifiers:
                lattice_modifier = imported_object.modifiers['SimpleDeform_LATTICE']
            else:
                lattice_modifier = imported_object.modifiers.new(name='SimpleDeform_LATTICE', type='SIMPLE_DEFORM')
                lattice_modifier.deform_method = 'LATTICE'

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

        # Apply frame-specific deformation if processing the vessel with multiple frames
        if apply_deformations:
            time_val = time_values[time_counter] if time_counter < len(time_values) else 0
            if USE_OLD_DEFORMS:
                # Old deformations: displacement and twist
                displace_modifier.strength = 0.5 + 0.2 * math.sin(time_val * math.pi / 2)
                twist_modifier.angle = math.radians(360 / 16) * math.sin(time_val * math.pi / 2)
            else:
                # Lattice_DEFORM strength goes from 0 to max_deform_strength as render continues
                lattice_modifier.strength = max_deform_strength * time_val

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

    # Reset modifiers back to 0/initial after rendering is complete
    if apply_deformations:
        if USE_OLD_DEFORMS:
            if 'Displace' in imported_object.modifiers:
                imported_object.modifiers['Displace'].strength = 0.5
            if 'SimpleDeform_TWIST' in imported_object.modifiers:
                imported_object.modifiers['SimpleDeform_TWIST'].angle = 0.0
        else:
            if 'SimpleDeform_LATTICE' in imported_object.modifiers:
                imported_object.modifiers['SimpleDeform_LATTICE'].strength = 0.0

    print(f"Rendering complete. Images saved to {output_dir}")
    print(f"JSON data saved to {json_file_path}")

# Function to perform the data extraction process
def extract_mesh_data(object_to_unhide, object_to_hide, blender_data_dir, max_deform_strength=0.5):
    # Hide and unhide appropriate objects
    set_object_visibility(object_to_unhide, True)
    set_object_visibility(object_to_hide, False)

    # Select the imported object
    imported_object = bpy.data.objects.get(object_to_unhide)

    if not imported_object:
        raise ValueError(f"Object named '{object_to_unhide}' not found")

    # CRITICAL: Clear any existing animation data to prevent unwanted translation/rotation
    # This ensures the object stays in place and only deforms
    clear_object_animation(imported_object)
    print(f"Cleared any existing animation data from {object_to_unhide}")

    # Get initial mesh statistics
    mesh_stats = get_mesh_statistics(imported_object)
    if mesh_stats:
        print(f"Mesh Statistics for {object_to_unhide}:")
        print(f"  Vertices: {mesh_stats['vertices']}")
        print(f"  Edges: {mesh_stats['edges']}")
        print(f"  Faces: {mesh_stats['faces']}")

    # Apply deformations only if processing the vessel
    apply_deformations = object_to_unhide == vessel_name

    if apply_deformations:
        # Optimize subdivision modifier for rendering speed
        if 'SubdivisionSurface' in imported_object.modifiers:
            subsurf_modifier = imported_object.modifiers['SubdivisionSurface']
        else:
            subsurf_modifier = imported_object.modifiers.new(name='SubdivisionSurface', type='SUBSURF')
            subsurf_modifier.levels = 1  # Viewport level (for preview)
        # Set render levels (0 = disabled for fastest rendering, 1+ = enabled)
        subsurf_modifier.render_levels = SUBDIVISION_RENDER_LEVELS
        
        # Disable subdivision modifier entirely if render_levels is 0 (for maximum speed)
        if SUBDIVISION_RENDER_LEVELS == 0:
            subsurf_modifier.show_render = False
        else:
            subsurf_modifier.show_render = True

        if USE_OLD_DEFORMS:
            # Ensure displacement modifier exists
            if 'Displace' in imported_object.modifiers:
                displace_modifier = imported_object.modifiers['Displace']
            else:
                displace_modifier = imported_object.modifiers.new(name='Displace', type='DISPLACE')
                texture = bpy.data.textures.new(name='DisplaceTexture', type='CLOUDS')
                displace_modifier.texture = texture
                displace_modifier.strength = 0.5  # Initial strength

            # Ensure twist modifier exists
            if 'SimpleDeform_TWIST' in imported_object.modifiers:
                twist_modifier = imported_object.modifiers['SimpleDeform_TWIST']
            else:
                twist_modifier = imported_object.modifiers.new(name='SimpleDeform_TWIST', type='SIMPLE_DEFORM')
                twist_modifier.deform_method = 'TWIST'
        else:
            # Ensure SimpleDeform_LATTICE modifier exists
            if 'SimpleDeform_LATTICE' in imported_object.modifiers:
                lattice_modifier = imported_object.modifiers['SimpleDeform_LATTICE']
            else:
                lattice_modifier = imported_object.modifiers.new(name='SimpleDeform_LATTICE', type='SIMPLE_DEFORM')
                lattice_modifier.deform_method = 'LATTICE'

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
    
    # Test numpy array creation and analysis
    print(f"\n{'='*60}")
    print(f"SINGLE-FILE EFFICIENT STORAGE ANALYSIS FOR {object_to_unhide}")
    print(f"{'='*60}")

    for frame in range(num_frames):
        bpy.context.scene.frame_set(frame)
        
        # CRITICAL: Update view layer to ensure frame change is evaluated
        # This ensures modifiers are properly evaluated for the current frame
        bpy.context.view_layer.update()

        # Apply frame-specific deformation if processing the vessel
        if apply_deformations:
            time_val = time_values[time_counter] if time_counter < len(time_values) else 0
            if USE_OLD_DEFORMS:
                # Old deformations: displacement and twist
                displace_modifier.strength = 0.5 + 0.2 * math.sin(time_val * math.pi / 2)
                twist_modifier.angle = math.radians(360 / 16) * math.sin(time_val * math.pi / 2)
                print(f"Applied deformations for frame {frame}:")
                print(f"  Displacement strength: {displace_modifier.strength:.3f}")
                print(f"  Twist angle: {math.degrees(twist_modifier.angle):.1f} degrees")
            else:
                # Lattice_DEFORM strength goes from 0 to max_deform_strength as render continues
                lattice_modifier.strength = max_deform_strength * time_val
                print(f"Applied deformations for frame {frame}:")
                print(f"  Lattice_DEFORM strength: {lattice_modifier.strength:.3f} (max: {max_deform_strength})")

        # Get mesh data for this frame (with evaluated modifiers)
        vertices, edges = get_mesh_data(imported_object)
        
        print(f"\nFrame {frame} Analysis:")
        print(f"Time value: {time_values[time_counter]}")
        
        # Analyze raw vertices and edges
        vertices_memory = analyze_array_details(vertices, "Raw Vertices")
        edges_memory = analyze_array_details(edges, "Raw Edges")
        
        if vertices is not None:
            # Single-file efficient storage approach
            print(f"\n--- Single-File Efficient Storage Approach ---")
            
            # Convert to float32 for memory efficiency
            vertices_float32 = vertices.astype(np.float32)
            vertices_memory_float32 = analyze_array_details(vertices_float32, "Float32 Vertices")
            
            # Store coordinates, frame numbers, and times
            all_vertices_coords.append(vertices_float32)
            all_frame_numbers.extend([frame] * len(vertices))  # Each vertex belongs to this frame
            all_times.append(time_values[time_counter])
            
            # Calculate metrics for this frame
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
            
            print(f"\n--- Memory Analysis ---")
            print(f"Raw vertices: {vertices_memory:.2f} MB")
            print(f"Raw edges: {edges_memory:.2f} MB")
            print(f"Float32 vertices: {vertices_memory_float32:.2f} MB")
            
            # Project to 160 frames
            print(f"\n--- Projected Memory for 160 Frames ---")
            print(f"Float32 vertices (160 frames): {vertices_memory_float32 * 160:.2f} MB = {vertices_memory_float32 * 160 / 1024:.2f} GB")

        time_counter += 1

    # Save single efficient file
    print(f"\n--- Saving Single Efficient File ---")
    
    if all_vertices_coords:
        # Create single data structure
        coords_array = np.vstack(all_vertices_coords)
        frame_numbers_array = np.array(all_frame_numbers, dtype=np.int32)
        times_array = np.array(all_times, dtype=np.float32)
        
        # Capture edges and initial lengths from the first evaluated frame 
        # (assuming constant topology for the deformation)
        first_frame_vertices = all_vertices_coords[0]
        _, edges_array = get_mesh_data(imported_object) # Topology is constant
        
        # Calculate initial lengths from frame 0
        if edges_array is not None and len(edges_array) > 0:
            v0 = first_frame_vertices[edges_array[:, 0]]
            v1 = first_frame_vertices[edges_array[:, 1]]
            initial_lengths_array = np.linalg.norm(v0 - v1, axis=1)
        else:
            initial_lengths_array = np.array([])
            
        # Create single file with all data (include strength in filename)
        strength_str = f"{max_deform_strength:.2f}".replace('.', '_')
        single_file_path = os.path.join(blender_data_dir, f'mesh_data_single.npz')
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
        
        print(f"Single file saved: {single_file_path}")
        print(f"  Coordinates: shape {coords_array.shape}, memory: {coords_memory:.2f} MB")
        print(f"  Edges: shape {edges_array.shape if edges_array is not None else (0,)}, memory: {edges_memory:.2f} MB")
        print(f"  Total memory: {total_memory:.2f} MB")
        
        # Project to 160 frames
        projected_memory_160 = total_memory * 160
        print(f"  Projected 160 frames: {projected_memory_160:.2f} MB = {projected_memory_160 / 1024:.2f} GB")
        
        # Show how to reconstruct data
        print(f"\n--- Data Reconstruction Example ---")
        print(f"To get time+XYZ for vertex 1000:")
        print(f"  frame_num = frame_numbers[1000]  # Which frame this vertex belongs to")
        print(f"  time_val = times[frame_num]      # Get time for that frame")
        print(f"  coords_val = coords[1000]        # Get coordinates")
        print(f"  result = [time_val, coords_val[0], coords_val[1], coords_val[2]]")

    # Save detailed metrics to text file (include strength in filename)
    strength_str = f"{max_deform_strength:.2f}".replace('.', '_')
    metrics_file_path = os.path.join(blender_data_dir, f'single_file_analysis_{object_to_unhide.replace(" ", "_")}_strength_{strength_str}.txt')
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
    
    print(f"Detailed analysis saved to {metrics_file_path}")

# Try to toggle console to make sure it's visible (if available)
try:
    if hasattr(bpy.ops.wm, 'console_toggle'):
        bpy.ops.wm.console_toggle()
except (AttributeError, RuntimeError):
    # Console toggle not available in this context, skip it
    pass

# Main execution loop - run once for each deformation strength level or once for old deforms
active_runs = [0.0] if USE_OLD_DEFORMS else LATTICE_DEFORM_STRENGTHS

print(f"\n{'='*70}")
print(f"STARTING {'OLD DEFORMATION' if USE_OLD_DEFORMS else 'MULTI-RUN LATTICE'} ANALYSIS")
print(f"{'='*70}")
print(f"Number of runs: {len(active_runs)}")
if not USE_OLD_DEFORMS:
    print(f"Deformation strengths: {active_runs}")
print(f"Frames per run: {num_frames}")
print(f"{'='*70}\n")

for run_idx, max_strength in enumerate(active_runs, 1):
    print(f"\n{'='*70}")
    if USE_OLD_DEFORMS:
        print(f"RUN {run_idx}/{len(active_runs)}: Using Old Displacement/Twist Deformations")
    else:
        print(f"RUN {run_idx}/{len(active_runs)}: Max Lattice Deformation Strength = {max_strength}")
    print(f"{'='*70}\n")
    
    # Create run-specific output directories
    if USE_OLD_DEFORMS:
        run_base_dir = f"{base_output_dir_template}old_deforms/"
    else:
        strength_str = f"{max_strength:.2f}".replace('.', '_')
        run_base_dir = f"{base_output_dir_template}lattice_strength_{strength_str}/"
    
    if is_windows:
        run_output_dirs = [f"{run_base_dir}{name}/" for name in ["train_chessboard", "train_vessel"]]
        run_json_file_paths = [f"{run_base_dir}transforms_{name}.json" for name in ["train_chessboard", "train_vessel"]]
        run_blender_data_dir = f"{run_base_dir}blender-data/"
    elif is_mac:
        run_output_dirs = [f"{run_base_dir}{name}/" for name in ["train_chessboard", "train_vessel"]]
        run_json_file_paths = [f"{run_base_dir}transforms_{name}.json" for name in ["train_chessboard", "train_vessel"]]
        run_blender_data_dir = f"{run_base_dir}blender-data/"
    
    # Create blender-data directory for this run
    if not os.path.exists(run_blender_data_dir):
        os.makedirs(run_blender_data_dir)
    
    # Extract mesh data for the vessel (if enabled)
    if EXPORT_MESH_DATA:
        extract_mesh_data(vessel_name, chessboard_name, run_blender_data_dir, max_deform_strength=max_strength)

    # Render scenes (if not skipped)
    if not SKIP_RENDERS:
        # Render the chessboard first (1 image per camera, no deformations)
        set_object_visibility(text_name_1, True)
        set_object_visibility(text_name_2, True)
        render_scene(run_output_dirs[0], run_json_file_paths[0], chessboard_name, vessel_name, 
                    frames_per_camera=1, max_deform_strength=max_strength)

        # Render the vessel second using the same cameras (num_frames images per camera with deformations)
        set_object_visibility(text_name_1, False)
        set_object_visibility(text_name_2, False)
        render_scene(run_output_dirs[1], run_json_file_paths[1], vessel_name, chessboard_name, 
                    frames_per_camera=num_frames, max_deform_strength=max_strength)
    
    print(f"\n✓ Completed run {run_idx}/{len(active_runs)}")

print(f"\n{'='*70}")
print(f"ALL RUNS COMPLETE")
print(f"{'='*70}")
print(f"Total runs: {len(active_runs)}")
print(f"Base output directory template: {base_output_dir_template}")
print(f"Output folders created:")
if USE_OLD_DEFORMS:
    print(f"  - old_deforms/")
else:
    for strength in LATTICE_DEFORM_STRENGTHS:
        strength_str = f"{strength:.2f}".replace('.', '_')
        print(f"  - lattice_strength_{strength_str}/")
print(f"{'='*70}\n")

if SKIP_RENDERS:
    print("Rendering was skipped (SKIP_RENDERS = True)")