#!/usr/bin/env python3
"""
Grid visualization of multiple vessels showing edge stress, saved as a video.

This script:
1. Loads multiple GT and ours numpy files with edge connectivity
2. Computes stress for each edge (relative to initial lengths) using Neo-Hookean model
3. Arranges vessels in a grid layout
4. Iterates through all frames, applies cropping and a colorbar overlay
5. Saves the final result as a video file
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import pyvista as pv
import imageio.v2 as imageio
from PIL import Image

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import vessel_utils as vu

# ============================================================================
# VIDEO & OVERLAY CONFIGURATION
# ============================================================================
# Specify the output video format (e.g., '.mp4', '.avi', '.mov', '.mkv')
VIDEO_FORMAT = '.mp4'
# Frames per second for the output video
FPS = 30 

# Cropping/expansion settings (in pixels)
# Positive = crop that many pixels from that side.
# Negative = expand (add padding) that many pixels on that side.
# CROP_LEFT = 320
# CROP_RIGHT = 80
# CROP_TOP = 110
# CROP_BOTTOM = 0

CROP_LEFT = 240
CROP_RIGHT = 0
CROP_TOP = 70
CROP_BOTTOM = 70

# Camera zoom settings
# 1.0 = Default fit, >1.0 = Zoomed in (closer), <1.0 = Zoomed out (further)
# CAMERA_ZOOM = 1.3
CAMERA_ZOOM = 1.45

# Overlay settings
# Path to the image to overlay (e.g., a colorbar)
OVERLAY_IMAGE_PATH = 'tools/generated_colorbars/stress_mpa_0_15.png'
# Padding from the right edge
OVERLAY_PADDING_RIGHT = 10
# Scale of the overlay image (1.0 = original size, 0.5 = 50% size)
OVERLAY_SCALE = 0.6
# ============================================================================

# ============================================================================
# VESSEL CONFIGURATION
# ============================================================================
# Grid spacing multipliers - controls spacing between vessels
GRID_SPACING_MULTIPLIER_X = 1.1
GRID_SPACING_MULTIPLIER_Y = 1.1

NUM_ROWS = 2
NUM_COLS = 5

# GT vs RAW GT logic
USE_RAW_GT = True

GT_BASE = Path.home() / "Documents/Blender/Vessel-Renders"
if USE_RAW_GT:
    GT_FILES = [
        # str(GT_BASE / "synthetic_translate/blender-data/mesh_data_single.npz"),
        str(GT_BASE / "lattice_strength/lattice_strength_0_25/blender-data/mesh_data_single.npz"),
        str(GT_BASE / "lattice_strength/lattice_strength_0_50/blender-data/mesh_data_single.npz"),
        str(GT_BASE / "lattice_strength/lattice_strength_0_75/blender-data/mesh_data_single.npz"),
        str(GT_BASE / "lattice_strength/lattice_strength_1_00/blender-data/mesh_data_single.npz"),
        str(GT_BASE / "lattice_strength/lattice_strength_1_25/blender-data/mesh_data_single.npz"),
    ]
else:
    GT_FILES = [
        r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_25.npy",
        r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_50.npy",
        r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_75.npy",
        r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_1_00.npy",
        r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_1_25.npy",
    ]

# List of ours numpy file paths - Row 2
OURS_FILES = [
    # r"cluster_data/cluster_export_synthetic_translate.npy",
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_0_25.npy",
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_0_50.npy",
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_0_75.npy",
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_1_00.npy",
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_1_25.npy",
]

# Physical constants
YOUNG_MODULUS_SILICON = vu.YOUNG_MODULUS_SILICON

# Visualization constants
EDGE_LINE_WIDTH = 5
MAX_STRESS_PA = vu.MAX_STRESS_PA

# Camera position/orientation
CAMERA_POSITION = [
    [-5.358697, 3.419817, 18.256446],
    [-0.021277, -0.067798, 0.104289],
    [-0.857634, -0.489335, -0.158160]
]

# Resolution for the PyVista plotter
WINDOW_SIZE = [1920, 1080]
# ============================================================================

def _resolve_path(file_path: str, project_root: Path) -> Path:
    """Resolve a file path relative to project root or as absolute path."""
    if os.path.isabs(file_path):
        return Path(file_path)
    resolved = project_root / file_path
    if resolved.exists():
        return resolved
    return Path(file_path).resolve()

def compute_rotation_matrix_from_camera(camera_position: List[List[float]]) -> np.ndarray:
    """Compute rotation matrix from camera position/orientation."""
    if camera_position is None:
        return np.eye(3)
    
    cam_pos = np.array(camera_position[0])
    focal = np.array(camera_position[1])
    up = np.array(camera_position[2])
    
    view_dir = focal - cam_pos
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-9)
    up = up / (np.linalg.norm(up) + 1e-9)
    right = np.cross(view_dir, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, view_dir)
    up = up / (np.linalg.norm(up) + 1e-9)
    
    camera_to_world = np.column_stack([right, up, -view_dir])
    return camera_to_world.T

def load_cluster_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load cluster data with edge connectivity."""
    experiment_name = file_path.stem
    if file_path.suffix == '.npz':
        # Load raw data and create Delaunay edges
        data = np.load(file_path)
        coords = data['coords']
        frame_numbers = data['frame_numbers']
        times = data['times']
        frames_data = vu.get_all_frames_data(coords, frame_numbers, times)
        frame_indices = sorted(frames_data.keys())
        T = len(frame_indices)
        if T == 0: return np.array([]), np.array([]), np.array([]), experiment_name
        N = len(frames_data[frame_indices[0]]['coords'])
        cluster_positions = np.zeros((T, N, 3))
        for i, f_idx in enumerate(frame_indices):
            cluster_positions[i] = frames_data[f_idx]['coords']
            
        # Create edges for first frame
        _, edges_raw, _ = vu.create_delaunay_edges(cluster_positions[0])
        valid_edges, _ = vu.remove_outlier_edges(cluster_positions[0], edges_raw, 1.8)
        
        # Filter by absolute length limit (consistent with interactive script)
        init_lens = np.linalg.norm(
            cluster_positions[0][valid_edges[:, 0]] - cluster_positions[0][valid_edges[:, 1]], axis=1
        )
        abs_limit_internal = 1.0 / vu.UNIT_TO_MM
        mask = init_lens <= abs_limit_internal
        edges = valid_edges[mask]
        initial_lengths = init_lens[mask]
        
        return cluster_positions, edges, initial_lengths, experiment_name

    data = np.load(str(file_path), allow_pickle=True).item()
    cluster_positions = data.get("cluster_positions")
    edges = data.get("edges")
    initial_lengths = data.get("initial_lengths")
    experiment_name = data.get("experiment_name", file_path.stem)
    
    if cluster_positions is None:
        raise ValueError(f"File {file_path} missing 'cluster_positions'")
    if edges is None:
        edges = np.array([]).reshape(0, 2)
    if initial_lengths is None and len(edges) > 0:
        initial_lengths = np.linalg.norm(
            cluster_positions[0][edges[:, 0]] - cluster_positions[0][edges[:, 1]], axis=1
        )
    return cluster_positions, edges, initial_lengths, experiment_name

def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box for a set of points."""
    if len(points) == 0: return np.array([0,0,0]), np.array([0,0,0])
    return np.min(points, axis=0), np.max(points, axis=0)

def compute_grid_positions(all_bounding_boxes, spacing_multiplier_x, spacing_multiplier_y):
    """Compute grid positions for all vessels."""
    num_vessels = len(all_bounding_boxes)
    all_sizes = [box[1] - box[0] for box in all_bounding_boxes]
    avg_size = np.mean(all_sizes, axis=0)
    spacing_x = avg_size[0] * spacing_multiplier_x
    spacing_y = avg_size[1] * spacing_multiplier_y
    
    grid_positions = []
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            v_idx = row * NUM_COLS + col
            if v_idx < num_vessels:
                grid_positions.append(np.array([col * spacing_x, -row * spacing_y, 0]))
    return grid_positions

def create_grid_video():
    """Main function to generate the grid video showing stress."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 1. Load Data
    all_vessel_data = []
    all_bounding_boxes = []
    rotation_matrix = compute_rotation_matrix_from_camera(CAMERA_POSITION)

    print("Loading vessel data...")
    # Helper to load both sets
    for file_list, is_gt in [(GT_FILES, True), (OURS_FILES, False)]:
        for f_path_str in file_list:
            f_path = _resolve_path(f_path_str, project_root)
            if not f_path.exists(): continue
            pos, edges, init_lens, name = load_cluster_data(f_path)
            if CAMERA_POSITION: pos = pos @ rotation_matrix.T
            min_c, max_c = compute_bounding_box(pos.reshape(-1, 3))
            all_bounding_boxes.append((min_c, max_c))
            all_vessel_data.append({
                "pos": pos, "edges": edges, "init_lens": init_lens, 
                "name": name, "is_gt": is_gt
            })
            type_str = "GT" if is_gt else "Ours"
            print(f"  ✓ {type_str}: {name} ({len(edges)} edges)")

    if not all_vessel_data:
        print("Error: No data loaded")
        return

    num_frames = min(v["pos"].shape[0] for v in all_vessel_data)
    grid_positions = compute_grid_positions(all_bounding_boxes, GRID_SPACING_MULTIPLIER_X, GRID_SPACING_MULTIPLIER_Y)
    cmap = vu.get_colormap()

    # 2. Setup Plotter
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background("white")
    plotter.view_xy()
    plotter.camera.enable_parallel_projection()
    
    # 3. Load Overlay
    overlay_img = None
    overlay_path = _resolve_path(OVERLAY_IMAGE_PATH, project_root)
    if overlay_path.exists():
        overlay_img = Image.open(overlay_path).convert('RGBA')
        if OVERLAY_SCALE != 1.0:
            new_size = (int(overlay_img.width * OVERLAY_SCALE), int(overlay_img.height * OVERLAY_SCALE))
            overlay_img = overlay_img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"✓ Loaded overlay: {overlay_path}")

    # 4. Fit Camera
    all_translated_points = []
    for v_idx, vessel in enumerate(all_vessel_data):
        all_translated_points.append(vessel["pos"][0] + grid_positions[v_idx])
    combined_points = np.vstack(all_translated_points)
    plotter.add_mesh(pv.PolyData(combined_points), opacity=0)
    plotter.reset_camera()
    if CAMERA_ZOOM != 1.0:
        plotter.camera.parallel_scale /= CAMERA_ZOOM

    # 5. Process Frames
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"grid_stress_{num_frames}f_{timestamp}{VIDEO_FORMAT}"
    print(f"\nGenerating video: {output_path}")
    writer = imageio.get_writer(str(output_path), fps=FPS)

    try:
        for f_idx in range(num_frames):
            for v_idx, vessel in enumerate(all_vessel_data):
                translated_pos = vessel["pos"][f_idx] + grid_positions[v_idx]
                edges = vessel["edges"]
                init_lens = vessel["init_lens"]
                
                if len(edges) > 0:
                    stresses = vu.compute_edge_stress(
                        vessel["pos"][f_idx], edges, init_lens, YOUNG_MODULUS_SILICON
                    )
                    colors = vu.get_colors(stresses, cmap, vmin=0, vmax=MAX_STRESS_PA, use_abs=True)
                    
                    # Create PolyData with lines
                    lines = np.hstack([np.full((edges.shape[0], 1), 2), edges])
                    mesh = pv.PolyData(translated_pos, lines=lines.flatten())
                    
                    plotter.add_mesh(
                        mesh,
                        scalars=colors, rgb=True,
                        line_width=EDGE_LINE_WIDTH,
                        lighting=False,
                        name=f"vessel_{v_idx}"
                    )
            
            # Capture frame
            image_np = plotter.screenshot()
            
            # 6. Apply Cropping
            pad_left = -CROP_LEFT if CROP_LEFT < 0 else 0
            crop_left = CROP_LEFT if CROP_LEFT > 0 else 0
            pad_right = -CROP_RIGHT if CROP_RIGHT < 0 else 0
            crop_right = CROP_RIGHT if CROP_RIGHT > 0 else 0
            pad_top = -CROP_TOP if CROP_TOP < 0 else 0
            crop_top = CROP_TOP if CROP_TOP > 0 else 0
            pad_bottom = -CROP_BOTTOM if CROP_BOTTOM < 0 else 0
            crop_bottom = CROP_BOTTOM if CROP_BOTTOM > 0 else 0

            h, w = image_np.shape[:2]
            l_idx = min(crop_left, w - 1)
            r_idx = max(w - crop_right, l_idx + 1)
            t_idx = min(crop_top, h - 1)
            b_idx = max(h - crop_bottom, t_idx + 1)
            image_np = image_np[t_idx:b_idx, l_idx:r_idx, :]
            
            if any([pad_left, pad_right, pad_top, pad_bottom]):
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
                image_np = np.pad(image_np, pad_width, mode='edge')
            
            # 7. Apply Overlay
            if overlay_img:
                base_img = Image.fromarray(image_np).convert('RGBA')
                bw, bh = base_img.size
                ow, oh = overlay_img.size
                x_pos = bw - ow - OVERLAY_PADDING_RIGHT
                y_pos = (bh - oh) // 2
                base_img.paste(overlay_img, (x_pos, y_pos), overlay_img)
                final_image = np.array(base_img.convert('RGB'))
            else:
                final_image = image_np
            
            writer.append_data(final_image)

            # Progress
            progress = int(((f_idx + 1) / num_frames) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {f_idx + 1}/{num_frames}", end="", flush=True)

    finally:
        writer.close()
        plotter.close()

    print(f"\n✓ Saved video to: {output_path}")

if __name__ == "__main__":
    create_grid_video()
