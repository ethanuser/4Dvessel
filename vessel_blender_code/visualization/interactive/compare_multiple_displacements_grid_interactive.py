#!/usr/bin/env python3
"""
Interactive visualization of multiple vessels in a grid layout showing displacements.

This script:
1. Loads multiple GT and ours numpy files
2. Computes displacements for each vessel (relative to first frame)
3. Arranges vessels in a 2x5 grid (Row 1: GT files, Row 2: Ours files)
4. Displays all vessels with displacement coloring using a global colormap scale
5. Single frame slider controls all vessels simultaneously

Usage:
    python compare_multiple_displacements_grid_interactive.py
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import pyvista as pv

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.interactive_utils import create_interactive_visualization, create_default_instructions
import vessel_utils as vu

# ============================================================================
# CONFIGURATION
# ============================================================================

# Grid spacing multipliers - controls spacing between vessels
# Higher values = more spacing between vessels
GRID_SPACING_MULTIPLIER_X = 1.1  # Spacing multiplier for X direction (columns)
GRID_SPACING_MULTIPLIER_Y = 1.1  # Spacing multiplier for Y direction (rows)

NUM_ROWS = 2
NUM_COLS = 5

# List of GT numpy file paths (from blender) - Row 1
# GT_FILES = [
#     r"cluster_data/cluster_mesh_timeseries_e4.npy",
# ]

# OURS_FILES = [
#     r"cluster_data/cluster_mesh_timeseries_e5.npy",
# ]

# GT vs RAW GT logic
USE_RAW_GT = True

GT_BASE = Path.home() / "Documents/Blender/Vessel-Renders"
if USE_RAW_GT:
    GT_FILES = [
        str(GT_BASE / "synthetic_translate/blender-data/mesh_data_single.npz"),
        # str(GT_BASE / "lattice_strength/lattice_strength_0_25/blender-data/mesh_data_single.npz"),
        # str(GT_BASE / "lattice_strength/lattice_strength_0_50/blender-data/mesh_data_single.npz"),
        # str(GT_BASE / "lattice_strength/lattice_strength_0_75/blender-data/mesh_data_single.npz"),
        # str(GT_BASE / "lattice_strength/lattice_strength_1_00/blender-data/mesh_data_single.npz"),
        # str(GT_BASE / "lattice_strength/lattice_strength_1_25/blender-data/mesh_data_single.npz"),
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
    r"cluster_data/cluster_export_synthetic_translate.npy",
    # r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_0_25.npy",
    # r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_0_50.npy",
    # r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_0_75.npy",
    # r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_1_00.npy",
    # r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_1_25.npy",
]

# Physical constants for displacement calculation
CHESS_TOTAL_LENGTH_M = 6.195 / 100.0
CHESS_SQUARES_ALONG_X = 8
CHESS_SQUARE_LENGTH_M = CHESS_TOTAL_LENGTH_M / CHESS_SQUARES_ALONG_X
CALIB_SCALING_FACTOR = 0.667125
CORRECTION_FACTOR = 6.195 / 5.053
UNIT_TO_MM = (CHESS_SQUARE_LENGTH_M / CALIB_SCALING_FACTOR) * 1000.0 * CORRECTION_FACTOR

# print(f"UNIT_TO_MM Before: {UNIT_TO_MM}")

# BLENDER_VESSEL_LENGTH = 8.62882 # m
# VESSEL_LENGTH_MM = 126
# UNIT_TO_MM = (VESSEL_LENGTH_MM / BLENDER_VESSEL_LENGTH)
# print(f"UNIT_TO_MM After: {UNIT_TO_MM}")

# Visualization constants
GT_CLUSTER_POINT_SIZE = 5
OURS_CLUSTER_POINT_SIZE = 10
# DEFAULT_MAX_DISPLACEMENT_MM = vu.MAX_DISPLACEMENT_MM
DEFAULT_MAX_DISPLACEMENT_MM = 15
AUTO_SCALE_MAX_DISPLACEMENT = False
AUTO_SCALE_METRIC = "percentile"
AUTO_SCALE_PERCENTILE_VALUE = 100

# Camera position/orientation (from show_camera_position.py)
# Format: [[focal_x, focal_y, focal_z], [cam_x, cam_y, cam_z], [up_x, up_y, up_z]]
# Set to None to use default orientation (no rotation applied to vessels)
# CAMERA_POSITION = [
#     [-4.654521, 3.656269, 14.835112],
#     [0.171334, 0.056576, 0.118714],
#     [-0.849932, -0.503400, -0.155579]
# ]

CAMERA_POSITION = [
    [-5.358697, 3.419817, 18.256446],
    [-0.021277, -0.067798, 0.104289],
    [-0.857634, -0.489335, -0.158160]
]





# ============================================================================
# UTILITY FUNCTIONS
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
    """
    Compute rotation matrix from camera position/orientation.
    
    The rotation matrix transforms vessel positions so that when viewed
    perpendicular to the grid plane (looking down -Z), vessels appear
    as if viewed from the specified camera angle.
    
    Args:
        camera_position: Camera position in format [[cam], [focal], [up]]
        
    Returns:
        3x3 rotation matrix to apply to vessel positions (determinant = +1)
    """
    if camera_position is None:
        return np.eye(3)  # Identity matrix (no rotation)
    
    cam_pos = np.array(camera_position[0])
    focal = np.array(camera_position[1])
    up = np.array(camera_position[2])
    
    # Compute view direction (from camera to focal point)
    view_dir = focal - cam_pos
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-9)
    
    # Normalize up vector
    up = up / (np.linalg.norm(up) + 1e-9)
    
    # Compute right vector (cross product of view_dir and up)
    right = np.cross(view_dir, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    
    # Recompute up to ensure orthogonality (right-handed coordinate system)
    up = np.cross(right, view_dir)
    up = up / (np.linalg.norm(up) + 1e-9)
    
    # Camera coordinate frame: [right, up, -view_dir] (columns)
    # This matrix transforms from camera space to world space
    camera_to_world = np.column_stack([right, up, -view_dir])
    
    # To view the vessel as seen from the camera when looking at it with a default camera
    # (which looks down -Z), we just need to transform the points from world space
    # into the camera's local coordinate space.
    # The transformation from world to camera space is the inverse of camera_to_world.
    # Since camera_to_world is an orthonormal matrix, its inverse is its transpose.
    rotation_matrix = camera_to_world.T
    
    return rotation_matrix


def load_cluster_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load data from numpy file (.npy) or raw point cloud (.npz).
    """
    experiment_name = file_path.stem

    if file_path.suffix == '.npz':
        # Load raw data
        data = np.load(file_path)
        coords = data['coords']
        frame_numbers = data['frame_numbers']
        times = data['times']
        frames_data = vu.get_all_frames_data(coords, frame_numbers, times)
        
        frame_indices = sorted(frames_data.keys())
        T = len(frame_indices)
        if T == 0:
            return np.array([]), np.array([]), experiment_name
            
        N = len(frames_data[frame_indices[0]]['coords'])
        cluster_positions = np.zeros((T, N, 3))
        for i, f_idx in enumerate(frame_indices):
            cluster_positions[i] = frames_data[f_idx]['coords']
            
        times_arr = np.array([frames_data[f]['time'] for f in frame_indices])
        return cluster_positions, times_arr, experiment_name

    data = np.load(str(file_path), allow_pickle=True).item()
    
    cluster_positions = data.get("cluster_positions")  # (T, K, 3)
    times = data.get("times")  # (T,)
    experiment_name = data.get("experiment_name", file_path.stem)
    
    if cluster_positions is None:
        raise ValueError(f"File {file_path} missing 'cluster_positions' key")
    
    if times is None:
        # Generate default times if missing
        num_frames = cluster_positions.shape[0]
        times = np.arange(num_frames, dtype=np.float64)
    
    return cluster_positions, times, experiment_name


# Removed local generate_displacement_colormap and get_color_from_displacement, now using vu.get_colors


def compute_displacements_for_vessel(cluster_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute displacements for all frames of a vessel.
    
    Args:
        cluster_positions: (T, K, 3) array of cluster positions
        
    Returns:
        Tuple of (initial_positions, frame_displacements)
        - initial_positions: (K, 3) initial cluster positions
        - frame_displacements: Dict mapping frame_idx to (K,) displacement magnitudes in mm
    """
    num_frames, num_clusters, _ = cluster_positions.shape
    initial_positions = cluster_positions[0]  # First frame as reference
    
    frame_displacements = {}
    for frame_idx in range(num_frames):
        current_positions = cluster_positions[frame_idx]
        
        # Compute displacement vectors
        d_vec = current_positions - initial_positions  # in model units
        displacements_units = np.linalg.norm(d_vec, axis=1)
        displacements_mm = displacements_units * UNIT_TO_MM
        
        frame_displacements[frame_idx] = displacements_mm
    
    return initial_positions, frame_displacements


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bounding box for a set of points.
    
    Args:
        points: (N, 3) array of points
        
    Returns:
        Tuple of (min_corner, max_corner)
    """
    if len(points) == 0:
        return np.array([0, 0, 0]), np.array([0, 0, 0])
    
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    return min_corner, max_corner


def compute_grid_positions(
    all_bounding_boxes: List[Tuple[np.ndarray, np.ndarray]],
    spacing_multiplier_x: float,
    spacing_multiplier_y: float
) -> List[np.ndarray]:
    """
    Compute grid positions for all vessels.
    
    Args:
        all_bounding_boxes: List of (min_corner, max_corner) tuples for each vessel
        spacing_multiplier_x: Multiplier for spacing in X direction (columns)
        spacing_multiplier_y: Multiplier for spacing in Y direction (rows)
        
    Returns:
        List of translation vectors (one per vessel)
    """
    num_vessels = len(all_bounding_boxes)
    num_rows = NUM_ROWS
    num_cols = NUM_COLS
    
    # Compute average bounding box size
    all_sizes = []
    for min_corner, max_corner in all_bounding_boxes:
        size = max_corner - min_corner
        all_sizes.append(size)
    
    avg_size = np.mean(all_sizes, axis=0)
    spacing_x = avg_size[0] * spacing_multiplier_x
    spacing_y = avg_size[1] * spacing_multiplier_y
    
    # Compute grid positions
    grid_positions = []
    for row in range(num_rows):
        for col in range(num_cols):
            vessel_idx = row * num_cols + col
            if vessel_idx < num_vessels:
                # Position: start at origin, then offset by spacing
                # Row 0 (GT): y = 0, Row 1 (Ours): y = -spacing_y
                # Columns: x increases by spacing_x
                x_offset = col * spacing_x
                y_offset = -row * spacing_y  # Negative because we want row 1 below row 0
                z_offset = 0
                
                translation = np.array([x_offset, y_offset, z_offset])
                grid_positions.append(translation)
    
    return grid_positions


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def create_grid_visualization():
    """Create interactive grid visualization of all vessels."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Load all files
    print("Loading vessel data...")
    all_vessel_data = []
    all_bounding_boxes = []
    
    # Compute rotation matrix from camera position (if provided)
    rotation_matrix = compute_rotation_matrix_from_camera(CAMERA_POSITION)
    if CAMERA_POSITION is not None:
        print(f"\nApplying camera rotation to vessels...")
        print(f"  Camera pos: {CAMERA_POSITION[0]}")
        print(f"  Camera focal: {CAMERA_POSITION[1]}")
        print(f"  Camera up: {CAMERA_POSITION[2]}")

    # Load GT files (Row 1)
    for gt_file in GT_FILES:
        gt_path = _resolve_path(gt_file, project_root)
        if not gt_path.exists():
            print(f"  ⚠️  WARNING: GT file not found: {gt_path}")
            continue
        
        cluster_positions, times, experiment_name = load_cluster_data(gt_path)
        
        # Apply rotation immediately upon loading
        if CAMERA_POSITION is not None:
            cluster_positions = cluster_positions @ rotation_matrix.T
        
        initial_positions, frame_displacements = compute_displacements_for_vessel(cluster_positions)
        
        # Compute bounding box from all frames
        all_points = cluster_positions.reshape(-1, 3)
        min_corner, max_corner = compute_bounding_box(all_points)
        all_bounding_boxes.append((min_corner, max_corner))
        
        all_vessel_data.append({
            "cluster_positions": cluster_positions,
            "initial_positions": initial_positions,
            "frame_displacements": frame_displacements,
            "times": times,
            "experiment_name": experiment_name,
            "is_gt": True,
        })
        print(f"  ✓ GT: {experiment_name} - {cluster_positions.shape[0]} frames, {cluster_positions.shape[1]} clusters")
    
    # Load Ours files (Row 2)
    for ours_file in OURS_FILES:
        ours_path = _resolve_path(ours_file, project_root)
        if not ours_path.exists():
            print(f"  ⚠️  WARNING: Ours file not found: {ours_path}")
            continue
        
        cluster_positions, times, experiment_name = load_cluster_data(ours_path)
        
        # Apply rotation immediately upon loading
        if CAMERA_POSITION is not None:
            cluster_positions = cluster_positions @ rotation_matrix.T
        
        initial_positions, frame_displacements = compute_displacements_for_vessel(cluster_positions)
        
        # Compute bounding box from all frames
        all_points = cluster_positions.reshape(-1, 3)
        min_corner, max_corner = compute_bounding_box(all_points)
        all_bounding_boxes.append((min_corner, max_corner))
        
        all_vessel_data.append({
            "cluster_positions": cluster_positions,
            "initial_positions": initial_positions,
            "frame_displacements": frame_displacements,
            "times": times,
            "experiment_name": experiment_name,
            "is_gt": False,
        })
        print(f"  ✓ Ours: {experiment_name} - {cluster_positions.shape[0]} frames, {cluster_positions.shape[1]} clusters")
    
    if len(all_vessel_data) == 0:
        print("ERROR: No vessel data loaded!")
        return
    
    print(f"\nLoaded {len(all_vessel_data)} vessels")
    
    # Verify all vessels have the same number of frames
    num_frames_list = [vessel["cluster_positions"].shape[0] for vessel in all_vessel_data]
    if len(set(num_frames_list)) > 1:
        print(f"⚠️  WARNING: Vessels have different frame counts: {num_frames_list}")
        print("   Using minimum frame count for all vessels")
        min_frames = min(num_frames_list)
        for vessel in all_vessel_data:
            vessel["cluster_positions"] = vessel["cluster_positions"][:min_frames]
            vessel["frame_displacements"] = {k: v for k, v in vessel["frame_displacements"].items() if k < min_frames}
    
    num_frames = min(num_frames_list)
    frame_indices = list(range(num_frames))
    
    # Compute grid positions
    grid_positions = compute_grid_positions(all_bounding_boxes, GRID_SPACING_MULTIPLIER_X, GRID_SPACING_MULTIPLIER_Y)
    
    # Compute global max displacement for colormap scaling
    print("\nComputing global displacement statistics...")
    all_displacements_flat = []
    for vessel in all_vessel_data:
        for frame_idx, displacements in vessel["frame_displacements"].items():
            if len(displacements) > 0:
                all_displacements_flat.append(displacements)
    
    if len(all_displacements_flat) > 0:
        flat_disps = np.concatenate(all_displacements_flat)
        if AUTO_SCALE_MAX_DISPLACEMENT:
            if AUTO_SCALE_METRIC == "max":
                max_disp_val = float(np.max(flat_disps))
                print(f"  Global max displacement: {max_disp_val:.4f} mm")
            elif AUTO_SCALE_METRIC == "percentile":
                max_disp_val = float(np.percentile(flat_disps, AUTO_SCALE_PERCENTILE_VALUE))
                print(f"  Global {AUTO_SCALE_PERCENTILE_VALUE}th percentile displacement: {max_disp_val:.4f} mm")
            else:
                max_disp_val = float(DEFAULT_MAX_DISPLACEMENT_MM)
                print(f"  Using default max displacement: {max_disp_val:.4f} mm")
        else:
            max_disp_val = float(DEFAULT_MAX_DISPLACEMENT_MM)
            print(f"  Using fixed max displacement: {max_disp_val:.4f} mm")
    else:
        max_disp_val = float(DEFAULT_MAX_DISPLACEMENT_MM)
        print(f"  No displacements found, using default: {max_disp_val:.4f} mm")
    
    # Generate colormap
    cmap = vu.get_colormap()
    
    # Pre-compute colors for all frames and all vessels
    print("\nPre-computing displacement colors...")
    all_vessel_frame_data = []
    frame_statistics = {}  # Pre-compute statistics for text generation
    
    for vessel_idx, vessel in enumerate(all_vessel_data):
        vessel_frame_data = {}
        for frame_idx in frame_indices:
            displacements = vessel["frame_displacements"].get(frame_idx, np.array([]))
            
            # Get cluster positions for this frame
            cluster_positions_frame = vessel["cluster_positions"][frame_idx]

            if len(displacements) > 0:
                colors = vu.get_colors(displacements, cmap, vmax=max_disp_val)
            else:
                colors = None
            
            # Apply grid translation
            translation = grid_positions[vessel_idx]
            translated_positions = cluster_positions_frame + translation
            
            # Create PolyData
            if len(translated_positions) > 0:
                polydata = pv.PolyData(translated_positions)
            else:
                polydata = None
            
            # Pre-compute statistics for this frame
            stats = {
                "max": 0.0, 
                "mean": 0.0, 
                "median": 0.0,
                "p75": 0.0,
                "p90": 0.0
            }
            if len(displacements) > 0:
                stats["max"] = np.max(displacements)
                stats["mean"] = np.mean(displacements)
                stats["median"] = np.median(displacements)
                stats["p75"] = np.percentile(displacements, 75)
                stats["p90"] = np.percentile(displacements, 90)
            
            vessel_frame_data[frame_idx] = {
                "polydata": polydata,
                "colors": colors,
                "displacements": displacements,
                "stats": stats
            }
            
            # Pre-compute statistics for global tracking (optional)
            if frame_idx not in frame_statistics:
                frame_statistics[frame_idx] = {"max_disps": [], "mean_disps": []}
            
            if len(displacements) > 0:
                frame_statistics[frame_idx]["max_disps"].append(stats["max"])
                frame_statistics[frame_idx]["mean_disps"].append(stats["mean"])
        
        all_vessel_frame_data.append(vessel_frame_data)
    
    print(f"  Pre-computed data for {len(all_vessel_frame_data)} vessels, {num_frames} frames")
    
    # Create PyVista plotter
    plotter = pv.Plotter(title="Multiple Vessels Displacement Grid - Frame Navigation")
    plotter.set_background("white")
    # Enable orthographic projection
    plotter.camera.enable_parallel_projection()
    
    # Store actors for all vessels
    vessel_actors = [None] * len(all_vessel_data)
    
    def update_callback(frame_list_idx: int, frame_idx: int, plotter: pv.Plotter) -> None:
        """Update callback for frame changes - updates all vessels."""
        nonlocal vessel_actors
        
        # Store old actors for removal after adding new ones (double-buffering for smooth updates)
        old_actors = vessel_actors.copy()
        
        # Add new actors for all vessels FIRST (before removing old ones)
        vessel_actors = []
        for vessel_idx, vessel_frame_data in enumerate(all_vessel_frame_data):
            if frame_idx in vessel_frame_data:
                frame_data = vessel_frame_data[frame_idx]
                polydata = frame_data["polydata"]
                colors = frame_data["colors"]
                
                # Determine point size based on vessel type
                is_gt = all_vessel_data[vessel_idx].get("is_gt", True)
                point_size = GT_CLUSTER_POINT_SIZE if is_gt else OURS_CLUSTER_POINT_SIZE
                
                if polydata is not None:
                    if colors is not None and len(colors) > 0:
                        actor = plotter.add_mesh(
                            polydata,
                            scalars=colors,
                            rgb=True,
                            render_points_as_spheres=True,
                            point_size=point_size,
                            lighting=False,
                            name=f"vessel_{vessel_idx}_frame_{frame_idx}",
                        )
                    else:
                        actor = plotter.add_mesh(
                            polydata,
                            render_points_as_spheres=True,
                            point_size=point_size,
                            color="blue",
                            lighting=False,
                            name=f"vessel_{vessel_idx}_frame_{frame_idx}",
                        )
                    vessel_actors.append(actor)
                else:
                    vessel_actors.append(None)
            else:
                vessel_actors.append(None)
        
        # Now remove old actors (after new ones are already added)
        for actor in old_actors:
            if actor is not None:
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass
    
    def text_generator(frame_idx: int, frame_data: Dict[str, Any]) -> str:
        """Generate text for frame display with per-vessel stats."""
        max_frame = frame_indices[-1]
        frame_width = len(str(max_frame))
        frame_str = f"{frame_idx:0{frame_width}d}"
        max_frame_str = f"{max_frame:0{frame_width}d}"
        
        lines = [
            f"Frame: {frame_str}/{max_frame_str}  Scale: {max_disp_val:.2f} mm",
            "-" * 60
        ]
        
        # Row 1: GT
        lines.append("Row 1 (GT):")
        for i in range(5):
            if i < len(all_vessel_frame_data):
                v_data = all_vessel_frame_data[i].get(frame_idx, {})
                stats = v_data.get("stats", {"mean": 0, "median": 0, "p75": 0, "p90": 0, "max": 0})
                mean = stats["mean"]
                med = stats["median"]
                p75 = stats["p75"]
                p90 = stats["p90"]
                mx = stats["max"]
                lines.append(f"  V{i}: Av={mean:.2f} Md={med:.2f} 75%={p75:.2f} 90%={p90:.2f} Mx={mx:.2f}")
        
        lines.append("-" * 30)
        
        # Row 2: Ours
        lines.append("Row 2 (Ours):")
        for i in range(5):
            v_idx = i + 5
            if v_idx < len(all_vessel_frame_data):
                v_data = all_vessel_frame_data[v_idx].get(frame_idx, {})
                stats = v_data.get("stats", {"mean": 0, "median": 0, "p75": 0, "p90": 0, "max": 0})
                mean = stats["mean"]
                med = stats["median"]
                p75 = stats["p75"]
                p90 = stats["p90"]
                mx = stats["max"]
                lines.append(f"  V{v_idx}: Av={mean:.2f} Md={med:.2f} 75%={p75:.2f} 90%={p90:.2f} Mx={mx:.2f}")
        
        return "\n".join(lines)
    
    # Create frames_data dict for interactive_utils (minimal, just for compatibility)
    frames_data = {idx: {"time": float(idx)} for idx in frame_indices}
    
    # Prepare instructions
    instructions = create_default_instructions(
        visualization_description="Grid of vessels showing displacements",
        additional_info=[
            "Row 1: GT files, Row 2: Ours files",
            "Purple/Dark = Low displacement, Yellow/Light = High displacement",
            f"Global colormap scale: {max_disp_val:.2f} mm",
            f"Grid spacing multipliers: X={GRID_SPACING_MULTIPLIER_X}, Y={GRID_SPACING_MULTIPLIER_Y}",
        ],
    )
    
    # Compute global bounds for camera
    all_translated_points = []
    for vessel_idx, vessel in enumerate(all_vessel_data):
        translation = grid_positions[vessel_idx]
        for frame_idx in frame_indices:
            positions = vessel["cluster_positions"][frame_idx]
            translated = positions + translation
            all_translated_points.append(translated)
    
    if len(all_translated_points) > 0:
        all_points = np.vstack(all_translated_points)
        bounds = [
            all_points[:, 0].min(),
            all_points[:, 0].max(),
            all_points[:, 1].min(),
            all_points[:, 1].max(),
            all_points[:, 2].min(),
            all_points[:, 2].max(),
        ]
    else:
        bounds = None
    
    # Use the standardized interactive visualization
    create_interactive_visualization(
        plotter=plotter,
        frame_indices=frame_indices,
        frames_data=frames_data,
        update_callback=update_callback,
        text_generator=text_generator,
        instructions=instructions,
        bounds=bounds,
        title="Multiple Vessels Displacement Grid - Frame Navigation",
        font='courier',
        text_font_size=8,
    )
    
    print("\nDisplaying grid of vessels with displacement coloring.\n")
    plotter.show()


if __name__ == "__main__":
    try:
        create_grid_visualization()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
