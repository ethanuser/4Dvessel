#!/usr/bin/env python3
"""
Interactive visualization of multiple vessels in a grid layout showing stress.

This script:
1. Loads multiple GT and ours numpy files
2. Computes stress for each vessel's edges (relative to initial lengths)
3. Arranges vessels in a 2x5 grid (Row 1: GT files, Row 2: Ours files)
4. Displays all vessels with stress-colored edges using a global colormap scale
5. Single frame slider controls all vessels simultaneously

Usage:
    python testing_continuum.py
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.interactive_utils import create_interactive_visualization, create_default_instructions
import vessel_utils as vu

# ============================================================================
# CONFIGURATION
# ============================================================================

# Grid spacing multiplier - controls spacing between vessels
# Higher values = more spacing between vessels
GRID_SPACING_MULTIPLIER = 1  # Adjust this to change spacing

# Stress computation method
# Options: "edge_linear" (original), "tri_linear", "tri_neo_hookean"
# 
# METHOD DETAILS:
# - "edge_linear": 1D spring model. Stress = E * (l - L0) / L0.
#   Simple but mesh-dependent and not a true surface stress.
#
# - "tri_neo_hookean": Surface membrane proxy using Incompressible Neo-Hookean model.
#   Computes deformation gradient F from 2D reference projection to 3D current configuration.
#   Scalar Proxy = mu * (lambda_max^2 - 1/lambda_max).
#   Represents "tension" magnitude. Nu approx 0.5.
#   LIMITATIONS: Assumes reference state is stress-free. 
#   Projection to 2D reference plane (PCA) may distort complex 3D texturing/folding if not locally flat.
#
# - "tri_linear": Max principal strain from Green-Lagrange tensor * Young's Modulus.
#
# STRESS_METHOD = "tri_neo_hookean"
STRESS_METHOD = "tri_linear"

# List of GT numpy file paths (from blender) - Row 1
GT_FILES = [
    r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_25.npy",
    r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_50.npy",
    r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_75.npy",
    r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_1_00.npy",
    r"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_1_25.npy",
]

# List of ours numpy file paths - Row 2
OURS_FILES = [
    r"cluster_data/cluster_mesh_timeseries_lattice_strength_0_25.npy",
    r"cluster_data/cluster_mesh_timeseries_lattice_strength_0_50.npy",
    r"cluster_data/cluster_mesh_timeseries_lattice_strength_0_75.npy",
    r"cluster_data/cluster_mesh_timeseries_lattice_strength_1_00.npy",
    r"cluster_data/cluster_mesh_timeseries_lattice_strength_1_25.npy",
]

# Physical constants
YOUNG_MODULUS_SILICON = vu.YOUNG_MODULUS_SILICON

# Visualization constants
EDGE_LINE_WIDTH = 5
EDGE_OPACITY = 0.7
MAX_STRESS_PA = vu.MAX_STRESS_PA  # absolute max stress for colormap


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


def load_cluster_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load cluster positions, edges, initial lengths, times, and experiment name from numpy file.
    
    Args:
        file_path: Path to the numpy file
        
    Returns:
        Tuple of (cluster_positions, edges, initial_lengths, times, experiment_name)
        - cluster_positions: (T, K, 3) array
        - edges: (E, 2) array of edge indices
        - initial_lengths: (E,) array of initial edge lengths
        - times: (T,) array
        - experiment_name: String
    """
    data = np.load(str(file_path), allow_pickle=True).item()
    
    cluster_positions = data.get("cluster_positions")  # (T, K, 3)
    edges = data.get("edges")  # (E, 2)
    initial_lengths = data.get("initial_lengths")  # (E,)
    times = data.get("times")  # (T,)
    experiment_name = data.get("experiment_name", file_path.stem)
    
    if cluster_positions is None:
        raise ValueError(f"File {file_path} missing 'cluster_positions' key")
    
    if edges is None:
        edges = np.array([]).reshape(0, 2)
    
    if initial_lengths is None:
        if len(edges) > 0 and cluster_positions.shape[0] > 0:
            # Compute initial lengths from first frame
            initial_positions = cluster_positions[0]
            initial_lengths = np.linalg.norm(
                initial_positions[edges[:, 0]] - initial_positions[edges[:, 1]], axis=1
            )
        else:
            initial_lengths = np.array([])
    
    if times is None:
        # Generate default times if missing
        num_frames = cluster_positions.shape[0]
        times = np.arange(num_frames, dtype=np.float64)
    
    return cluster_positions, edges, initial_lengths, times, experiment_name


# Removed local generate_stress_colormap and get_color_from_stress, now using vu.get_colors


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
    spacing_multiplier: float
) -> List[np.ndarray]:
    """
    Compute grid positions for all vessels.
    
    Args:
        all_bounding_boxes: List of (min_corner, max_corner) tuples for each vessel
        spacing_multiplier: Multiplier for spacing between vessels
        
    Returns:
        List of translation vectors (one per vessel)
    """
    num_vessels = len(all_bounding_boxes)
    num_rows = 2
    num_cols = 5
    
    # Compute average bounding box size
    all_sizes = []
    for min_corner, max_corner in all_bounding_boxes:
        size = max_corner - min_corner
        all_sizes.append(size)
    
    avg_size = np.mean(all_sizes, axis=0)
    spacing = avg_size * spacing_multiplier
    
    # Compute grid positions
    grid_positions = []
    for row in range(num_rows):
        for col in range(num_cols):
            vessel_idx = row * num_cols + col
            if vessel_idx < num_vessels:
                # Position: start at origin, then offset by spacing
                # Row 0 (GT): y = 0, Row 1 (Ours): y = -spacing_y
                # Columns: x increases by spacing_x
                x_offset = col * spacing[0]
                y_offset = -row * spacing[1]  # Negative because we want row 1 below row 0
                z_offset = 0
                
                translation = np.array([x_offset, y_offset, z_offset])
                grid_positions.append(translation)
    
    return grid_positions


# ============================================================================
# CONTINUUM MECHANICS UTILS
# ============================================================================

def build_reference_triangulation(points0: np.ndarray, max_edge_length: Optional[float] = None) -> np.ndarray:
    """
    Build a fixed triangulation using 3D Alpha Shapes (Delaunay 3D + Surface Extraction).
    This handles non-planar 3D geometries (tubes, folds) better than 2D projection.
    
    Args:
        points0: (K, 3) vertices at rest/reference configuration.
        max_edge_length: Alpha parameter for Delaunay 3D (controls "tightness" of mesh).
                         If None, defaults to infinite (Convex Hull).
    
    Returns:
        tri: (F, 3) integer indices for triangles.
    """
    if len(points0) < 4:
        return np.array([])

    try:
        # Create PyVista cloud
        cloud = pv.PolyData(points0)
        
        # Delaunay 3D
        # alpha=0 implies convex hull in VTK if not specified, but here alpha > 0 acts as filter
        # If max_edge_length is None, we run without alpha (Convex Hull)
        alpha = max_edge_length if max_edge_length is not None else 0.0
        
        grid = cloud.delaunay_3d(alpha=alpha)
        
        # Extract surface geometry
        # pass_pointid=True keeps track of original indices (verified from error msg)
        surf = grid.extract_surface(pass_pointid=True)
        
        # Clean the mesh to remove degenerate cells/points
        surf = surf.clean()
        
        # If extraction failed or yielded no cells, return empty
        if surf.n_cells == 0:
            return np.array([])
        
        # Get triangles (faces)
        # surf.faces is a flat array: [n_pts, p0, p1, ..., n_pts, p0, p1, ...]
        # We assume triangular mesh from extract_surface
        if not surf.is_all_triangles:
            surf = surf.triangulate()
            
        # Reshape faces to (F, 4) -> drop the first column (always 3)
        faces = surf.faces.reshape(-1, 4)[:, 1:]
        
        # Map back to original indices
        # "vtkOriginalPointIds" contains the index in 'points0' for each point in 'surf'
        if "vtkOriginalPointIds" in surf.point_data:
            orig_ids = surf.point_data["vtkOriginalPointIds"]
            mapped_tri = orig_ids[faces]
            return mapped_tri
        else:
            # If point count matches exactly and no reordering happened (unlikely but possible)
            if surf.n_points == len(points0):
                return faces
            return np.array([])
            
    except Exception as e:
        print(f"  Error in 3D triangulation: {e}")
        return np.array([])


def precompute_triangle_bases(points0: np.ndarray, tri: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Precompute local reference bases and shape matrices for each triangle.
    
    For each triangle (nodes n0, n1, n2), we define a local 2D basis (t1, t2).
    Usually we align t1 with edge (n1 - n0).
    
    Args:
        points0: (K, 3) reference vertices
        tri: (F, 3) triangle indices
        
    Returns:
        Dict containing:
            'bases': (F, 3, 3) array where each row is [t1, t2, n]
            'inv_Ds0': (F, 2, 2) array of inverse reference shape matrices (2x2)
    """
    # Vertices: (F, 3, 3) -> [tri_idx, node_idx, xyz]
    v0 = points0[tri[:, 0]]
    v1 = points0[tri[:, 1]]
    v2 = points0[tri[:, 2]]
    
    # Edge vectors in 3D
    e1 = v1 - v0
    e2 = v2 - v0
    
    # Compute normal
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    # Avoid division by zero for degenerate triangles
    n_norm[n_norm < 1e-12] = 1.0 
    n = n / n_norm
    
    # Construct local basis (t1, t2, n)
    # t1 aligned with e1
    t1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    # t2 is n cross t1 (orthonormal)
    t2 = np.cross(n, t1)
    
    # Store basis as (F, 3, 3) -> each (3,3) is columns [t1, t2, n]
    # But for projection we usually want rows for dot product, so let's store rows.
    # Basis matrix B = [t1^T; t2^T; n^T]
    bases = np.stack([t1, t2, n], axis=1) # (F, 3, 3)
    
    # Project reference edge vectors e1, e2 into this basis to get Ds0 (2x2)
    # Ds0 = [ (x1-x0)_2d, (x2-x0)_2d ]
    # x_2d = [dot(x, t1), dot(x, t2)]
    
    e1_2d = np.stack([np.sum(e1 * t1, axis=1), np.sum(e1 * t2, axis=1)], axis=1) # (F, 2)
    e2_2d = np.stack([np.sum(e2 * t1, axis=1), np.sum(e2 * t2, axis=1)], axis=1) # (F, 2)
    
    # Stack columns to make 2x2 matrix
    # Ds0[:, 0] = e1_2d, Ds0[:, 1] = e2_2d
    Ds0 = np.stack([e1_2d, e2_2d], axis=2) # (F, 2, 2)
    
    # Invert Ds0
    # Check determinant or condition number if needed, but for now just invert
    # Add epsilon for stability
    inv_Ds0 = np.linalg.inv(Ds0) # Vectorized inversion
    
    return {
        'bases': bases,
        'inv_Ds0': inv_Ds0
    }


def compute_triangle_strain_and_stress(
    points_t: np.ndarray, 
    tri: np.ndarray, 
    precomp: Dict[str, np.ndarray], 
    method: str = "tri_neo_hookean"
) -> np.ndarray:
    """
    Compute scalar stress/strain proxy for each triangle using Finite Element principles.
    
    Args:
        points_t: (K, 3) current vertices
        tri: (F, 3) triangle indices
        precomp: Dict from precompute_triangle_bases
        method: "tri_linear" or "tri_neo_hookean"
        
    Returns:
        scalars: (F,) array of scalar values (stress proxy)
    """
    bases = precomp['bases'] # (F, 3, 3) -> [t1, t2, n] rows
    inv_Ds0 = precomp['inv_Ds0'] # (F, 2, 2)
    
    v0 = points_t[tri[:, 0]]
    v1 = points_t[tri[:, 1]]
    v2 = points_t[tri[:, 2]]
    
    e1 = v1 - v0
    e2 = v2 - v0
    
    # Project current edges into REFERENCE basis
    # We maintain the material frame. 
    # F = Ds * Ds0^-1.
    # We construct F such that it maps reference config to current config.
    
    # Stack edges as columns: (F, 3, 2). 3 rows (xyz), 2 cols (edge1, edge2)
    edges_mat = np.stack([e1, e2], axis=2) 
    
    # Matrix multiply: (F, 3, 2) @ (F, 2, 2) -> (F, 3, 2)
    # Deformation gradient F mapping 2D parameter space to 3D current space
    F_mat = np.matmul(edges_mat, inv_Ds0)
    
    # Green-Lagrange Strain Tensor E = 0.5 * (F.T @ F - I) -> (F, 2, 2)
    FTF = np.matmul(np.transpose(F_mat, axes=(0, 2, 1)), F_mat) # (F, 2, 2)
    I = np.eye(2)[None, :, :]
    E_green = 0.5 * (FTF - I)
    
    if method == "tri_linear":
        # Linear strain proxy: Trace or Max Principal
        # Max principal strain
        eigvals = np.linalg.eigvalsh(E_green) # (F, 2)
        max_strain = np.max(eigvals, axis=1)
        # Simple Linear Stress Proxy, clip negative values if we care mostly about tension?
        # But for general stress, keep sign.
        return YOUNG_MODULUS_SILICON * max_strain
        
    elif method == "tri_neo_hookean":
        # Incompressible Neo-Hookean membrane proxy
        # mu = E_mod / 3 (assuming nu=0.5)
        mu = YOUNG_MODULUS_SILICON / 3.0
        
        # Right Cauchy-Green C = F.T F
        C = FTF
        # 2D Principal stretches squared
        eigvals_C = np.linalg.eigvalsh(C) # lambda^2
        eigvals_C = np.clip(eigvals_C, 1e-6, None) # Avoid sqrt(neg)
        lambdas = np.sqrt(eigvals_C)
        
        # Strain Energy W = (mu/2) * (I1 - 3) ? No, for 2D/Membrane:
        # For thin sheet: sigma_1 = mu * (lambda1^2 - 1/lambda3^2) ... 
        # let's stick to the user's proxy:
        # sigma_proxy = mu * (max(lam)^2 - 1/max(lam))
        
        lam_max = np.max(lambdas, axis=1)
        
        # Avoid division by zero
        lam_max[lam_max < 1e-6] = 1e-6
        
        sigma_proxy = mu * (lam_max**2 - 1.0/lam_max)
        return sigma_proxy
        
    return np.zeros(len(tri))


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
    
    # Load GT files (Row 1)
    for gt_file in GT_FILES:
        gt_path = _resolve_path(gt_file, project_root)
        if not gt_path.exists():
            print(f"  ⚠️  WARNING: GT file not found: {gt_path}")
            continue
        
        cluster_positions, edges, initial_lengths, times, experiment_name = load_cluster_data(gt_path)
        
        # Compute bounding box from all frames
        all_points = cluster_positions.reshape(-1, 3)
        min_corner, max_corner = compute_bounding_box(all_points)
        all_bounding_boxes.append((min_corner, max_corner))
        
        all_vessel_data.append({
            "cluster_positions": cluster_positions,
            "edges": edges,
            "initial_lengths": initial_lengths,
            "times": times,
            "experiment_name": experiment_name,
            "is_gt": True,
        })
        print(f"  ✓ GT: {experiment_name} - {cluster_positions.shape[0]} frames, {cluster_positions.shape[1]} clusters, {len(edges)} edges")
    
    # Load Ours files (Row 2)
    for ours_file in OURS_FILES:
        ours_path = _resolve_path(ours_file, project_root)
        if not ours_path.exists():
            print(f"  ⚠️  WARNING: Ours file not found: {ours_path}")
            continue
        
        cluster_positions, edges, initial_lengths, times, experiment_name = load_cluster_data(ours_path)
        
        # Compute bounding box from all frames
        all_points = cluster_positions.reshape(-1, 3)
        min_corner, max_corner = compute_bounding_box(all_points)
        all_bounding_boxes.append((min_corner, max_corner))
        
        all_vessel_data.append({
            "cluster_positions": cluster_positions,
            "edges": edges,
            "initial_lengths": initial_lengths,
            "times": times,
            "experiment_name": experiment_name,
            "is_gt": False,
        })
        print(f"  ✓ Ours: {experiment_name} - {cluster_positions.shape[0]} frames, {cluster_positions.shape[1]} clusters, {len(edges)} edges")
    
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
    
    num_frames = min(num_frames_list)
    frame_indices = list(range(num_frames))
    
    # Compute grid positions
    grid_positions = compute_grid_positions(all_bounding_boxes, GRID_SPACING_MULTIPLIER)
    
    # Generate colormap
    cmap = vu.get_colormap()
    
    # Pre-compute stress and colors for all frames and all vessels
    print("\nPre-computing stress and colors...")
    all_vessel_frame_data = []
    frame_statistics = {}  # Pre-compute statistics for text generation
    
    for vessel_idx, vessel in enumerate(all_vessel_data):
        vessel_frame_data = {}
        
        # Precompute reference data for continuum method if needed
        continuum_precomp = None
        tri = None
        
        if STRESS_METHOD.startswith("tri_"):
            try:
                # Use first frame as reference
                points0 = vessel["cluster_positions"][0]
                
                # Heuristic for max edge length: 1.5x the median initial edge length
                # Lower value (1.5 instead of 3.0) preserves holes/topology better
                initial_lengths = vessel["initial_lengths"]
                max_edge_len = None
                if len(initial_lengths) > 0:
                    median_len = np.median(initial_lengths)
                    max_edge_len = median_len * 1.4
                
                tri = build_reference_triangulation(points0, max_edge_length=max_edge_len)
                
                if len(tri) > 0:
                    continuum_precomp = precompute_triangle_bases(points0, tri)
                    filter_msg = f"(filtered > {max_edge_len:.4f})" if max_edge_len else ""
                    print(f"  Vessel {vessel_idx}: Built triangulation with {len(tri)} triangles {filter_msg}")
                else:
                    print(f"  Vessel {vessel_idx}: Failed to build triangulation (too few points or all filtered?)")
            except Exception as e:
                print(f"  Vessel {vessel_idx}: Triangulation/Precomputation failed: {e}")
                tri = None 
                continuum_precomp = None

        for frame_idx in frame_indices:
            cluster_positions_frame = vessel["cluster_positions"][frame_idx]
            edges = vessel["edges"]
            initial_lengths = vessel["initial_lengths"]
            
            # Apply grid translation to cluster positions (do this early for PolyData)
            translation = grid_positions[vessel_idx]
            translated_positions = cluster_positions_frame + translation
            
            stresses = np.array([])
            colors = None
            polydata = None
            
            # ---------------------------------------------------------
            # Tri/Continuum Method
            # ---------------------------------------------------------
            if continuum_precomp is not None and tri is not None:
                # Compute triangle stresses
                try:
                    stresses = compute_triangle_strain_and_stress(
                        cluster_positions_frame, tri, continuum_precomp, method=STRESS_METHOD
                    )
                    
                    if len(stresses) > 0:
                        colors = vu.get_colors(stresses, cmap, vmin=0, vmax=MAX_STRESS_PA, use_abs=True)
                    
                    # Create Surface PolyData
                    # PyVista requires faces array: [3, p0, p1, p2, 3, p0, p1, p2, ...]
                    faces = np.hstack([np.full((tri.shape[0], 1), 3), tri])
                    polydata = pv.PolyData(translated_positions, faces=faces.flatten())
                    
                except Exception as e:
                    print(f"  Error computing tri stress frame {frame_idx}: {e}")
                    polydata = None

            # ---------------------------------------------------------
            # Fallback / Edge Method
            # ---------------------------------------------------------
            if polydata is None:
                # Calculate edge stress (Fallback or default)
                stresses = vu.compute_edge_stress(
                    cluster_positions_frame,
                    edges,
                    initial_lengths,
                    YOUNG_MODULUS_SILICON
                )
                
                if len(stresses) > 0:
                    colors = vu.get_colors(stresses, cmap, vmin=0, vmax=MAX_STRESS_PA, use_abs=True)
                
                # Create Edge PolyData
                if len(translated_positions) > 0 and len(edges) > 0:
                    lines = np.hstack([np.full((edges.shape[0], 1), 2), edges])
                    polydata = pv.PolyData(translated_positions, lines=lines.flatten())

            vessel_frame_data[frame_idx] = {
                "polydata": polydata,
                "colors": colors,
                "stresses": stresses,
                "is_surface": (continuum_precomp is not None)
            }
            
            # Pre-compute statistics for this frame
            stats = {
                "max": 0.0, 
                "mean": 0.0, 
                "median": 0.0,
                "p75": 0.0,
                "p90": 0.0
            }
            if len(stresses) > 0:
                stats["max"] = np.max(stresses)
                stats["mean"] = np.mean(stresses)
                stats["median"] = np.median(stresses)
                stats["p75"] = np.percentile(stresses, 75)
                stats["p90"] = np.percentile(stresses, 90)
            
            vessel_frame_data[frame_idx] = {
                "polydata": polydata,
                "colors": colors,
                "stresses": stresses,
                "is_surface": (continuum_precomp is not None),
                "stats": stats
            }
            
            # Global stats (kept for reference or if needed)
            if frame_idx not in frame_statistics:
                frame_statistics[frame_idx] = {"max_stresses": [], "mean_stresses": []}
            
            if len(stresses) > 0:
                frame_statistics[frame_idx]["max_stresses"].append(stats["max"])
                frame_statistics[frame_idx]["mean_stresses"].append(stats["mean"])
        
        all_vessel_frame_data.append(vessel_frame_data)
    
    print(f"  Pre-computed data for {len(all_vessel_frame_data)} vessels, {num_frames} frames")
    
    # Create PyVista plotter
    plotter = pv.Plotter(title="Multiple Vessels Stress Grid - Frame Navigation")
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
                is_surface = frame_data.get("is_surface", False)
                
                if polydata is not None:
                    if colors is not None and len(colors) > 0:
                        if is_surface:
                            # Color faces by stress
                            actor = plotter.add_mesh(
                                polydata,
                                scalars=colors,
                                rgb=True,
                                show_edges=False,    # Disabled edges as requested
                                lighting=False,      # Disabled lighting (flat/unlit)
                                name=f"vessel_{vessel_idx}_frame_{frame_idx}",
                                opacity=1.0,
                            )
                        else:
                            # Color edges by stress
                            actor = plotter.add_mesh(
                                polydata,
                                scalars=colors,
                                rgb=True,
                                line_width=EDGE_LINE_WIDTH,
                                name=f"vessel_{vessel_idx}_frame_{frame_idx}",
                                opacity=EDGE_OPACITY,
                            )
                    else:
                        actor = plotter.add_mesh(
                            polydata,
                            color="blue",
                            name=f"vessel_{vessel_idx}_frame_{frame_idx}",
                            **({"show_edges": False, "lighting": False} if is_surface else {"line_width": EDGE_LINE_WIDTH})
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
            f"Frame: {frame_str}/{max_frame_str}  Scale: 0 to {MAX_STRESS_PA/1e6:.1f} MPa (absolute)",
            "-" * 60
        ]
        
        # Row 1: GT
        lines.append("Row 1 (GT):")
        for i in range(5):
            if i < len(all_vessel_frame_data):
                v_data = all_vessel_frame_data[i].get(frame_idx, {})
                stats = v_data.get("stats", {"mean": 0, "median": 0, "p75": 0, "p90": 0, "max": 0})
                # Convert to MPa
                mean = stats["mean"] / 1e6
                med = stats["median"] / 1e6
                p75 = stats["p75"] / 1e6
                p90 = stats["p90"] / 1e6
                mx = stats["max"] / 1e6
                lines.append(f"  V{i}: Av={mean:.2f} Md={med:.2f} 75%={p75:.2f} 90%={p90:.2f} Mx={mx:.2f}")
        
        lines.append("-" * 30)
        
        # Row 2: Ours
        lines.append("Row 2 (Ours):")
        for i in range(5):
            v_idx = i + 5
            if v_idx < len(all_vessel_frame_data):
                v_data = all_vessel_frame_data[v_idx].get(frame_idx, {})
                stats = v_data.get("stats", {"mean": 0, "median": 0, "p75": 0, "p90": 0, "max": 0})
                # Convert to MPa
                mean = stats["mean"] / 1e6
                med = stats["median"] / 1e6
                p75 = stats["p75"] / 1e6
                p90 = stats["p90"] / 1e6
                mx = stats["max"] / 1e6
                lines.append(f"  V{v_idx}: Av={mean:.2f} Md={med:.2f} 75%={p75:.2f} 90%={p90:.2f} Mx={mx:.2f}")
        
        return "\n".join(lines)
    
    # Create frames_data dict for interactive_utils (minimal, just for compatibility)
    frames_data = {idx: {"time": float(idx)} for idx in frame_indices}
    
    # Prepare instructions
    instructions = create_default_instructions(
        visualization_description="Grid of vessels showing stress",
        additional_info=[
            "Row 1: GT files, Row 2: Ours files",
            "Purple/Dark = Zero absolute stress, Yellow/Light = High absolute stress",
            f"Global colormap scale: 0 to {MAX_STRESS_PA / 1e6:.1f} MPa (absolute)",
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
        title="Multiple Vessels Stress Grid - Frame Navigation",
        font='courier',
        text_font_size=8,  # Smaller font as requested
    )
    
    print("\nDisplaying grid of vessels with stress-colored surfaces.\n")
    plotter.show()


if __name__ == "__main__":
    try:
        create_grid_visualization()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
