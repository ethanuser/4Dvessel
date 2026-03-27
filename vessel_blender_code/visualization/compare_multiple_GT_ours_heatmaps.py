#!/usr/bin/env python3
"""
Compare multiple ground truth numpy files from blender against multiple Ours numpy files.
Generates heatmap tables showing chamfer distance and/or ESDD comparisons between all pairs of GT and Ours files.

This script:
1. Loads multiple GT cluster numpy files
2. Loads multiple Ours cluster numpy files
3. Computes chamfer distance and/or ESDD for each pair
4. Creates heatmap tables for enabled metrics
5. Saves the heatmaps as PNG images

Usage:
    python compare_multiple_GT_ours_heatmaps.py

Configuration:
    Edit GT_FILES and OURS_FILES lists in the CONFIGURATION section.
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict
import datetime

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.distance_utils import compute_chamfer_distance
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import vessel_utils as vu

# ============================================================================
# CONFIGURATION
# ============================================================================
# Computation flags - set to True to compute and generate heatmaps for each metric
COMPUTE_CHAMFER = True   # Compute chamfer distance and generate heatmap
COMPUTE_DELTA_CD = True  # Compute Temporal Chamfer Delta (ΔCD) and generate heatmap
COMPUTE_F_SCORE = True   # Compute F-score and generate heatmap
COMPUTE_PRECISION = True # Compute Precision and generate heatmap
COMPUTE_RECALL = True    # Compute Recall and generate heatmap

# Multi-panel configuration
GENERATE_MULTIPANEL = False  # If True, combines heatmaps into a single multi-panel figure
# Order and inclusion of metrics in the multi-panel figure. 
# Metrics in this list will be included in the multi-panel figure in the specified order.
# Available options: "F-score", "Precision", "Recall", "Chamfer", "Delta CD", "ESDD"
# Note: They must also be set to True in computation flags above to be included.
MULTIPANEL_METRICS = ["Chamfer", "Delta CD", "Precision", "Recall", "F-score"]
MULTIPANEL_TITLE = "Comparison of Agreement Metrics"

# Parallel configuration
NUM_WORKERS = -1  # Set to -1 to use all available CPU cores, or specify a number (e.g., 4)

# F-score threshold (tau) for Precision/Recall/F-score computation
FSCORE_TAU = 0.07  # Distance threshold in same units as point coordinates
# Backwards-compatible alias (in case F_SCORE_TAU is referenced elsewhere)
F_SCORE_TAU = FSCORE_TAU

# GT vs RAW GT logic
USE_RAW_GT = True

GT_BASE = Path.home() / "Documents/Blender/Vessel-Renders/lattice_strength"
if USE_RAW_GT:
    GT_FILES = [
        str(GT_BASE / "lattice_strength_0_25/blender-data/mesh_data_single.npz"),
        str(GT_BASE / "lattice_strength_0_50/blender-data/mesh_data_single.npz"),
        str(GT_BASE / "lattice_strength_0_75/blender-data/mesh_data_single.npz"),
        str(GT_BASE / "lattice_strength_1_00/blender-data/mesh_data_single.npz"),
        str(GT_BASE / "lattice_strength_1_25/blender-data/mesh_data_single.npz"),
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
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_0_25.npy",
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_0_50.npy",
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_0_75.npy",
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_1_00.npy",
    r"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_1_25.npy",
]

# Output directory for heatmaps
OUTPUT_DIR = r"visualization/screenshots"
FRAME_AVERAGE_MODE = "mean"  # "mean" or "median" - how to aggregate distances across frames

# Heatmap customization
# Titles for each metric (use None to auto-generate, or provide custom string)
CHAMFER_TITLE = None    # e.g., "Chamfer Distance Comparison" or None for auto
DELTA_CD_TITLE = None   # e.g., "Temporal Chamfer Delta (ΔCD) Comparison" or None for auto
# FSCORE_TITLE = rf"F-Score Comparison (τ = {FSCORE_TAU}, Mean Across Frames)"     # e.g., "F-Score Comparison" or None for auto
FSCORE_TITLE = None     # e.g., "F-Score Comparison" or None for auto
PRECISION_TITLE = None  # e.g., "Precision Comparison" or None for auto
RECALL_TITLE = None     # e.g., "Recall Comparison" or None for auto

X_AXIS_LABEL = "Ours (mm)"  # Simplified label
Y_AXIS_LABEL = "GT (mm)"    # Simplified label

# Condition labels (row and column labels)
# If None, labels will be auto-generated from experiment names or filenames
# If provided, must match the number of GT_FILES and OURS_FILES respectively
GT_CONDITION_LABELS = ["1", "2", "3", "4", "5"]
OURS_CONDITION_LABELS = ["1", "2", "3", "4", "5"]

# Scaling: Blender Units to mm
CHESS_TOTAL_LENGTH_M = 6.195 / 100.0
CHESS_SQUARES_ALONG_X = 8
CHESS_SQUARE_LENGTH_M = CHESS_TOTAL_LENGTH_M / CHESS_SQUARES_ALONG_X
CALIB_SCALING_FACTOR = 0.667125
CORRECTION_FACTOR = 6.195 / 5.053
UNIT_TO_MM = (CHESS_SQUARE_LENGTH_M / CALIB_SCALING_FACTOR) * 1000.0 * CORRECTION_FACTOR

# ============================================================================
# STYLING & LAYOUT (Multi-panel & Individual)
# ============================================================================
# Font sizes
FS_PANEL_LABEL = 40      # FontSize for A, B, C labels
FS_MAIN_TITLE = 32       # FontSize for figure suptitle
FS_SUB_TITLE = 20        # FontSize for plot titles (F-score Comparison, etc.)
FS_AXIS_LABEL = 24       # FontSize for X/Y axis labels (Ours, GT)
FS_TICK_LABEL = 24       # FontSize for axis tick labels (1mm, 2mm, etc.)
FS_ANNOT = 20            # FontSize for numbers inside the heatmap cells
FS_CBAR_LABEL = 24       # FontSize for colorbar label

# Layout
PLOT_WIDTH = 22          # Total figure width for 2-column layout
ROW_HEIGHT = 8           # Height per row of plots
WSPACE = 0.05             # Horizontal space between columns (increase if labels overlap)
HSPACE = 0.4             # Vertical space between rows
TOP_MARGIN = 0.94        # Space above subplots. Increased from 0.86 since main title was removed.
BOTTOM_MARGIN = 0.08     # Bottom margin
SUPTITLE_Y = 0.985       # Vertical position of the master title (0 to 1)
PANEL_LABEL_Y = 1.07     # Vertical position of A, B, C... labels relative to axis
TITLE_PAD = 20           # Padding between individual plot titles (e.g. F-score Comparison) and their heatmaps


def format_clean_sci(val: float, fmt: str) -> str:
    """
    Format a value in scientific notation, removing redundant zero padding in the exponent.
    Example: '4.5e-04' becomes '4.5e-4'.
    """
    if not np.isfinite(val):
        return str(val)
    s = f"{val:{fmt}}"
    if 'e' in s.lower():
        prefix, exponent = s.lower().split('e')
        return f"{prefix}e{int(exponent)}"
    return s

def compute_chamfer_distance_kdtree(
    points_a: np.ndarray, points_b: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute chamfer distance between two point clouds using KDTree for speed.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf"), float("inf"), float("inf")

    # Use cKDTree for fast nearest neighbor search
    tree_b = cKDTree(points_b)
    dist_a_to_b, _ = tree_b.query(points_a, k=1)
    a_to_b_mean = float(np.mean(dist_a_to_b))

    tree_a = cKDTree(points_a)
    dist_b_to_a, _ = tree_a.query(points_b, k=1)
    b_to_a_mean = float(np.mean(dist_b_to_a))

    # Symmetric chamfer distance
    chamfer_distance = (a_to_b_mean + b_to_a_mean) / 2.0
    return chamfer_distance, a_to_b_mean, b_to_a_mean

def precompute_internal_deltas(cluster_positions: np.ndarray) -> List[float]:
    """
    Precompute frame-to-frame symmetric Chamfer distances within a sequence.
    This avoids redundant calculations during many-to-many comparisons.
    """
    T = cluster_positions.shape[0]
    if T < 2:
        return []
    
    deltas = []
    for t in range(T - 1):
        f1 = cluster_positions[t]
        f2 = cluster_positions[t+1]
        
        # Fast KDTree chamfer
        d, _, _ = compute_chamfer_distance_kdtree(f1, f2)
        deltas.append(d)
    return deltas

def compute_delta_cd_from_deltas(gt_deltas: List[float], ours_deltas: List[float]) -> float:
    """
    Compute ΔCD using precomputed internal sequence deltas.
    """
    T = min(len(gt_deltas), len(ours_deltas))
    if T == 0:
        return float("inf")
    
    diffs = []
    for t in range(T):
        if np.isfinite(gt_deltas[t]) and np.isfinite(ours_deltas[t]):
            diffs.append(abs(gt_deltas[t] - ours_deltas[t]))
            
    if not diffs:
        return float("inf")
        
    return float(np.mean(diffs))

def random_downsample(points: np.ndarray, target_count: int, random_seed: int = None) -> np.ndarray:
    """
    Randomly downsample a point cloud to a target number of points.
    
    Args:
        points: Point cloud (N, 3)
        target_count: Target number of points (must be <= N)
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Downsampled point cloud (target_count, 3)
    """
    if len(points) == 0:
        return points
    
    if target_count >= len(points):
        return points
    
    # Use a local random state to avoid affecting global random state
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random
    
    # Randomly sample indices
    indices = rng.choice(len(points), size=target_count, replace=False)
    return points[indices]


def compute_frame_to_frame_chamfer(
    sequence_positions: np.ndarray,
    t: int,
    chamfer_func
) -> float:
    """
    Compute symmetric Chamfer distance between frame t and t+1 in a sequence.
    
    Args:
        sequence_positions: Cluster positions (T, K, 3) where T is frames, K is clusters
        t: Frame index (must satisfy 0 <= t < T-1)
        chamfer_func: Function to compute Chamfer distance (e.g., compute_chamfer_distance)
        
    Returns:
        Symmetric Chamfer distance between frame t and t+1, or inf if either frame is empty
    """
    if t < 0 or t >= sequence_positions.shape[0] - 1:
        return float("inf")
    
    frame_t = sequence_positions[t]
    frame_t_plus_1 = sequence_positions[t + 1]
    
    # Skip if either frame is empty
    if len(frame_t) == 0 or len(frame_t_plus_1) == 0:
        return float("inf")
    
    # Compute symmetric Chamfer distance
    chamfer_result = chamfer_func(frame_t, frame_t_plus_1)
    return chamfer_result[0]  # Return the main distance metric


def compute_delta_cd(
    gt_cluster_positions: np.ndarray,
    ours_cluster_positions: np.ndarray,
    chamfer_func
) -> float:
    """
    Compute Temporal Chamfer Delta (ΔCD) between GT and ours sequences.
    
    ΔCD measures how similarly two sequences evolve frame-to-frame by comparing
    the frame-to-frame Chamfer distances in each sequence.
    
    Args:
        gt_cluster_positions: GT cluster positions (T_gt, K, 3)
        ours_cluster_positions: Ours cluster positions (T_ours, K, 3)
        chamfer_func: Function to compute Chamfer distance (e.g., compute_chamfer_distance)
        
    Returns:
        Temporal Chamfer Delta (ΔCD), or inf if too many frames are missing
    """
    num_frames_gt = gt_cluster_positions.shape[0]
    num_frames_ours = ours_cluster_positions.shape[0]
    T = min(num_frames_gt, num_frames_ours)
    
    if T < 2:
        return float("inf")  # Need at least 2 frames to compute frame-to-frame distances
    
    delta_cd_values = []
    
    for t in range(T - 1):
        # Compute frame-to-frame Chamfer for ours sequence
        cd_t_pred = compute_frame_to_frame_chamfer(ours_cluster_positions, t, chamfer_func)
        
        # Compute frame-to-frame Chamfer for GT sequence
        cd_t_gt = compute_frame_to_frame_chamfer(gt_cluster_positions, t, chamfer_func)
        
        # Skip if either computation failed (empty frames)
        if not np.isfinite(cd_t_pred) or not np.isfinite(cd_t_gt):
            continue
        
        # Compute absolute difference
        delta_cd_values.append(abs(cd_t_pred - cd_t_gt))
    
    # If too many frames were skipped, return inf
    if len(delta_cd_values) == 0:
        return float("inf")
    
    # Average absolute differences across all valid frame pairs
    delta_cd = np.mean(delta_cd_values)
    return float(delta_cd)

def run_single_comparison(
    i: int, j: int,
    gt_path: Path, ours_path: Path,
    gt_cluster_positions: np.ndarray,
    ours_cluster_positions: np.ndarray,
    gt_deltas: List[float],
    ours_deltas: List[float],
    gt_edges: np.ndarray,
    ours_edges: np.ndarray,
    gt_initial_lengths: np.ndarray,
    ours_initial_lengths: np.ndarray,
    compute_flags: Dict[str, bool],
    unit_to_mm: float,
    fscore_tau: float,
    frame_average_mode: str
) -> Tuple[int, int, Dict[str, float]]:
    """
    Worker function to compute all metrics for a single (GT, Ours) pair.
    """
    results = {}
    
    # 1. Chamfer Distance (last frame)
    if compute_flags.get("CHAMFER"):
        num_frames_gt = gt_cluster_positions.shape[0]
        num_frames_ours = ours_cluster_positions.shape[0]
        
        if num_frames_gt == 0 or num_frames_ours == 0:
            results["chamfer"] = float("inf")
        else:
            gt_last_frame = gt_cluster_positions[-1]
            ours_last_frame = ours_cluster_positions[-1]
            
            # Use KDTree version for speed
            min_point_count = min(len(gt_last_frame), len(ours_last_frame))
            if min_point_count == 0:
                results["chamfer"] = float("inf")
            else:
                # Downsampling for "fairness" as per original implementation
                seed = i * 1000 + j # Unique enough seed
                gt_down = random_downsample(gt_last_frame, min_point_count, random_seed=seed)
                ours_down = random_downsample(ours_last_frame, min_point_count, random_seed=seed + 10000)
                cd, _, _ = compute_chamfer_distance_kdtree(gt_down, ours_down)
                results["chamfer"] = cd * unit_to_mm

    # 2. Temporal Chamfer Delta (ΔCD)
    if compute_flags.get("DELTA_CD"):
        results["delta_cd"] = compute_delta_cd_from_deltas(gt_deltas, ours_deltas) * unit_to_mm

    # 3. Precision/Recall/F-score
    if compute_flags.get("PRECISION") or compute_flags.get("RECALL") or compute_flags.get("F_SCORE"):
        num_frames = min(gt_cluster_positions.shape[0], ours_cluster_positions.shape[0])
        p_vals, r_vals, f_vals = [], [], []
        
        for f in range(num_frames):
            p, r, fsys = compute_precision_recall_fscore(ours_cluster_positions[f], gt_cluster_positions[f], fscore_tau)
            p_vals.append(p)
            r_vals.append(r)
            f_vals.append(fsys)
            
        if p_vals:
            agg = np.mean if frame_average_mode == "mean" else np.median
            results["precision"] = float(agg(p_vals))
            results["recall"] = float(agg(r_vals))
            results["fscore"] = float(agg(f_vals))
        else:
            results["precision"] = results["recall"] = results["fscore"] = 0.0

    return i, j, results


def _resolve_path(file_path: str, project_root: Path) -> Path:
    """
    Resolve a relative or absolute path against the project root and CWD.
    """
    if not os.path.isabs(file_path):
        # Try relative to project root first
        resolved = project_root / file_path
        if not resolved.exists():
            # Fallback: relative to current working directory
            resolved = Path(file_path).resolve()
    else:
        resolved = Path(file_path)
    return resolved


def load_cluster_data(file_path: Path) -> Tuple[np.ndarray, str]:
    """
    Load cluster data from a numpy file (.npy) or raw point cloud (.npz).
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
            return np.array([]), experiment_name
            
        N = len(frames_data[frame_indices[0]]['coords'])
        cluster_positions = np.zeros((T, N, 3))
        for i, f_idx in enumerate(frame_indices):
            cluster_positions[i] = frames_data[f_idx]['coords']
            
        return cluster_positions, experiment_name

    # Load standard npy
    data = np.load(str(file_path), allow_pickle=True).item()
    cluster_positions = data.get("cluster_positions")
    experiment_name = data.get("experiment_name", file_path.stem)
    
    if cluster_positions is None:
        raise ValueError(f"File {file_path} is missing 'cluster_positions' key.")
    
    return cluster_positions, experiment_name


    return precision, recall, fscore


def compute_precision_recall_fscore(
    P: np.ndarray,
    G: np.ndarray,
    tau: float
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F-score between two point sets.
    
    Uses nearest neighbor distances with threshold tau:
    - Precision = (1/|P|) * sum_{p in P} 1[ d_{P->G}(p) < τ ]
    - Recall    = (1/|G|) * sum_{g in G} 1[ d_{G->P}(g) < τ ]
    - F-score   = 2 * Precision * Recall / (Precision + Recall)
    
    Args:
        P: Ours point set (N, 3)
        G: GT point set (M, 3)
        tau: Distance threshold
        
    Returns:
        Tuple of (precision, recall, fscore)
    """
    # Handle empty sets
    if len(P) == 0 or len(G) == 0:
        return 0.0, 0.0, 0.0
    
    # Build KDTree for fast nearest neighbor queries
    G_tree = cKDTree(G)
    P_tree = cKDTree(P)
    
    # Compute d_{P->G}(p) for each point in P
    P_to_G_distances, _ = G_tree.query(P, k=1)
    
    # Compute d_{G->P}(g) for each point in G
    G_to_P_distances, _ = P_tree.query(G, k=1)
    
    # Compute Precision: fraction of P points within tau of G
    precision = float(np.mean(P_to_G_distances < tau))
    
    # Compute Recall: fraction of G points within tau of P
    recall = float(np.mean(G_to_P_distances < tau))
    
    # Compute F-score
    if precision + recall > 0:
        fscore = 2.0 * precision * recall / (precision + recall)
    else:
        fscore = 0.0
    
    return precision, recall, fscore




def draw_heatmap_to_ax(
    ax,
    distance_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    cmap: str = "viridis",
    x_axis_label: str = "Ours",
    y_axis_label: str = "GT",
    cbar_label: str = "Distance",
    fmt: str = ".2e",
    panel_label: str = None
) -> None:
    """
    Draw a professional heatmap onto a specific Matplotlib axis.
    """
    # Determine if we should use custom scientific formatting for annotations
    if fmt.endswith('e'):
        annot_labels = np.array([[format_clean_sci(val, fmt) for val in row] for row in distance_matrix])
        use_annot = annot_labels
        use_fmt = ""  # Disable standard formatting since labels are already strings
    else:
        use_annot = True
        use_fmt = fmt

    # Create heatmap
    heatmap = sns.heatmap(
        distance_matrix,
        annot=use_annot,
        annot_kws={"size": FS_ANNOT, "weight": "bold"},
        fmt=use_fmt,
        cmap=cmap,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        square=True,
        cbar=True
    )
    
    # Format colorbar based on format string
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FS_TICK_LABEL - 2)
    if fmt.endswith('e'):
        # Use same clean scientific formatting for colorbar ticks
        # The colorbar typically needs a bit more precision than the cells
        # so we use a fallback if fmt is too short, or just use fmt directly
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format_clean_sci(x, fmt)))
    else:
        if fmt == ".4f":
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.4f}'))
        elif fmt == ".3f":
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3f}'))
        else:
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.4f}'))
    
    # Professional styling
    # ax.set_title(title, fontsize=FS_SUB_TITLE, fontweight="bold", pad=TITLE_PAD)
    ax.set_xlabel(x_axis_label, fontsize=FS_AXIS_LABEL)
    ax.set_ylabel(y_axis_label, fontsize=FS_AXIS_LABEL)
    cbar.ax.set_ylabel(cbar_label, fontsize=FS_CBAR_LABEL)
    
    # Improve tick label readability
    ax.tick_params(axis='x', labelsize=FS_TICK_LABEL, rotation=0, labelbottom=True)
    ax.tick_params(axis='y', labelsize=FS_TICK_LABEL, rotation=0, labelleft=True)
    
    # Bold the tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Add Panel Label (A, B, C...)
    if panel_label:
        # Explicitly use DejaVu Sans for panel labels with 'black' weight for maximum boldness
        t = ax.text(0.0, PANEL_LABEL_Y, panel_label, transform=ax.transAxes, 
                   fontsize=FS_PANEL_LABEL, fontweight='black', fontname='DejaVu Sans',
                   va='bottom', ha='left')

def create_heatmap(
    distance_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    save_path: str,
    cmap: str = "viridis",
    x_axis_label: str = "Ours",
    y_axis_label: str = "GT",
    cbar_label: str = "Distance",
    fmt: str = ".2e"
) -> None:
    """
    Create and save a professional heatmap visualization (legacy standalone wrapper).
    """
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.2), max(8, len(row_labels) * 0.8)))
    draw_heatmap_to_ax(ax, distance_matrix, row_labels, col_labels, title, cmap, x_axis_label, y_axis_label, cbar_label, fmt)
    plt.tight_layout()
    pdf_path = save_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap to: {Path(pdf_path).name}")


def main():
    """
    Main entry point for multiple GT vs Ours comparison with heatmaps.
    """
    # Determine project root (script directory's parent, since script is in visualization/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("=" * 80)
    print("MULTIPLE GT vs OURS COMPARISON - HEATMAP GENERATION")
    print("=" * 80)
    
    # Resolve all file paths
    print(f"\nResolving file paths...")
    gt_paths = []
    ours_paths = []
    
    for gt_file in GT_FILES:
        resolved = _resolve_path(gt_file, project_root)
        if not resolved.exists():
            print(f"⚠️  WARNING: GT file not found: {gt_file}")
            print(f"   Resolved path: {resolved}")
        else:
            gt_paths.append(resolved)
            print(f"  ✓ GT: {resolved.name}")
    
    for ours_file in OURS_FILES:
        resolved = _resolve_path(ours_file, project_root)
        if not resolved.exists():
            print(f"⚠️  WARNING: Ours file not found: {ours_file}")
            print(f"   Resolved path: {resolved}")
        else:
            ours_paths.append(resolved)
            print(f"  ✓ Ours: {resolved.name}")
    
    if len(gt_paths) == 0:
        print("\n⚠️  ERROR: No valid GT files found!")
        return
    
    if len(ours_paths) == 0:
        print("\n⚠️  ERROR: No valid ours files found!")
        return
    
    print(f"\nFound {len(gt_paths)} GT files and {len(ours_paths)} ours files")
    print(f"Total comparisons: {len(gt_paths) * len(ours_paths)}")
    
    # Load all GT data
    print(f"\nLoading GT files...")
    gt_data = {}
    gt_experiment_names = {}  # Store experiment names for validation
    gt_labels = []
    for gt_path in gt_paths:
        try:
            cluster_positions, experiment_name = load_cluster_data(gt_path)
            gt_data[gt_path] = cluster_positions
            gt_experiment_names[gt_path] = experiment_name
            # Use custom label if provided, otherwise use experiment name or filename
            if GT_CONDITION_LABELS is not None and len(gt_labels) < len(GT_CONDITION_LABELS):
                label = GT_CONDITION_LABELS[len(gt_labels)]
            else:
                label = experiment_name if experiment_name else gt_path.stem
            gt_labels.append(label)
            print(f"  ✓ {gt_path.name}: {cluster_positions.shape[0]} frames, {cluster_positions.shape[1]} clusters")
        except Exception as e:
            print(f"  ✗ ERROR loading {gt_path.name}: {e}")
            gt_experiment_names[gt_path] = None
            # Use custom label if available, otherwise use filename
            if GT_CONDITION_LABELS is not None and len(gt_labels) < len(GT_CONDITION_LABELS):
                gt_labels.append(GT_CONDITION_LABELS[len(gt_labels)])
            else:
                gt_labels.append(gt_path.stem)
    
    # Load all Ours data
    print(f"\nLoading Ours files...")
    ours_data = {}
    ours_experiment_names = {}  # Store experiment names for validation
    ours_labels = []
    for ours_path in ours_paths:
        try:
            cluster_positions, experiment_name = load_cluster_data(ours_path)
            ours_data[ours_path] = cluster_positions
            ours_experiment_names[ours_path] = experiment_name
            # Use custom label if provided, otherwise use experiment name or filename
            if OURS_CONDITION_LABELS is not None and len(ours_labels) < len(OURS_CONDITION_LABELS):
                label = OURS_CONDITION_LABELS[len(ours_labels)]
            else:
                label = experiment_name if experiment_name else ours_path.stem
            ours_labels.append(label)
            print(f"  ✓ {ours_path.name}: {cluster_positions.shape[0]} frames, {cluster_positions.shape[1]} clusters")
        except Exception as e:
            print(f"  ✗ ERROR loading {ours_path.name}: {e}")
            ours_experiment_names[ours_path] = None
            # Use custom label if available, otherwise use filename
            if OURS_CONDITION_LABELS is not None and len(ours_labels) < len(OURS_CONDITION_LABELS):
                ours_labels.append(OURS_CONDITION_LABELS[len(ours_labels)])
            else:
                ours_labels.append(ours_path.stem)
    
    # Validate custom labels if provided (check after loading to ensure counts match)
    if GT_CONDITION_LABELS is not None and len(GT_CONDITION_LABELS) != len(gt_labels):
        print(f"\n⚠️  WARNING: GT_CONDITION_LABELS has {len(GT_CONDITION_LABELS)} labels but {len(gt_labels)} GT files loaded.")
        print(f"   Using auto-generated labels instead.")
        # Re-generate labels without custom ones
        gt_labels = []
        for gt_path in gt_paths:
            experiment_name = gt_experiment_names.get(gt_path)
            label = experiment_name if experiment_name else gt_path.stem
            gt_labels.append(label)
    
    if OURS_CONDITION_LABELS is not None and len(OURS_CONDITION_LABELS) != len(ours_labels):
        print(f"\n⚠️  WARNING: OURS_CONDITION_LABELS has {len(OURS_CONDITION_LABELS)} labels but {len(ours_labels)} ours files loaded.")
        print(f"   Using auto-generated labels instead.")
        # Re-generate labels without custom ones
        ours_labels = []
        for ours_path in ours_paths:
            experiment_name = ours_experiment_names.get(ours_path)
            label = experiment_name if experiment_name else ours_path.stem
            ours_labels.append(label)
    
    # Precompute internal deltas for all sequences
    print(f"\nPrecomputing internal sequence deltas (for ΔCD optimization)...")
    gt_deltas_map = {}
    for gt_path in gt_paths:
        if gt_path in gt_data:
            print(f"  - {gt_path.name}...")
            gt_deltas_map[gt_path] = precompute_internal_deltas(gt_data[gt_path])
            
    ours_deltas_map = {}
    for ours_path in ours_paths:
        if ours_path in ours_data:
            print(f"  - {ours_path.name}...")
            ours_deltas_map[ours_path] = precompute_internal_deltas(ours_data[ours_path])

    # Compute distance matrices
    print(f"\nComputing distances (mode: {FRAME_AVERAGE_MODE})...")
    start_time = time.time()
    
    # Prepare comparison tasks
    tasks = []
    compute_flags = {
        "CHAMFER": COMPUTE_CHAMFER,
        "DELTA_CD": COMPUTE_DELTA_CD,
        "F_SCORE": COMPUTE_F_SCORE,
        "PRECISION": COMPUTE_PRECISION,
        "RECALL": COMPUTE_RECALL
    }
    
    for i, gt_path in enumerate(gt_paths):
        if gt_path not in gt_data: continue
        for j, ours_path in enumerate(ours_paths):
            if ours_path not in ours_data: continue
            
            tasks.append((
                i, j,
                gt_path, ours_path,
                gt_data[gt_path],
                ours_data[ours_path],
                gt_deltas_map.get(gt_path, []),
                ours_deltas_map.get(ours_path, []),
                None, # gt_edges
                None, # ours_edges
                None, # gt_initial_lengths
                None, # ours_initial_lengths
                compute_flags,
                UNIT_TO_MM,
                FSCORE_TAU,
                FRAME_AVERAGE_MODE
            ))

    # Initialize matrices
    if COMPUTE_CHAMFER: chamfer_matrix = np.zeros((len(gt_paths), len(ours_paths)))
    if COMPUTE_DELTA_CD: delta_cd_matrix = np.zeros((len(gt_paths), len(ours_paths)))
    if COMPUTE_PRECISION: precision_matrix = np.zeros((len(gt_paths), len(ours_paths)))
    if COMPUTE_RECALL: recall_matrix = np.zeros((len(gt_paths), len(ours_paths)))
    if COMPUTE_F_SCORE: fscore_matrix = np.zeros((len(gt_paths), len(ours_paths)))

    # Run in parallel
    num_workers = NUM_WORKERS if NUM_WORKERS > 0 else multiprocessing.cpu_count()
    print(f"Running {len(tasks)} comparisons across {num_workers} parallel workers...")
    
    total_tasks = len(tasks)
    completed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(run_single_comparison, *task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                i, j, res = future.result()
                if "chamfer" in res: chamfer_matrix[i, j] = res["chamfer"]
                if "delta_cd" in res: delta_cd_matrix[i, j] = res["delta_cd"]
                if "precision" in res: precision_matrix[i, j] = res["precision"]
                if "recall" in res: recall_matrix[i, j] = res["recall"]
                if "fscore" in res: fscore_matrix[i, j] = res["fscore"]
                
                completed += 1
                progress = int((completed / total_tasks) * 50)
                bar = "█" * progress + "░" * (50 - progress)
                print(f"\r[{bar}] {completed}/{total_tasks} comparisons finished", end="", flush=True)
            except Exception as e:
                print(f"\n  ✗ Error in comparison: {e}")

    duration = time.time() - start_time
    print(f"\nComputation finished in {duration:.2f} seconds.")
    
    print()  # New line after progress bar
    
    # Print summary statistics for enabled metrics
    if COMPUTE_CHAMFER and chamfer_matrix is not None:
        print(f"\nChamfer Distance Statistics:")
        valid_chamfer = chamfer_matrix[np.isfinite(chamfer_matrix)]
        if len(valid_chamfer) > 0:
            print(f"  Min: {np.min(valid_chamfer):.4f} mm")
            print(f"  Max: {np.max(valid_chamfer):.4f} mm")
            print(f"  Mean: {np.mean(valid_chamfer):.4f} mm")
            print(f"  Median: {np.median(valid_chamfer):.4f} mm")
    
    if COMPUTE_DELTA_CD and delta_cd_matrix is not None:
        print(f"\nTemporal Chamfer Delta (ΔCD) Statistics:")
        valid_delta_cd = delta_cd_matrix[np.isfinite(delta_cd_matrix)]
        if len(valid_delta_cd) > 0:
            print(f"  Min: {np.min(valid_delta_cd):.4e} mm")
            print(f"  Max: {np.max(valid_delta_cd):.4e} mm")
            print(f"  Mean: {np.mean(valid_delta_cd):.4e} mm")
            print(f"  Median: {np.median(valid_delta_cd):.4e} mm")
    
    if COMPUTE_F_SCORE and fscore_matrix is not None:
        print(f"\nPrecision Statistics (τ={FSCORE_TAU}):")
        valid_precision = precision_matrix[precision_matrix >= 0]
        if len(valid_precision) > 0:
            print(f"  Min: {np.min(valid_precision):.4f}")
            print(f"  Max: {np.max(valid_precision):.4f}")
            print(f"  Mean: {np.mean(valid_precision):.4f}")
            print(f"  Median: {np.median(valid_precision):.4f}")
    
    if COMPUTE_RECALL and recall_matrix is not None:
        print(f"\nRecall Statistics (τ={FSCORE_TAU}):")
        valid_recall = recall_matrix[recall_matrix >= 0]
        if len(valid_recall) > 0:
            print(f"  Min: {np.min(valid_recall):.4f}")
            print(f"  Max: {np.max(valid_recall):.4f}")
            print(f"  Mean: {np.mean(valid_recall):.4f}")
            print(f"  Median: {np.median(valid_recall):.4f}")
    
    if COMPUTE_F_SCORE and fscore_matrix is not None:
        print(f"\nF-Score Statistics (τ={FSCORE_TAU}):")
        valid_fscore = fscore_matrix[fscore_matrix >= 0]
        if len(valid_fscore) > 0:
            print(f"  Min: {np.min(valid_fscore):.4f}")
            print(f"  Max: {np.max(valid_fscore):.4f}")
            print(f"  Mean: {np.mean(valid_fscore):.4f}")
            print(f"  Median: {np.median(valid_fscore):.4f}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Create and save heatmaps for enabled metrics
    print(f"\nGenerating heatmaps...")
    saved_heatmaps = []

    # Prepare data for multi-panel if enabled
    multipanel_data = []

    # Helper to track status of current metrics
    metric_map = {
        "Chamfer": (COMPUTE_CHAMFER, chamfer_matrix if COMPUTE_CHAMFER else None, CHAMFER_TITLE or "Chamfer Distance (mm) Comparison", "Chamfer Distance (mm)", ".4f", "viridis"),
        "Delta CD": (COMPUTE_DELTA_CD, delta_cd_matrix if COMPUTE_DELTA_CD else None, DELTA_CD_TITLE or "Temporal Chamfer Delta (mm) Comparison", "ΔCD (mm)", ".1e", "viridis"),
        "F-score": (COMPUTE_F_SCORE, fscore_matrix if COMPUTE_F_SCORE else None, FSCORE_TITLE or "F-score Comparison", "F-score", ".4f", "viridis"),
        "Precision": (COMPUTE_PRECISION, precision_matrix if COMPUTE_PRECISION else None, PRECISION_TITLE or "Precision Comparison", "Precision", ".4f", "viridis"),
        "Recall": (COMPUTE_RECALL, recall_matrix if COMPUTE_RECALL else None, RECALL_TITLE or "Recall Comparison", "Recall", ".4f", "viridis")
    }

    # Individual Heatmaps (only if multi-panel is disabled)
    if not GENERATE_MULTIPANEL:
        for m_name, (enabled, matrix, title, cbar_label, fmt, cmap) in metric_map.items():
            if enabled and matrix is not None:
                save_name = f"{m_name.lower().replace(' ', '_')}_heatmap_{timestamp}.png"
                create_heatmap(matrix, gt_labels, ours_labels, title, str(output_dir / save_name), cmap, X_AXIS_LABEL, Y_AXIS_LABEL, cbar_label, fmt)
                saved_heatmaps.append(f"  - {m_name}: {save_name} (+ PDF)")

    # Multi-panel Heatmap
    if GENERATE_MULTIPANEL:
        from matplotlib import gridspec
        valid_metrics = []
        for m_name in MULTIPANEL_METRICS:
            if m_name in metric_map:
                enabled, matrix, title, cbar_label, fmt, cmap = metric_map[m_name]
                if enabled and matrix is not None:
                    valid_metrics.append((m_name, matrix, title, cbar_label, fmt, cmap))
        
        if valid_metrics:
            n_panels = len(valid_metrics)
            n_cols = 2
            n_rows = (n_panels + 1) // 2
            
            # Figure size using centralized constants
            fig = plt.figure(figsize=(PLOT_WIDTH, ROW_HEIGHT * n_rows))
            
            # Use 4 sub-columns to allow centering of an odd panel
            gs = gridspec.GridSpec(n_rows, 4)
            
            for idx, (m_name, matrix, title, cbar_label, fmt, cmap) in enumerate(valid_metrics):
                panel_label = chr(65 + idx) # A, B, C...
                curr_row = idx // 2
                
                # If it's the last one and total count is odd, center it
                if idx == n_panels - 1 and n_panels % 2 != 0:
                    ax = fig.add_subplot(gs[curr_row, 1:3]) # Mid columns
                else:
                    col_start = (idx % 2) * 2
                    ax = fig.add_subplot(gs[curr_row, col_start:col_start+2])
                
                draw_heatmap_to_ax(ax, matrix, gt_labels, ours_labels, title, cmap, X_AXIS_LABEL, Y_AXIS_LABEL, cbar_label, fmt, panel_label=panel_label)
            
# fig.suptitle(MULTIPANEL_TITLE, fontsize=FS_MAIN_TITLE, fontweight='bold', y=SUPTITLE_Y)
            # Adjust spacing between subplots using centralized constants
            plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE, top=TOP_MARGIN, bottom=BOTTOM_MARGIN, left=0.1, right=0.9)
            
            pdf_path = output_dir / f"multipanel_heatmap_{timestamp}.pdf"
            plt.savefig(str(pdf_path), bbox_inches="tight")
            plt.close()
            print(f"  Saved multi-panel heatmap to: {pdf_path.name}")
            saved_heatmaps.append(f"  - Multi-panel: {pdf_path.name}")

    print(f"\n✓ Completed! Heatmaps saved to: {output_dir.absolute()}")
    for heatmap_info in saved_heatmaps:
        print(heatmap_info)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n⚠️  ERROR:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()