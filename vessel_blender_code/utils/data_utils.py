"""
Data loading utilities for numpy files and cluster data.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Add project root to path for vessel_utils import
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import vessel_utils as vu

from utils.file_utils import resolve_path, get_project_root


def load_clustered_numpy(
    clustered_npy_path: str,
    project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load clustered numpy file exported by export_cluster_mesh.py.
    
    Args:
        clustered_npy_path: Path to the exported cluster mesh numpy file
        project_root: Project root directory (optional, will be inferred if not provided)
        
    Returns:
        Dictionary containing:
            - cluster_positions: (T, K, 3) array
            - times: (T,) array
            - initial_cluster_means: (K, 3) array
            - edges: (E, 2) array or None
            - initial_lengths: (E,) array or None
            - experiment_name: str
    """
    if project_root is None:
        project_root = get_project_root()
    
    # Resolve path
    resolved_path = resolve_path(clustered_npy_path, project_root)
    
    if not resolved_path.exists():
        raise FileNotFoundError(f"Clustered numpy file not found: {clustered_npy_path}")
    
    print(f"Loading clustered numpy file: {resolved_path}")
    data = np.load(str(resolved_path), allow_pickle=True).item()
    
    cluster_positions = data.get("cluster_positions")  # (T, K, 3)
    times = data.get("times")  # (T,)
    initial_cluster_means = data.get("initial_cluster_means")  # (K, 3)
    edges = data.get("edges")  # (E, 2) or None
    initial_lengths = data.get("initial_lengths")  # (E,) or None
    experiment_name = data.get("experiment_name", "UnknownExperiment")
    
    if cluster_positions is None:
        raise ValueError("Clustered numpy file missing 'cluster_positions' key")
    
    num_frames, num_clusters, _ = cluster_positions.shape
    
    print(f"  Loaded {num_frames} frames, {num_clusters} clusters")
    if edges is not None:
        print(f"  Edges: {len(edges)}")
    print(f"  Experiment: {experiment_name}")
    
    return {
        "cluster_positions": cluster_positions,
        "times": times,
        "initial_cluster_means": initial_cluster_means,
        "edges": edges,
        "initial_lengths": initial_lengths,
        "experiment_name": experiment_name,
    }


def clustered_numpy_to_frames_data(
    cluster_positions: np.ndarray,
    times: np.ndarray
) -> Dict[int, Dict[str, Any]]:
    """
    Convert cluster positions array to frames_data format.
    
    Args:
        cluster_positions: (T, K, 3) array of cluster positions over time
        times: (T,) array of time values
        
    Returns:
        Dictionary mapping frame indices to frame data
    """
    num_frames = cluster_positions.shape[0]
    frames_data = {}
    
    for frame_idx in range(num_frames):
        cluster_means = cluster_positions[frame_idx]  # (K, 3)
        time_val = float(times[frame_idx]) if times is not None and len(times) > frame_idx else float(frame_idx)
        
        frames_data[frame_idx] = {
            "cluster_means": cluster_means,
            "num_clusters": len(cluster_means),
            "time": time_val,
        }
    
    return frames_data


def load_npz_file(
    npz_path: str,
    project_root: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load npz file containing mesh data.
    
    Args:
        npz_path: Path to the .npz file
        project_root: Project root directory (optional)
        
    Returns:
        Tuple of (coords, frame_numbers, times)
    """
    if project_root is None:
        project_root = get_project_root()
    
    resolved_path = resolve_path(npz_path, project_root)
    
    if not resolved_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    print(f"Loading npz file: {resolved_path}")
    npz_data = np.load(str(resolved_path))
    
    coords = npz_data["coords"]
    frame_numbers = npz_data["frame_numbers"]
    times = npz_data["times"]
    
    return coords, frame_numbers, times

