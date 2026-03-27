#!/usr/bin/env python3
"""
Export per-vertex stress timeseries for vessel experiments.
Aggregates edge-based stresses from the cluster mesh models into vertex-based stresses.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import vessel utils for stress calculation if available, 
# but we will also implement local versions for robustness
try:
    import utils.vessel_utils as vu
except ImportError:
    vu = None

# ============================================================================
# CONFIGURATION
# ============================================================================
# List of experiment identifiers to process
EXPERIMENTS = [f"a{i}" for i in range(5, 19)]

# Base directory for processed experiment data
BASE_DATA_DIR = "./data/processed"

# Default stress aggregation method: "mean" or "max"
AGGREGATION_METHOD = "mean"

# Material properties for stress calculation
YOUNG_MODULUS = vu.YOUNG_MODULUS_SILICON if vu else 1.15e6  # Silicone young modulus (Pa)
POISSON_RATIO = 0.5         # Incompressible

# Filename patterns
INPUT_FOLDER = "cluster_mesh_export"
INPUT_FILENAME_TEMPLATE = "cluster_mesh_timeseries_{exp}.npy"
OUTPUT_FILENAME_TEMPLATE = "vertex_stress_timeseries_{exp}.npy"

# ============================================================================

def faces_to_edges(faces: np.ndarray) -> np.ndarray:
    """
    Convert a triangle faces array to a unique edges array.
    """
    if faces is None or len(faces) == 0:
        return np.array([]).reshape(0, 2)
    
    edges_set = set()
    for f in faces:
        # Sort each tuple to ensure uniqueness regardless of orientation
        edges_set.add(tuple(sorted((f[0], f[1]))))
        edges_set.add(tuple(sorted((f[1], f[2]))))
        edges_set.add(tuple(sorted((f[2], f[0]))))
    
    return np.array(list(edges_set), dtype=np.int32)


def compute_edge_stresses(
    positions: np.ndarray, 
    edges: np.ndarray, 
    initial_lengths: np.ndarray, 
    young_modulus: float,
    poisson_ratio: float = 0.5
) -> np.ndarray:
    """
    Compute rigid/neo-hookean stress for each edge in a frame.
    """
    if len(edges) == 0 or len(positions) == 0:
        return np.array([])
    
    # Calculate current edge lengths
    v1 = positions[edges[:, 0]]
    v2 = positions[edges[:, 1]]
    current_lengths = np.linalg.norm(v1 - v2, axis=1)
    
    # Avoid division by zero
    safe_initial = np.where(initial_lengths > 0, initial_lengths, 1.0)
    lam = current_lengths / safe_initial
    
    # Shear modulus
    mu = young_modulus / (2 * (1 + poisson_ratio))
    
    # Stress (Neo-Hookean proxy)
    # sigma = mu * (lam^2 - 1/lam)
    # We use a small epsilon to avoid div by zero if lam is 0 (though physical points shouldn't overlap)
    lam_safe = np.where(lam > 1e-6, lam, 1e-6)
    stress = mu * (lam_safe**2 - 1.0 / lam_safe)
    
    # zero out where initial length was zero
    stress[initial_lengths <= 0] = 0
    
    return stress


def edge_to_vertex_stress(
    edge_stresses: np.ndarray, 
    edges: np.ndarray, 
    num_vertices: int, 
    method: str = "mean"
) -> np.ndarray:
    """
    Aggregate edge-based stresses onto vertices.
    """
    v_stress = np.zeros(num_vertices, dtype=np.float64)
    v_count = np.zeros(num_vertices, dtype=np.int32)
    
    if method == "mean":
        # Accumulate values and counts
        for i, (u, v) in enumerate(edges):
            s = edge_stresses[i]
            if np.isfinite(s):
                v_stress[u] += s
                v_count[u] += 1
                v_stress[v] += s
                v_count[v] += 1
        
        # Compute mean
        mask = v_count > 0
        v_stress[mask] /= v_count[mask]
        
    elif method == "max":
        v_stress.fill(-np.inf)
        for i, (u, v) in enumerate(edges):
            s = edge_stresses[i]
            if np.isfinite(s):
                v_stress[u] = max(v_stress[u], s)
                v_stress[v] = max(v_stress[v], s)
        
        # Clean up -inf for unvisited vertices
        v_stress[v_stress == -np.inf] = 0
        
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
        
    return v_stress


def process_experiment(exp_name: str) -> None:
    """
    Load data, compute vertex stresses, and save results for a single experiment.
    """
    exp_dir = Path(BASE_DATA_DIR) / exp_name
    input_dir = exp_dir / INPUT_FOLDER
    input_path = input_dir / INPUT_FILENAME_TEMPLATE.format(exp=exp_name)
    output_path = input_dir / OUTPUT_FILENAME_TEMPLATE.format(exp=exp_name)
    
    if not input_path.exists():
        print(f"  [SKIPPED] Input file does not exist: {input_path}")
        return
    
    print(f"  Processing {exp_name}...")
    
    # 1. Load data
    try:
        data = np.load(str(input_path), allow_pickle=True).item()
    except Exception as e:
        print(f"  [ERROR] Failed to load {input_path}: {e}")
        return
    
    # Check for required keys
    required_keys = ["cluster_positions", "edges", "initial_lengths"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        available_keys = list(data.keys())
        raise KeyError(f"Missing required keys in {input_path}: {missing_keys}. Available keys: {available_keys}")
    
    cluster_positions = data["cluster_positions"]  # (T, N, 3)
    edges = data["edges"]                          # (E, 2)
    initial_lengths = data["initial_lengths"]      # (E,)
    times = data.get("times", np.arange(cluster_positions.shape[0]))
    faces = data.get("faces", np.array([]))         # May be missing
    
    num_frames = cluster_positions.shape[0]
    num_vertices = cluster_positions.shape[1]
    
    # 2. Compute vertex stresses over time
    vertex_stress_timeseries = np.zeros((num_frames, num_vertices), dtype=np.float64)
    
    for t in range(num_frames):
        # Progress (optional, but keep quiet for loop)
        positions_t = cluster_positions[t]
        
        # Calculate edge stresses for this frame
        edge_stresses = compute_edge_stresses(
            positions_t, 
            edges, 
            initial_lengths, 
            YOUNG_MODULUS, 
            POISSON_RATIO
        )
        
        # Map to vertices
        v_stress = edge_to_vertex_stress(
            edge_stresses, 
            edges, 
            num_vertices, 
            method=AGGREGATION_METHOD
        )
        
        vertex_stress_timeseries[t] = v_stress
    
    # 3. Save results
    output_data = {
        "vertex_stress": vertex_stress_timeseries,
        "vertices": cluster_positions,
        "faces": faces, # Might be empty
        "times": times,
        "experiment_name": exp_name
    }
    
    np.save(str(output_path), output_data, allow_pickle=True)
    
    # 4. Print summary
    valid_stress = vertex_stress_timeseries[np.isfinite(vertex_stress_timeseries)]
    if len(valid_stress) > 0:
        s_min, s_max = np.min(valid_stress), np.max(valid_stress)
    else:
        s_min, s_max = 0, 0
        
    print(f"    ✓ Exported to {output_path}")
    print(f"    Summary: {num_frames} frames, {num_vertices} vertices")
    print(f"    Stress: min={s_min:.2e} Pa, max={s_max:.2e} Pa")


def main():
    print(f"=== EXPORT VERTEX STRESS TIMESERIES ===")
    print(f"Aggregation: {AGGREGATION_METHOD}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Young's Modulus: {YOUNG_MODULUS:.2e} Pa")
    print("-" * 40)
    
    for exp in EXPERIMENTS:
        try:
            process_experiment(exp)
        except Exception as e:
            print(f"  [ERROR] Experiment {exp} failed: {e}")
            import traceback
            traceback.print_exc()
            
    print("-" * 40)
    print("Process completed.")

if __name__ == "__main__":
    main()
