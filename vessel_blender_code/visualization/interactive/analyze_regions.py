#!/usr/bin/env python3
"""
Analyzes both displacement and stress statistics (Average and Median) for specific 
regions across all GT and Clustered Ours vessels.

If USE_RAW_GT is True, GT data is loaded from raw .npz files and meshed with Delaunay on the fly.
Otherwise, GT data is loaded from pre-clustered .npy files.
Ours data is always loaded from pre-clustered .npy files.

Outputs:
    - region_analysis_[raw_]disp_average.csv
    - region_analysis_[raw_]disp_median.csv
    - region_analysis_[raw_]stress_average.csv
    - region_analysis_[raw_]stress_median.csv
"""

import sys
import os
import json
import numpy as np
import csv
from pathlib import Path
from scipy.spatial.distance import cdist

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import vessel_utils as vu

# ============================================================================
# CONFIGURATION
# ============================================================================

# Flag to toggle between Raw GT and Clustered GT
USE_RAW_GT = True

# Use the centralized UNIT_TO_MM from vessel_utils
UNIT_TO_MM = vu.UNIT_TO_MM
YOUNG_MODULUS_SILICON = vu.YOUNG_MODULUS_SILICON

# Meshing parameters for Raw GT
OUTLIER_EDGE_THRESHOLD = 1.8
MAX_EDGE_LENGTH_MM = 1.0

# Conditions and Labels
# Use 0.25..1.25 to stay consistent with lattice strength filenames
CONDITIONS = ["0.25", "0.50", "0.75", "1.00", "1.25"]
COND_LABELS = ["0_25", "0_50", "0_75", "1_00", "1_25"]

# GT Paths
GT_BASE = Path.home() / "Documents/Blender/Vessel-Renders/lattice_strength"
GT_NPZ_FILES = [
    GT_BASE / f"lattice_strength_{c}/blender-data/mesh_data_single.npz" 
    for c in COND_LABELS
]
GT_NPY_FILES = [
    f"cluster_data/cluster_mesh_timeseries_GT_lattice_strength_{c}.npy"
    for c in COND_LABELS
]

# Ours Paths (Clustered .npy)
OURS_NPY_FILES = [
    f"cluster_data/lattice_strength_processed4/cluster_export_lattice_strength_{c}.npy"
    for c in COND_LABELS
]

# ============================================================================

def load_regions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_npz_raw(file_path):
    """Load raw npz data and organize by frames."""
    print(f"Loading raw GT npz: {file_path}")
    data = np.load(file_path)
    coords = data['coords']
    frame_numbers = data['frame_numbers']
    times = data['times']
    
    # Organize by frames
    frames_data = vu.get_all_frames_data(coords, frame_numbers, times)
    return frames_data

def load_npy_clustered(file_path, project_root):
    """Load clustered npy data."""
    abs_path = project_root / file_path
    print(f"Loading clustered npz: {abs_path}")
    data = np.load(str(abs_path), allow_pickle=True).item()
    cluster_positions = data.get("cluster_positions")
    edges = data.get("edges")
    initial_lengths = data.get("initial_lengths")
    return cluster_positions, edges, initial_lengths

def process_clustered_frames(file_rel_path, project_root, cond_label, vessel_type, regions_data, results):
    file_path = project_root / file_rel_path
    if not file_path.exists():
        print(f"Skipping missing {vessel_type} file: {file_path}")
        return 0
        
    print(f"\n--- Processing {vessel_type} (Clustered): {cond_label} ---")
    cluster_positions, edges, initial_lengths = load_npy_clustered(file_rel_path, project_root)
    num_frames = cluster_positions.shape[0]
    init_pos = cluster_positions[0]
    
    # Region Indices
    region_point_indices = {}
    region_edge_indices = {}
    for r in regions_data:
        center = np.array(r['center'])
        radius = r['radius']
        dists = cdist([center], init_pos).flatten()
        pt_indices = np.where(dists <= radius)[0]
        region_point_indices[r['title']] = pt_indices
        
        pt_set = set(pt_indices)
        edge_mask = []
        if edges is not None and len(edges) > 0:
            for edge in edges:
                if edge[0] in pt_set or edge[1] in pt_set:
                    edge_mask.append(True)
                else:
                    edge_mask.append(False)
            region_edge_indices[r['title']] = np.where(edge_mask)[0]
        else:
            region_edge_indices[r['title']] = np.array([])

    # Loop frames
    for fidx in range(num_frames):
        curr_pos = cluster_positions[fidx]
        all_disps_mm = np.linalg.norm(curr_pos - init_pos, axis=1) * UNIT_TO_MM
        if edges is not None and len(edges) > 0:
            all_stresses = vu.compute_edge_stress(curr_pos, edges, initial_lengths, YOUNG_MODULUS_SILICON)
        else:
            all_stresses = np.array([])
        
        for r in regions_data:
            r_title = r['title']
            key = (vessel_type, cond_label, r_title)
            
            pt_idx = region_point_indices[r_title]
            ed_idx = region_edge_indices[r_title]
            
            d_avg = np.mean(all_disps_mm[pt_idx]) if len(pt_idx) > 0 else 0.0
            d_med = np.median(all_disps_mm[pt_idx]) if len(pt_idx) > 0 else 0.0
            s_avg = np.mean(np.abs(all_stresses[ed_idx])) if len(ed_idx) > 0 else 0.0
            s_med = np.median(np.abs(all_stresses[ed_idx])) if len(ed_idx) > 0 else 0.0
            
            for m, val in [("Disp_Average", d_avg), ("Disp_Median", d_med), 
                           ("Stress_Average", s_avg), ("Stress_Median", s_med)]:
                if key not in results[m]: results[m][key] = []
                results[m][key].append(val)
        
        if fidx % 20 == 0: vu.print_progress_bar(fidx + 1, num_frames, bar_length=20)
    print()
    return num_frames

def process_raw_frames(npz_path, cond_label, vessel_type, regions_data, results):
    if not npz_path.exists():
        print(f"Skipping missing RAW {vessel_type} file: {npz_path}")
        return 0
        
    print(f"\n--- Processing {vessel_type} (RAW): {cond_label} ---")
    frames_data = load_npz_raw(npz_path)
    frame_indices = sorted(frames_data.keys())
    num_frames = len(frame_indices)
    
    first_frame_idx = frame_indices[0]
    first_coords = frames_data[first_frame_idx]['coords']
    
    print(f"Generating Delaunay mesh for GT ({len(first_coords)} points)...")
    _, edges_raw, _ = vu.create_delaunay_edges(first_coords)
    
    valid_edges_rel, _ = vu.remove_outlier_edges(first_coords, edges_raw, OUTLIER_EDGE_THRESHOLD)
    initial_lengths_raw = np.linalg.norm(
        first_coords[valid_edges_rel[:, 0]] - first_coords[valid_edges_rel[:, 1]], axis=1
    )
    abs_limit_internal = MAX_EDGE_LENGTH_MM / vu.UNIT_TO_MM
    abs_mask = initial_lengths_raw <= abs_limit_internal
    valid_edges = valid_edges_rel[abs_mask]
    initial_lengths = initial_lengths_raw[abs_mask]
    
    print(f"  Valid {vessel_type} edges: {len(valid_edges)} ({len(edges_raw) - len(valid_edges)} removed)")
    
    region_point_indices = {}
    region_edge_indices = {}
    for r in regions_data:
        center = np.array(r['center'])
        radius = r['radius']
        dists = cdist([center], first_coords).flatten()
        pt_indices = np.where(dists <= radius)[0]
        region_point_indices[r['title']] = pt_indices
        
        pt_set = set(pt_indices)
        edge_mask = []
        for edge in valid_edges:
            if edge[0] in pt_set or edge[1] in pt_set:
                edge_mask.append(True)
            else:
                edge_mask.append(False)
        region_edge_indices[r['title']] = np.where(edge_mask)[0]

    for fidx, frame_idx in enumerate(frame_indices):
        curr_coords = frames_data[frame_idx]['coords']
        if len(curr_coords) != len(first_coords):
            print(f"Warning: Frame {frame_idx} point count mismatch. Skipping.")
            continue
            
        all_disps_mm = np.linalg.norm(curr_coords - first_coords, axis=1) * UNIT_TO_MM
        all_stresses = vu.compute_edge_stress(curr_coords, valid_edges, initial_lengths, YOUNG_MODULUS_SILICON)
        
        for r in regions_data:
            r_title = r['title']
            key = (vessel_type, cond_label, r_title)
            
            pt_idx = region_point_indices[r_title]
            ed_idx = region_edge_indices[r_title]
            
            d_avg = np.mean(all_disps_mm[pt_idx]) if len(pt_idx) > 0 else 0.0
            d_med = np.median(all_disps_mm[pt_idx]) if len(pt_idx) > 0 else 0.0
            s_avg = np.mean(np.abs(all_stresses[ed_idx])) if len(ed_idx) > 0 else 0.0
            s_med = np.median(np.abs(all_stresses[ed_idx])) if len(ed_idx) > 0 else 0.0
            
            for m, val in [("Disp_Average", d_avg), ("Disp_Median", d_med), 
                           ("Stress_Average", s_avg), ("Stress_Median", s_med)]:
                if key not in results[m]: results[m][key] = []
                results[m][key].append(val)
        
        if fidx % 20 == 0: vu.print_progress_bar(fidx + 1, num_frames, bar_length=20)
    print()
    return num_frames

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # 1. Load Regions
    region_json_path = project_root / "cluster_data/vessel_regions.json"
    if not region_json_path.exists():
        print(f"Error: Region file not found at {region_json_path}")
        return

    regions_data = load_regions(region_json_path)
    print(f"Loaded {len(regions_data)} regions: {[r['title'] for r in regions_data]}")

    results = {
        "Disp_Average": {},
        "Disp_Median": {},
        "Stress_Average": {},
        "Stress_Median": {}
    }
    
    max_frames = 0

    # 2. Process GT
    for idx, cond_label in enumerate(CONDITIONS):
        if USE_RAW_GT:
            npz_path = GT_NPZ_FILES[idx]
            nf = process_raw_frames(npz_path, cond_label, "GT", regions_data, results)
        else:
            npy_rel_path = GT_NPY_FILES[idx]
            nf = process_clustered_frames(npy_rel_path, project_root, cond_label, "GT", regions_data, results)
        if nf > max_frames: max_frames = nf

    # 3. Process Clustered Ours (NPY)
    for idx, npy_rel_path in enumerate(OURS_NPY_FILES):
        cond_label = CONDITIONS[idx]
        nf = process_clustered_frames(npy_rel_path, project_root, cond_label, "Ours", regions_data, results)
        if nf > max_frames: max_frames = nf

    # 4. Write CSVs
    output_dir = project_root / "data_analysis"
    output_dir.mkdir(exist_ok=True)
    
    sorted_keys = []
    for v_type in ["GT", "Ours"]:
        for cond in CONDITIONS:
            for r in regions_data:
                sorted_keys.append((v_type, cond, r['title']))
    
    header = ["Frame"] + [f"{k[0]}_{k[1]}_{k[2]}" for k in sorted_keys]
    
    prefix = "raw_" if USE_RAW_GT else ""
    csv_configs = [
        ("Disp_Average", f"region_analysis_{prefix}disp_average.csv"),
        ("Disp_Median", f"region_analysis_{prefix}disp_median.csv"),
        ("Stress_Average", f"region_analysis_{prefix}stress_average.csv"),
        ("Stress_Median", f"region_analysis_{prefix}stress_median.csv"),
    ]
    
    for metric_key, filename in csv_configs:
        path = output_dir / filename
        print(f"Writing {path}...")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t in range(max_frames):
                row = [t]
                for key in sorted_keys:
                    series = results[metric_key].get(key, [])
                    val = series[t] if t < len(series) else ""
                    row.append(val)
                writer.writerow(row)

if __name__ == "__main__":
    main()
