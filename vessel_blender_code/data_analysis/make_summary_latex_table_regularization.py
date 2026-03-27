#!/usr/bin/env python3
"""
Generates a LaTeX summary table for synthetic validation outputs.
Compares ground truth (GT) and reconstruction (Ours) for:
- Bulk translation
- 1, 2, 3, 4, 5 mm pulling conditions

Metrics:
- Pull (mm)
- Chamfer Distance (mm) - Global, computed on the last frame 
- Mean displacement error (mm) - Average of regional peak mean displacement errors
- Mean stress proxy error (%) - Average of regional peak median stress-proxy percent errors

Outputs saved in data_analysis/synthetic_regularization_summary/
"""

import sys
import os
import json
import csv
import numpy as np
import subprocess
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from typing import Tuple, List, Dict

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import vessel_utils as vu
from utils.distance_utils import compute_chamfer_distance

# ============================================================================
# CONFIGURATION
# ============================================================================

# Physical constants for scaling (Blender Units to mm)
CHESS_TOTAL_LENGTH_M = 6.195 / 100.0
CHESS_SQUARES_ALONG_X = 8
CHESS_SQUARE_LENGTH_M = CHESS_TOTAL_LENGTH_M / CHESS_SQUARES_ALONG_X
CALIB_SCALING_FACTOR = 0.667125
CORRECTION_FACTOR = 6.195 / 5.053
UNIT_TO_MM = (CHESS_SQUARE_LENGTH_M / CALIB_SCALING_FACTOR) * 1000.0 * CORRECTION_FACTOR

# F-score threshold (tau) for metric computation (in Blender units)
FSCORE_TAU = 0.07 

# Young's Modulus for Stress calculation (Pa)
YOUNG_MODULUS_SILICON = vu.YOUNG_MODULUS_SILICON

# Static Paths
GT_FILE = "cluster_data/cluster_mesh_timeseries_GT_lattice_strength_1_25.npy"
BASELINE_FILE = "cluster_data/cluster_mesh_timeseries_lattice_strength_1_25.npy"
BUNDLE_DIR = Path("cluster_data/lattice_strength_1_25")

# Output directory
OUTPUT_BASE_DIR = Path("data_analysis/synthetic_regularization_summary")
OUTPUT_SUBDIR = OUTPUT_BASE_DIR / f"{BUNDLE_DIR.name}_regularization_summary"

# Computation flags
SKIP_TEMPORAL_CD = False  # Set to True to skip the slow Temporal Chamfer Distance (Delta CD) computation

# Target RGD values (in mm)
TARGET_PULLS = [0.0, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0, 1000.0]

def discover_conditions():
    """Scan bundle directory and build conditions list dynamically."""
    import re
    
    full_bundle_dir = project_root / BUNDLE_DIR
    
    conditions = []
    
    if not full_bundle_dir.exists():
        print(f"Warning: Bundle directory {full_bundle_dir} not found.")
        return conditions

    # 2. Add files from bundle
    # Match rgd, spd, phy values. Use specific pattern to be more robust.
    pattern = re.compile(r"rgd([\d\.]+)(?:_spd([\d\.]+))?(?:_phy([\d\.]+)?)?")
    
    found_any = False
    for f in full_bundle_dir.glob("*.npy"):
        match = pattern.search(f.stem)
        if match:
            try:
                # Extract values, default to 0.0 if not present
                rgd_str = match.group(1)
                spd_str = match.group(2) if match.group(2) else "0.0"
                phy_str = match.group(3) if match.group(3) else "0.0"
                
                rgd = float(rgd_str)
                spd = float(spd_str)
                phy = float(phy_str)
                
                # Filter for specific rgd values
                if rgd not in TARGET_PULLS:
                    continue
                    
                conditions.append({
                    "label": f.stem,
                    "params": [rgd, spd, phy],
                    "gt": GT_FILE,
                    "ours": str(f.relative_to(project_root)),
                    "pull": 0
                })
                found_any = True
            except ValueError:
                continue
    
    if not found_any:
        print(f"Warning: No valid .npy bundle files matching pattern found in {full_bundle_dir}")
            
    # Sort by params for a cleaner table
    conditions = sorted(conditions, key=lambda x: x["params"])
    print(f"Discovered {len(conditions)} conditions.")
    return conditions

CONDITIONS = discover_conditions()

# ============================================================================

def load_npy_data(file_path):
    """Load cluster data including edges and initial lengths."""
    data = np.load(str(file_path), allow_pickle=True).item()
    cluster_positions = data.get("cluster_positions")
    edges = data.get("edges")
    initial_lengths = data.get("initial_lengths")
    return cluster_positions, edges, initial_lengths

def load_regions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def random_downsample(points: np.ndarray, target_count: int, random_seed: int = None) -> np.ndarray:
    """Randomly downsample a point cloud to a target number of points."""
    if len(points) == 0 or target_count >= len(points):
        return points
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    indices = rng.choice(len(points), size=target_count, replace=False)
    return points[indices]

def compute_frame_to_frame_chamfer(sequence_positions: np.ndarray, t: int) -> float:
    """Compute symmetric Chamfer distance between frame t and t+1."""
    if t < 0 or t >= sequence_positions.shape[0] - 1:
        return float("inf")
    frame_t = sequence_positions[t]
    frame_t_plus_1 = sequence_positions[t + 1]
    if len(frame_t) == 0 or len(frame_t_plus_1) == 0:
        return float("inf")
    res = compute_chamfer_distance(frame_t, frame_t_plus_1)
    return res[0]

def compute_delta_cd(gt_seq: np.ndarray, ours_seq: np.ndarray) -> float:
    """Compute Temporal Chamfer Delta (ΔCD) in mm."""
    T = min(gt_seq.shape[0], ours_seq.shape[0])
    if T < 2: return 0.0
    diffs = []
    for t in range(T - 1):
        cd_gt = compute_frame_to_frame_chamfer(gt_seq, t)
        cd_ours = compute_frame_to_frame_chamfer(ours_seq, t)
        if np.isfinite(cd_gt) and np.isfinite(cd_ours):
            diffs.append(abs(cd_ours - cd_gt))
    return np.mean(diffs) * UNIT_TO_MM if diffs else 0.0

def compute_precision_recall_fscore(P: np.ndarray, G: np.ndarray, tau: float) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F-score."""
    if len(P) == 0 or len(G) == 0: return 0.0, 0.0, 0.0
    G_tree = cKDTree(G)
    P_tree = cKDTree(P)
    P_to_G_dist, _ = G_tree.query(P, k=1)
    G_to_P_dist, _ = P_tree.query(G, k=1)
    precision = np.mean(P_to_G_dist < tau)
    recall = np.mean(G_to_P_dist < tau)
    fscore = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, fscore

def get_region_indices(pos_0, regions_data, edges=None):
    """Get point and edge indices for each region."""
    region_point_indices = {}
    region_edge_indices = {}
    
    for r in regions_data:
        center = np.array(r['center'])
        radius = r['radius']
        dists = cdist([center], pos_0).flatten()
        pt_mask = dists <= radius
        pt_indices = np.where(pt_mask)[0]
        region_point_indices[r['title']] = pt_indices
        
        if edges is not None:
            pt_set = set(pt_indices)
            edge_mask = []
            for edge in edges:
                if edge[0] in pt_set or edge[1] in pt_set:
                    edge_mask.append(True)
                else:
                    edge_mask.append(False)
            region_edge_indices[r['title']] = np.where(edge_mask)[0]
        else:
            region_edge_indices[r['title']] = np.array([])
            
    return region_point_indices, region_edge_indices

def main():
    print("Starting Synthetic Validation Summary Table Generation...")
    
    # Setup Output
    full_output_dir = project_root / OUTPUT_SUBDIR
    full_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Regions
    region_json_path = project_root / "cluster_data/vessel_regions.json"
    if not region_json_path.exists():
        print(f"Error: Region file not found at {region_json_path}")
        return
    regions_data = load_regions(region_json_path)
    print(f"Loaded {len(regions_data)} regions.")

    results = []

    for cond in CONDITIONS:
        gt_path = project_root / cond["gt"]
        ours_path = project_root / cond["ours"]
        
        if not gt_path.exists() or not ours_path.exists():
            print(f"Skipping {cond['label']}: File not found.")
            continue
            
        print(f"Processing {cond['label']}...")
        
        # Load Data
        gt_pos, gt_edges, gt_init_lens = load_npy_data(gt_path)
        ours_pos, ours_edges, ours_init_lens = load_npy_data(ours_path)
        
        # Calculate Internal Spacing (Median NN)
        gt_tree = cKDTree(gt_pos[0]) # Use frame 0 for spacing
        dist_gt, _ = gt_tree.query(gt_pos[0], k=2)
        gt_spacing_mm = np.median(dist_gt[:, 1]) * UNIT_TO_MM

        ours_tree = cKDTree(ours_pos[0])
        dist_ours, _ = ours_tree.query(ours_pos[0], k=2)
        ours_spacing_mm = np.median(dist_ours[:, 1]) * UNIT_TO_MM
        
        T = min(gt_pos.shape[0], ours_pos.shape[0])
        
        # Identify Regions Indices at Frame 0 for both GT and Ours
        gt_pt_idx, gt_ed_idx = get_region_indices(gt_pos[0], regions_data, gt_edges)
        ours_pt_idx, ours_ed_idx = get_region_indices(ours_pos[0], regions_data, ours_edges)
        
        # --- GLOBAL METRICS ---
        # 1. Chamfer Distance (Last Frame) in mm
        gt_last = gt_pos[-1]
        ours_last = ours_pos[-1]
        min_pts = min(len(gt_last), len(ours_last))
        gt_down = random_downsample(gt_last, min_pts, random_seed=42)
        ours_down = random_downsample(ours_last, min_pts, random_seed=43)
        chamfer_mm = compute_chamfer_distance(gt_down, ours_down)[0] * UNIT_TO_MM
        
        # 2. Temporal Chamfer Distance (Delta CD) in mm
        if SKIP_TEMPORAL_CD:
            delta_cd_mm = 0.0
        else:
            delta_cd_mm = compute_delta_cd(gt_pos, ours_pos)
        
        # 3. F-score (Mean across all frames)
        fscore_vals = [compute_precision_recall_fscore(ours_pos[t], gt_pos[t], FSCORE_TAU)[2] for t in range(T)]
        mean_fscore = np.mean(fscore_vals)
        
        # --- REGIONAL METRICS ---
        region_disp_errors = []
        region_stress_errors = []
        region_stress_diffs = []
        
        for r in regions_data:
            r_title = r['title']
            
            # Regional Peak Mean Displacement
            def get_peak_mean_disp(pos_seq, pt_indices):
                if len(pt_indices) == 0: return 0.0
                # Mean displacement per frame
                means = []
                init_pos = pos_seq[0][pt_indices]
                for t in range(T):
                    curr_pos = pos_seq[t][pt_indices]
                    disps = np.linalg.norm(curr_pos - init_pos, axis=1)
                    means.append(np.mean(disps))
                return np.max(means) * UNIT_TO_MM
            
            peak_mean_gt = get_peak_mean_disp(gt_pos, gt_pt_idx[r_title])
            peak_mean_ours = get_peak_mean_disp(ours_pos, ours_pt_idx[r_title])
            region_disp_errors.append(abs(peak_mean_ours - peak_mean_gt))
            
            # Debug print for displacement
            print(f"    - Region: {r_title:20} | GT Max Mean Disp: {peak_mean_gt:7.4f} mm | Ours Max Mean Disp: {peak_mean_ours:7.4f} mm | Error: {abs(peak_mean_ours - peak_mean_gt):7.4f} mm")
            
            # Regional Peak Median Stress
            def get_peak_median_stress(pos_seq, edges, init_lens, ed_indices):
                if len(ed_indices) == 0: return 0.0
                medians = []
                for t in range(T):
                    stresses = vu.compute_edge_stress(pos_seq[t], edges, init_lens, YOUNG_MODULUS_SILICON)
                    medians.append(np.median(np.abs(stresses[ed_indices])))
                return np.max(medians)

            peak_med_gt = get_peak_median_stress(gt_pos, gt_edges, gt_init_lens, gt_ed_idx[r_title])
            peak_med_ours = get_peak_median_stress(ours_pos, ours_edges, ours_init_lens, ours_ed_idx[r_title])
            
            diff = peak_med_ours - peak_med_gt
            region_stress_diffs.append(diff)
            # Avoid division by zero/noise if GT stress is near 0 (translation case)
            if peak_med_gt > 1.0: 
                region_stress_errors.append(100.0 * abs(diff) / peak_med_gt)
            else:
                region_stress_errors.append(0.0)

        mean_disp_err_mm = np.mean(region_disp_errors)
        mean_stress_err_pct = np.mean(region_stress_errors)
        mean_stress_diff_pa = np.mean(region_stress_diffs)
        
        results.append({
            "label": cond["label"],
            "params": cond.get("params", [0, 0, 0]),
            "gt_spacing": gt_spacing_mm,
            "ours_spacing": ours_spacing_mm,
            "chamfer": chamfer_mm,
            "delta_cd": delta_cd_mm,
            "fscore": mean_fscore,
            "disp_err": mean_disp_err_mm,
            "stress_err": mean_stress_err_pct,
            "stress_diff_mpa": mean_stress_diff_pa / 1e6
        })

    # Add relative improvements vs Baseline (first entry)
    baseline = results[0]
    for r in results:
        r["chamfer_rel"] = 100.0 * (r["chamfer"] - baseline["chamfer"]) / baseline["chamfer"]
        r["disp_err_rel"] = 100.0 * (r["disp_err"] - baseline["disp_err"]) / baseline["disp_err"] if baseline["disp_err"] > 1e-6 else 0.0
        r["stress_bias_rel"] = 100.0 * (r["stress_diff_mpa"] - baseline["stress_diff_mpa"]) / baseline["stress_diff_mpa"] if abs(baseline["stress_diff_mpa"]) > 1e-6 else 0.0

    # Determine best values for highlighting
    best_vals = {
        "chamfer": min(r["chamfer"] for r in results),
        "fscore": max(r["fscore"] for r in results),
        "disp_err": min(r["disp_err"] for r in results),
        "delta_cd": min(r["delta_cd"] for r in results if r["delta_cd"] > 0) or 0,
        "stress_diff_mpa": min(abs(r["stress_diff_mpa"]) for r in results)
    }

    # Save CSV
    csv_path = full_output_dir / "regularization_study_results.csv"
    save_results_csv(results, csv_path)
    
    # Perform Sensitivity Analysis
    analyze_hyperparameter_sensitivity(results)
    
    # --- Save Spacing Log ---
    spacing_log_path = full_output_dir / "spacing_log.txt"
    with open(spacing_log_path, 'w') as f:
        f.write("Point Cloud Spacing Log (Median NN distance)\n")
        f.write("==============================================\n")
        f.write(f"{'Condition':<15} | {'GT (mm)':<10} | {'Ours (mm)':<10}\n")
        f.write("-" * 45 + "\n")
        for r in results:
            f.write(f"{r['label']:<15} | {r['gt_spacing']:<10.4f} | {r['ours_spacing']:<10.4f}\n")
    print(f"Spacing log saved to {spacing_log_path}")

    # ============================================================================
    # GENERATE LATEX TABLE
    # ============================================================================
    
    latex_content = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{a4paper, margin=0.5in, landscape}
\usepackage{siunitx}
\usepackage{graphicx}
\usepackage[table]{xcolor}

\definecolor{lightgray}{gray}{0.95}
\rowcolors{2}{}{lightgray}

\begin{document}
\thispagestyle{empty}

\begin{table}[h]
\centering
\caption{Hyperparameter Regularization Study. Comparison of different regularization weights $\lambda_{rigid}$ for bulk translation. Relative change (\%) is reported compared to the "No Hyperparams" baseline. Best values in each column (excluding parameters) are highlighted in \textbf{bold}.}
\vspace{0.4cm}
\label{tab:regularization_study}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
\textbf{Param} & \multicolumn{3}{c}{\textbf{Chamfer Distance (CD)}} & \textbf{$\Delta$CD} & \textbf{F-score} & \multicolumn{2}{c}{\textbf{Disp. Error}} & \multicolumn{2}{c}{\textbf{Stress Bias}} \\
\cmidrule(lr){1-1} \cmidrule(lr){2-4} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
$\lambda_{rigid}$ & (mm) & Norm. & $\Delta\%$ & (mm) & & (mm) & $\Delta\%$ & (MPa) & $\Delta\%$ \\
\midrule
"""
    for idx, r in enumerate(results):
        # Normalization
        cd_norm = r['chamfer'] / r['gt_spacing']
        
        # Format Delta CD based on flag
        if SKIP_TEMPORAL_CD:
            dcd_str = "--"
        else:
            dcd_str = f"{r['delta_cd']:.2e}"
            if r['delta_cd'] == best_vals["delta_cd"] and idx > 0 and r['delta_cd'] > 1e-10:
                dcd_str = rf"\textbf{{{dcd_str}}}"
        
        # Highlighting logic helpers
        def fmt(val, key, sig=3, is_abs=False):
            s = f"{val:.{sig}f}"
            v_check = abs(val) if is_abs else val
            # Only bold if it's the best and not zero
            if v_check == best_vals[key] and idx > 0 and v_check > 1e-10:
                return rf"\textbf{{{s}}}"
            return s

        l_rigid = r["params"][0]
        
        # Format relative improvements
        c_rel = f"{r['chamfer_rel']:+1.1f}\\%" if idx > 0 else "--"
        d_rel = f"{r['disp_err_rel']:+1.1f}\\%" if idx > 0 else "--"
        s_rel = f"{r['stress_bias_rel']:+1.1f}\\%" if idx > 0 else "--"
        
        row = (
            f"{l_rigid} & "
            f"{fmt(r['chamfer'], 'chamfer')} & {cd_norm:.3f} & {c_rel} & "
            f"{dcd_str} & {fmt(r['fscore'], 'fscore')} & "
            f"{fmt(r['disp_err'], 'disp_err')} & {d_rel} & "
            f"{fmt(r['stress_diff_mpa'], 'stress_diff_mpa', 3, is_abs=True)} & {s_rel} \\\\\n"
        )
        latex_content += row
        
    latex_content += r"""\bottomrule
\end{tabular}%
}
\end{table}

\end{document}
"""
    
    tex_path = full_output_dir / "regularization_summary_table.tex"
    pdf_path = full_output_dir / "regularization_summary_table.pdf"
    png_path = full_output_dir / "regularization_summary_table.png"
    
    with open(tex_path, 'w') as f:
        f.write(latex_content)
    print(f"LaTeX file saved to {tex_path}")
    
    # Run pdflatex
    try:
        # Note: we need to run in the output directory or pass full path
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=full_output_dir, check=True, capture_output=True)
        print(f"PDF generated at {pdf_path}")
        
        # Convert to PNG using sips (Mac)
        subprocess.run(["sips", "-s", "format", "png", pdf_path.name, "--out", png_path.name], cwd=full_output_dir, check=True, capture_output=True)
        print(f"PNG generated at {png_path}")
        
    except subprocess.CalledProcessError as e:
        print("Error during PDF/PNG generation:")
        print(e.stdout.decode())
        print(e.stderr.decode())

def save_results_csv(results, output_path):
    """Saves all results to a CSV file."""
    if not results: return
    keys = results[0].keys()
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV results saved to {output_path}")

def analyze_hyperparameter_sensitivity(results):
    """
    Analyzes which parameters are affecting metrics the most and suggests directions.
    """
    if len(results) < 2: return
    
    baseline = results[0] # Assumes first is baseline [0,0,0]
    
    # Extract parameter values
    rigid_vals = sorted(list(set(r["params"][0] for r in results)))
    speed_vals = sorted(list(set(r["params"][1] for r in results)))
    phys_vals = sorted(list(set(r["params"][2] for r in results)))
    
    print("\n" + "="*50)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*50)
    
    # Metrics to track
    metrics = ["chamfer", "disp_err"]
    labels = ["CD (Chamfer)", "Displacement Error"]
    
    recommendations = []
    
    for m_idx, metric in enumerate(metrics):
        print(f"\nMetric: {labels[m_idx]}")
        
        # Calculate impact for each parameter
        impacts = {}
        for p_idx, p_name in enumerate(["lambda_rigid", "lambda_speed", "lambda_phys"]):
            # Group by this parameter
            by_val = {}
            for r in results:
                v = r["params"][p_idx]
                if v not in by_val: by_val[v] = []
                by_val[v].append(r[metric])
            
            # Simple trend: Average metric at each value
            means = {v: np.mean(vals) for v, vals in by_val.items()}
            sorted_v = sorted(means.keys())
            
            if len(sorted_v) > 1:
                slope = (means[sorted_v[-1]] - means[sorted_v[0]]) / (sorted_v[-1] - sorted_v[0] + 1e-9)
                impacts[p_name] = slope
                direction = "Decreasing" if slope < 0 else "Increasing"
                print(f"  - {p_name:12}: {direction} impact (Slope: {slope:.4f})")
        
        # Suggestion based on strongest negative slope (best improvement)
        if impacts:
            best_param = min(impacts, key=impacts.get)
            if impacts[best_param] < 0:
                recommendations.append(f"To improve {labels[m_idx]}, try INCREASING {best_param}.")
            else:
                best_to_down = max(impacts, key=impacts.get)
                if impacts[best_to_down] > 0:
                    recommendations.append(f"To improve {labels[m_idx]}, try DECREASING {best_to_down}.")

    print("\n" + "="*50)
    print("TUNING RECOMMENDATIONS")
    print("="*50)
    for rec in recommendations:
        print(f"🚀 {rec}")
    
    # Final directional summary
    print("\nQuick Summary:")
    print("- lambda_rigid: Heavy influence on geometric noise but can over-smooth if too high.")
    print("- lambda_speed: affects temporal stability (look at Delta CD).")
    print("- lambda_phys: enforces volume/edge constraints; essential for stress bias accuracy.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()