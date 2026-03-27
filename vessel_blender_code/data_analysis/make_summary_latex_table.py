#!/usr/bin/env python3
"""
Generates a LaTeX summary table for synthetic validation outputs.
Compares ground truth (GT) and reconstruction (Ours) for:
- Bulk translation
- 1, 2, 3, 4, 5 mm pulling conditions

Metrics:
- Pull (mm)
- Chamfer Distance (mm) - Global, computed on the last frame
- Temporal Chamfer Distance (mm) - Global, frame-to-frame evolution (Delta CD)
- F-score - Global, aggregated across frames
- Mean displacement error (mm) - Average of regional peak mean displacement errors
- Mean stress proxy error (%) - Average of regional peak median stress-proxy percent errors

Outputs saved in data_analysis/synthetic_validation_summary/
"""

import sys
import json
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
FSCORE_MM = 1.0
FSCORE_TAU = FSCORE_MM / UNIT_TO_MM

# Young's Modulus for Stress calculation (Pa)
YOUNG_MODULUS_SILICON = vu.YOUNG_MODULUS_SILICON

# Output directory
OUTPUT_SUBDIR = Path("data_analysis/synthetic_validation_summary")

# Computation flags
SKIP_TEMPORAL_CD = False  # Set to True to skip the slow Temporal Chamfer Distance (Delta CD) computation
USE_RAW_GT = True

GT_BASE = Path.home() / "Documents/Blender/Vessel-Renders"

# Condition configurations
CONDITIONS = [
    {
        "label": "Bulk",
        "pull": 0,
        "gt": str(GT_BASE / "synthetic_translate/blender-data/mesh_data_single.npz") if USE_RAW_GT else "cluster_data/cluster_mesh_timeseries_GT_synthetic_translate.npy",
        "ours": "cluster_data/cluster_export_synthetic_translate.npy"
    },
    {
        "label": "1 mm",
        "pull": 1,
        "gt": str(GT_BASE / "lattice_strength/lattice_strength_0_25/blender-data/mesh_data_single.npz") if USE_RAW_GT else "cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_25.npy",
        "ours": "cluster_data/cluster_mesh_timeseries_lattice_strength_0_25.npy"
    },
    {
        "label": "2 mm",
        "pull": 2,
        "gt": str(GT_BASE / "lattice_strength/lattice_strength_0_50/blender-data/mesh_data_single.npz") if USE_RAW_GT else "cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_50.npy",
        "ours": "cluster_data/cluster_mesh_timeseries_lattice_strength_0_50.npy"
    },
    {
        "label": "3 mm",
        "pull": 3,
        "gt": str(GT_BASE / "lattice_strength/lattice_strength_0_75/blender-data/mesh_data_single.npz") if USE_RAW_GT else "cluster_data/cluster_mesh_timeseries_GT_lattice_strength_0_75.npy",
        "ours": "cluster_data/cluster_mesh_timeseries_lattice_strength_0_75.npy"
    },
    {
        "label": "4 mm",
        "pull": 4,
        "gt": str(GT_BASE / "lattice_strength/lattice_strength_1_00/blender-data/mesh_data_single.npz") if USE_RAW_GT else "cluster_data/cluster_mesh_timeseries_GT_lattice_strength_1_00.npy",
        "ours": "cluster_data/cluster_mesh_timeseries_lattice_strength_1_00.npy"
    },
    {
        "label": "5 mm",
        "pull": 5,
        "gt": str(GT_BASE / "lattice_strength/lattice_strength_1_25/blender-data/mesh_data_single.npz") if USE_RAW_GT else "cluster_data/cluster_mesh_timeseries_GT_lattice_strength_1_25.npy",
        "ours": "cluster_data/cluster_mesh_timeseries_lattice_strength_1_25.npy"
    }
]

# Tau sweep for supplementary metrics (in mm)
TAU_SWEEP_MM = [1.0, 2.0, 3.0]

# ============================================================================

def load_npy_data(file_path):
    """Load cluster data including edges and initial lengths, supporting .npz RAW mode."""
    file_path = Path(file_path)
    if file_path.suffix == '.npz':
        data = np.load(file_path)
        coords = data['coords']
        frame_numbers = data['frame_numbers']
        times = data['times']
        frames_data = vu.get_all_frames_data(coords, frame_numbers, times)
        
        frame_indices = sorted(frames_data.keys())
        T = len(frame_indices)
        if T == 0:
            return np.array([]), None, None
            
        N = len(frames_data[frame_indices[0]]['coords'])
        cluster_positions = np.zeros((T, N, 3))
        for i, f_idx in enumerate(frame_indices):
            cluster_positions[i] = frames_data[f_idx]['coords']
            
        _, edges_raw, _ = vu.create_delaunay_edges(cluster_positions[0])
        valid_edges_rel, _ = vu.remove_outlier_edges(cluster_positions[0], edges_raw, 1.8)
        init_lens = np.linalg.norm(
            cluster_positions[0][valid_edges_rel[:, 0]] - cluster_positions[0][valid_edges_rel[:, 1]], axis=1
        )
        abs_limit_internal = 1.0 / vu.UNIT_TO_MM
        abs_mask = init_lens <= abs_limit_internal
        edges = valid_edges_rel[abs_mask]
        initial_lengths = init_lens[abs_mask]
        
        return cluster_positions, edges, initial_lengths

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

def compute_chamfer_directed(P: np.ndarray, G: np.ndarray) -> Tuple[float, float]:
    """Compute directed Chamfer distance terms: P->G and G->P using cKDTree."""
    if len(P) == 0 or len(G) == 0:
        return float("inf"), float("inf")
    
    P_tree = cKDTree(P)
    G_tree = cKDTree(G)
    
    dist_p_to_g, _ = G_tree.query(P, k=1)
    dist_g_to_p, _ = P_tree.query(G, k=1)
    
    return float(np.mean(dist_p_to_g)), float(np.mean(dist_g_to_p))

def tcd_seq(sequence: np.ndarray) -> float:
    """Compute mean symmetric Chamfer distance (sum of directed terms) between subsequent frames."""
    T = sequence.shape[0]
    if T < 2:
        return 0.0
    
    cds = []
    for t in range(T - 1):
        p_to_g, g_to_p = compute_chamfer_directed(sequence[t], sequence[t+1])
        cds.append(p_to_g + g_to_p)
    return float(np.mean(cds))

def compute_delta_cd(gt_seq: np.ndarray, ours_seq: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute Temporal Chamfer Delta (ΔCD) in mm and its relative normalization factor.
    Returns: (tcd_gt_mm, tcd_ours_mm, delta_cd_mm, delta_cd_rel)
    """
    T = min(gt_seq.shape[0], ours_seq.shape[0])
    if T < 2: return 0.0, 0.0, 0.0, 0.0
    
    gt_temps = []
    ours_temps = []
    diffs = []
    
    for t in range(T - 1):
        gt_o2g, gt_g2o = compute_chamfer_directed(gt_seq[t], gt_seq[t+1])
        ours_o2g, ours_g2o = compute_chamfer_directed(ours_seq[t], ours_seq[t+1])
        
        cd_gt = gt_o2g + gt_g2o
        cd_ours = ours_o2g + ours_g2o
        
        gt_temps.append(cd_gt)
        ours_temps.append(cd_ours)
        diffs.append(abs(cd_ours - cd_gt))
            
    tcd_gt_mm = np.mean(gt_temps) * UNIT_TO_MM
    tcd_ours_mm = np.mean(ours_temps) * UNIT_TO_MM
    delta_cd_mm = np.mean(diffs) * UNIT_TO_MM
    median_gt_temporal_cd_mm = np.median(gt_temps) * UNIT_TO_MM
    
    if median_gt_temporal_cd_mm < 1e-8:
        delta_cd_rel = 0.0
    else:
        delta_cd_rel = delta_cd_mm / median_gt_temporal_cd_mm
        
    return tcd_gt_mm, tcd_ours_mm, delta_cd_mm, delta_cd_rel

def format_sci_latex(val: float, precision: int = 2) -> str:
    """Formats a number in LaTeX scientific notation: $base \times 10^{exp}$."""
    if abs(val) < 1e-10: return "0.00"
    s = f"{val:.{precision}e}"
    base, exp = s.split('e')
    return f"${base} \\times 10^{{{int(exp)}}}$"

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
        # Ensuring cd_sym = cd_o2g + cd_g2o
        gt_last = gt_pos[-1]
        ours_last = ours_pos[-1]
        cd_o2g, cd_g2o = compute_chamfer_directed(ours_last, gt_last)
        
        cd_o2g_mm = cd_o2g * UNIT_TO_MM
        cd_g2o_mm = cd_g2o * UNIT_TO_MM
        cd_sym_mm = cd_o2g_mm + cd_g2o_mm
        
        # 2. Temporal Chamfer Delta (ΔCD) in mm and Relative Normalization
        if SKIP_TEMPORAL_CD:
            tcd_gt_mm, tcd_ours_mm, delta_cd_mm, delta_cd_rel = 0.0, 0.0, 0.0, 0.0
        else:
            tcd_gt_mm, tcd_ours_mm, delta_cd_mm, delta_cd_rel = compute_delta_cd(gt_pos, ours_pos)
        
        # 3. Precision@tau (Ours->GT) for main table
        p_val_main, r_val_main, f_val_main = compute_precision_recall_fscore(ours_pos[-1], gt_pos[-1], FSCORE_TAU)
        
        # 4. Tau Sweep for Supplementary metrics (on last frame)
        tau_sweep_results = []
        for t_mm in TAU_SWEEP_MM:
            t_internal = t_mm / UNIT_TO_MM
            p, r, f = compute_precision_recall_fscore(ours_pos[-1], gt_pos[-1], t_internal)
            tau_sweep_results.append({"tau_mm": t_mm, "precision": p, "recall": r, "fscore": f})

        # --- REGIONAL METRICS ---
        region_disp_errors = []
        region_disp_pct_errors = []
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
            if peak_mean_gt > 1e-4:
                r_disp_err_pct = 100.0 * abs(peak_mean_ours - peak_mean_gt) / peak_mean_gt
            else:
                r_disp_err_pct = 0.0
            region_disp_pct_errors.append(r_disp_err_pct)
            
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

        mean_disp_err_mm = np.mean(region_disp_errors)
        mean_disp_err_pct = np.mean(region_disp_pct_errors)
        mean_stress_diff_pa = np.mean(region_stress_diffs)
        
        # --- Store results ---
        results.append({
            "label": cond["label"],
            "gt_spacing": gt_spacing_mm,
            "ours_spacing": ours_spacing_mm,
            "cd_o2g_mm": cd_o2g_mm,
            "cd_g2o_mm": cd_g2o_mm,
            "cd_sym_mm": cd_sym_mm,
            "tcd_gt_mm": tcd_gt_mm,
            "tcd_ours_mm": tcd_ours_mm,
            "delta_cd_mm": delta_cd_mm,
            "delta_cd_rel": delta_cd_rel,
            "precision_main": p_val_main,
            "recall_main": r_val_main,
            "fscore_main": f_val_main,
            "tau_sweep": tau_sweep_results,
            "disp_err": mean_disp_err_mm,
            "disp_err_pct": mean_disp_err_pct,
            "stress_diff_mpa": mean_stress_diff_pa / 1e6
        })

    # --- Save Spacing Log ---
    spacing_log_path = full_output_dir / "spacing_log.txt"
    with open(spacing_log_path, 'w') as f:
        f.write("Point Cloud Spacing Log (Median NN distance)\n")
        f.write("==============================================\n")
        f.write(f"{'Condition':<20} | {'GT (mm)':<10} | {'Ours (mm)':<10}\n")
        f.write("-" * 55 + "\n")
        for r in results:
            f.write(f"{r['label']:<20} | {r['gt_spacing']:<10.4f} | {r['ours_spacing']:<10.4f}\n")
    print(f"Spacing log saved to {spacing_log_path}")

    # ============================================================================
    # GENERATE MAIN PAPER TABLE
    # ============================================================================
    tau_mm = FSCORE_TAU * UNIT_TO_MM
    main_latex = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{siunitx}
\usepackage{graphicx}

\begin{document}
\thispagestyle{empty}

\begin{table}[h]
\centering
\caption{Main synthetic validation results across controlled vessel deformations. Geometric accuracy is reported via symmetric Chamfer Distance (CD, defined as the sum of Ours-to-Ground-truth and Ground-truth-to-Ours average terms) and $\mathrm{CD}_{\text{norm}}$ (CD normalized by median ground-truth point spacing). Temporal consistency is assessed by relative temporal CD ($\Delta\mathrm{CD}_{\text{rel}}$). Regional displacement error (absolute and percentage) and stress bias (Ours $-$ GT) are also reported. Precision is computed at $\tau=\SI{__TAU_MM__}{\milli\meter}$.}
\vspace{0.4cm}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Condition} & \multicolumn{2}{c}{\textbf{CD}} & \textbf{$\Delta\mathrm{CD}_{\text{rel}}$} & \textbf{Precision} & \multicolumn{2}{c}{\textbf{Displacement Error}} & \textbf{Stress Bias} \\
\cmidrule(lr){2-3} \cmidrule(lr){6-7}
& \textbf{(mm)} & \textbf{$\mathrm{CD}_{\text{norm}}$} & & & \textbf{(mm)} & \textbf{(\%)} & \textbf{(MPa)} \\
\midrule
"""
    main_latex = main_latex.replace("__TAU_MM__", f"{int(tau_mm)}")
    
    for r in results:
        cd_norm = r['cd_sym_mm'] / r['gt_spacing']
        dcd_rel_str = f"{r['delta_cd_rel']:.3f}" if not SKIP_TEMPORAL_CD else "--"
        
        row = (f"{r['label']} & {r['cd_sym_mm']:.3f} & {cd_norm:.3f} & {dcd_rel_str} & "
               f"{r['precision_main']:.3f} & {r['disp_err']:.3f} & {r['disp_err_pct']:.2f} & {r['stress_diff_mpa']:.3f} \\\\\n")
        main_latex += row
        
    main_latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
\end{document}
"""
    save_and_build(main_latex, full_output_dir, "synthetic_summary_main")

    # ============================================================================
    # GENERATE SUPPLEMENTARY TABLE
    # ============================================================================
    supp_latex = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{a4paper, margin=0.5in}
\usepackage{siunitx}
\usepackage{graphicx}

\begin{document}
\thispagestyle{empty}

\begin{table}[h]
\centering
\caption{Supplementary synthetic validation details. We report directed Chamfer Distance (CD) components: Ours-to-Ground-truth (O2G) and Ground-truth-to-Ours (G2O), along with the symmetric sum (Sym = O2G + G2O) and normalized value ($\mathrm{CD}_{\text{norm}}$). Absolute (mm) and relative (rel) temporal CD ($\Delta\mathrm{CD}$) are shown. Point-cloud overlap metrics (Precision P, Recall R, F-score F) are reported across a sweep of proximity thresholds $\tau$. Ground-truth comparison metrics are reported at $\tau = \SI{1}{\milli\meter}$ by default.}
\vspace{0.4cm}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
\textbf{Condition} & \multicolumn{4}{c}{\textbf{CD}} & \multicolumn{2}{c}{\textbf{$\Delta\mathrm{CD}$}} & \textbf{P} & \textbf{R} & \textbf{F} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-7}
& \textbf{O2G (mm)} & \textbf{G2O (mm)} & \textbf{Sym (mm)} & \textbf{$\mathrm{CD}_{\text{norm}}$} & \textbf{mm} & \textbf{rel} & & & \\
\midrule
"""
    for r in results:
        # cd_norm for supp
        cd_norm = r['cd_sym_mm'] / r['gt_spacing']
        
        # Base row for standard tau
        dcd_mm_str = format_sci_latex(r['delta_cd_mm']) if not SKIP_TEMPORAL_CD else "--"
        dcd_rel_str = f"{r['delta_cd_rel']:.3f}" if not SKIP_TEMPORAL_CD else "--"
        
        # Row for main tau
        row_main = (f"{r['label']} & {r['cd_o2g_mm']:.3f} & {r['cd_g2o_mm']:.3f} & {r['cd_sym_mm']:.3f} & {cd_norm:.3f} & "
                    f"{dcd_mm_str} & {dcd_rel_str} & "
                    f"{r['precision_main']:.3f} & {r['recall_main']:.3f} & {r['fscore_main']:.3f} \\\\\n")
        supp_latex += row_main
        
        # Additional rows for tau sweep
        for s in r['tau_sweep']:
            # Skip if tau_mm is close to base FSCORE_MM
            if abs(s['tau_mm'] - FSCORE_MM) < 1e-4:
                continue
            row_sweep = (f"  \\textit{{$\\tau = \\SI{{{int(s['tau_mm'])}}}{{mm}}$}} & & & & & & & "
                         f"{s['precision']:.3f} & {s['recall']:.3f} & {s['fscore']:.3f} \\\\\n")
            supp_latex += row_sweep
        if r != results[-1]:
            supp_latex += r"\midrule" + "\n"
        
    supp_latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
\end{document}
"""
    save_and_build(supp_latex, full_output_dir, "synthetic_summary_supp")

def save_and_build(latex_content, output_dir, name):
    tex_path = output_dir / f"{name}.tex"
    pdf_path = output_dir / f"{name}.pdf"
    
    with open(tex_path, 'w') as f:
        f.write(latex_content)
    print(f"LaTeX file saved to {tex_path}")
    
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=output_dir, check=True, capture_output=True)
        print(f"PDF generated at {pdf_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during {name} generation:")
        print(e.stdout.decode())
        print(e.stderr.decode())

if __name__ == "__main__":
    main()
