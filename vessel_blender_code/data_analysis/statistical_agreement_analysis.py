#!/usr/bin/env python3
"""
Performs statistical agreement analysis between Ground Truth (GT) and Ours data.
Calculates error metrics, correlations, and generates comparison plots.
Supports toggling between Clustered GT and RAW GT.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_RAW_GT = True

# ============================================================================
# STYLING & LAYOUT
# ============================================================================
# Font sizes
FS_PANEL_LABEL = 40
FS_AXIS_LABEL = 24
FS_TICK_LABEL = 22
FS_LEGEND = 24
FS_ANNOT = 24
FS_BA_ANNOT = 22

# Colors
COLOR_OBSERVATIONS = '#4a86e8'  # Professional Blue
COLOR_FIT_LINE = '#e06666'      # Subtle Red
COLOR_DIFF_POINTS = '#6aa84f'   # Professional Green
COLOR_BIAS_LINE = '#cc0000'     # Darker Red
COLOR_LOA_BAND = '#e67e22'      # Orange
COLOR_IDENTITY = '#333333'      # Dark Gray

# Dimensions
FIG_SIZE_SINGLE = (18, 8)
FIG_SIZE_COMBINED = (18, 14.5)
MARKER_SIZE = 70
ALPHA = 0.7
PANEL_LABEL_Y = 1.02            # Vertical position of A, B labels
PANEL_LABEL_X = -0.1            # Horizontal position of A, B labels

def parse_summary_csv(csv_path):
    """
    Parses the summary CSV and extracts paired GT vs Ours data.
    """
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return None

    results = {}
    current_metric = None
    conditions = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            
            # Start of a new metric block
            upper_row0 = row[0].upper()
            if "MAX OF" in upper_row0:
                title = row[0]
                # Determine metric category (Max Mean/Max Median) and data type (Disp/Stress)
                cat = "Max Mean" if ("AVERAGE" in upper_row0 or "MEAN" in upper_row0) else "Max Median"
                dtype = "Displacement" if "DISPLACEMENT" in upper_row0 else "Stress"
                unit = "mm" if dtype == "Displacement" else "Pa"
                
                current_metric = f"{cat} {dtype}"
                results[current_metric] = {"GT": [], "Ours": [], "Labels": [], "Unit": unit}
                conditions = [] # Reset conditions for new block
                continue
            
            # Header row for the conditions
            if row[0].strip().lower() in ["", "condition"] and len(row) > 1:
                conditions = row[1:]
                continue
            
            # Data row handling for interleaved format
            if current_metric:
                first_col = row[0].strip()
                v_type = None
                region_name = None
                
                if first_col.startswith("GT "):
                    v_type = "GT"
                    region_name = first_col[3:]
                elif first_col.startswith("Ours "):
                    v_type = "Ours"
                    region_name = first_col[5:]
                
                if v_type and region_name:
                    vals = [float(v) for v in row[1:] if v]
                    results[current_metric][v_type].extend(vals)
                    
                    if v_type == "GT":
                        if not conditions:
                            for _ in range(len(vals)):
                                results[current_metric]["Labels"].append(region_name)
                        else:
                            for cond in conditions:
                                results[current_metric]["Labels"].append(f"{region_name}_{cond}")
                        
    return results

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mask = y_true != 0
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0
    mean_diff = np.mean(y_pred - y_true)
    std_diff = np.std(y_pred - y_true)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    return {
        "slope": slope, "intercept": intercept, "r_squared": r_value**2,
        "mae": mae, "rmse": rmse, "mape": mape, "bias": mean_diff,
        "loa_upper": loa_upper, "loa_lower": loa_lower
    }

def add_panel_label(ax, label):
    ax.text(PANEL_LABEL_X, PANEL_LABEL_Y, label, transform=ax.transAxes, fontsize=FS_PANEL_LABEL,
            fontweight=1000, fontname='DejaVu Sans', va='bottom', ha='left', zorder=100)

def annotate_ba_line(ax, y, text, color, unit, va='bottom', precision=2):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_span = ylim[1] - ylim[0]
    x_pos = xlim[1] - 0.02 * (xlim[1] - xlim[0])
    offset = 0.015 * y_span if va == 'bottom' else -0.015 * y_span
    ax.text(x_pos, y + offset, f"{text} = {y:.{precision}f} {unit}", 
            va=va, ha='right', color=color, fontweight='bold', fontsize=FS_BA_ANNOT)

def draw_agreement_pair(ax1, ax2, metric_name, unit, gt_vals, ours_vals, metrics, labels):
    gt_vals = np.array(gt_vals)
    ours_vals = np.array(ours_vals)
    display_unit = unit
    display_gt = gt_vals
    display_ours = ours_vals
    display_metrics = metrics.copy()
    
    if unit == "Pa" and np.max(gt_vals) > 1000:
        display_unit = "MPa"
        display_gt = gt_vals / 1e6
        display_ours = ours_vals / 1e6
        for k in ['mae', 'rmse', 'bias', 'loa_upper', 'loa_lower', 'intercept']:
            display_metrics[k] /= 1e6

    marker_style = dict(s=MARKER_SIZE, alpha=ALPHA, edgecolors='white', linewidth=0.5)

    # Correlation
    all_vals = np.concatenate([display_gt, display_ours])
    v_min, v_max = all_vals.min(), all_vals.max()
    pad = (v_max - v_min) * 0.1
    lims = [v_min - pad, v_max + pad]
    
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.scatter(display_gt, display_ours, color=COLOR_OBSERVATIONS, **marker_style, label='Observations')
    ax1.plot(lims, lims, color=COLOR_IDENTITY, linestyle='--', alpha=0.5, linewidth=1.5, label='Identity')
    x_reg = np.array(lims)
    y_reg = display_metrics['slope'] * x_reg + display_metrics['intercept']
    ax1.plot(x_reg, y_reg, color=COLOR_FIT_LINE, linestyle='-', linewidth=2.5, label='Linear Fit')
    
    gt_prefix = "RAW GT" if USE_RAW_GT else "Ground Truth"
    ax1.set_xlabel(f'{gt_prefix} ({display_unit})', fontsize=FS_AXIS_LABEL)
    ax1.set_ylabel(f'Ours ({display_unit})', fontsize=FS_AXIS_LABEL)
    ax1.legend(frameon=True, loc='lower right', fontsize=FS_LEGEND)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    textstr = '\n'.join((
        f'Slope: {display_metrics["slope"]:.3f}',
        f'Intercept: {display_metrics["intercept"]:.3f}',
        f'R²: {display_metrics["r_squared"]:.3f}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=FS_ANNOT, va='top', bbox=props)

    # Bland-Altman
    averages = (display_gt + display_ours) / 2
    diffs = display_ours - display_gt
    ax2.scatter(averages, diffs, color=COLOR_DIFF_POINTS, **marker_style)
    ax2.axhline(0, color=COLOR_IDENTITY, linestyle='-', alpha=0.3)
    ax2.axhline(display_metrics['bias'], color=COLOR_BIAS_LINE, linewidth=2)
    
    curr_xlim = [min(averages) - 0.1*(max(averages)-min(averages)), max(averages) + 0.1*(max(averages)-min(averages))]
    ax2.set_xlim(curr_xlim)
    
    # Draw LoA band and lines
    ax2.fill_between(curr_xlim, display_metrics['loa_lower'], display_metrics['loa_upper'], color=COLOR_LOA_BAND, alpha=0.08)
    ax2.axhline(display_metrics['loa_upper'], color=COLOR_LOA_BAND, linestyle='--', alpha=0.6)
    ax2.axhline(display_metrics['loa_lower'], color=COLOR_LOA_BAND, linestyle='--', alpha=0.6)
    
    # Calculate symmetric Y limits around the bias to center the band
    max_diff_val = np.max(diffs) if len(diffs) > 0 else display_metrics['loa_upper']
    min_diff_val = np.min(diffs) if len(diffs) > 0 else display_metrics['loa_lower']
    
    max_dist = max(
        abs(display_metrics['loa_upper'] - display_metrics['bias']),
        abs(display_metrics['loa_lower'] - display_metrics['bias']),
        abs(max_diff_val - display_metrics['bias']),
        abs(min_diff_val - display_metrics['bias'])
    )
    # Applying 30% padding to make the band more prominent
    y_half_range = max_dist * 1.3
    ax2.set_ylim(display_metrics['bias'] - y_half_range, display_metrics['bias'] + y_half_range)
    
    # Unified precision for both metric types as per request (Disp: 2->3, Stress: 4->3)
    precision = 3
    annotate_ba_line(ax2, display_metrics['loa_upper'], "Upper LoA", COLOR_LOA_BAND, display_unit, va='bottom', precision=precision)
    annotate_ba_line(ax2, display_metrics['bias'], "Mean Bias", COLOR_BIAS_LINE, display_unit, va='bottom', precision=precision)
    annotate_ba_line(ax2, display_metrics['loa_lower'], "Lower LoA", COLOR_LOA_BAND, display_unit, va='top', precision=precision)

    ax2.set_xlabel(f'Mean ({gt_prefix} & Ours) ({display_unit})', fontsize=FS_AXIS_LABEL)
    ax2.set_ylabel(f'Difference (Ours - {gt_prefix}) ({display_unit})', fontsize=FS_AXIS_LABEL)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    add_panel_label(ax1, labels[0])
    add_panel_label(ax2, labels[1])

def plot_combined_analysis(cat_name, disp_data, stress_data, output_path):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': FS_TICK_LABEL, 'axes.labelweight': 'bold'
    })
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_COMBINED, constrained_layout=True)
    if disp_data:
        draw_agreement_pair(axes[0,0], axes[0,1], disp_data[0], disp_data[4], disp_data[1], disp_data[2], disp_data[3], ["A", "B"])
    if stress_data:
        draw_agreement_pair(axes[1,0], axes[1,1], stress_data[0], stress_data[4], stress_data[1], stress_data[2], stress_data[3], ["C", "D"])
    plt.savefig(output_path.with_suffix('.pdf')); plt.close()

def plot_analysis(metric_name, unit, gt_vals, ours_vals, metrics, output_path):
    """
    Legacy 1x2 plot function.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif'],
        'font.size': FS_TICK_LABEL,
        'axes.labelweight': 'bold'
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE_SINGLE, constrained_layout=True)
    
    is_stress = "Stress" in metric_name
    labels = ["C", "D"] if is_stress else ["A", "B"]
    draw_agreement_pair(ax1, ax2, metric_name, unit, gt_vals, ours_vals, metrics, labels)
    
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.close()

def main():
    data_dir = script_dir if script_dir.name == "data_analysis" else script_dir / "data_analysis"
    
    csv_file = "summary_tables_raw.csv" if USE_RAW_GT else "summary_tables.csv"
    csv_path = data_dir / csv_file
    
    plots_folder = "agreement_plots_raw" if USE_RAW_GT else "agreement_plots"
    plots_dir = data_dir / plots_folder
    plots_dir.mkdir(exist_ok=True)
    
    stats_filename = "agreement_statistics_raw.txt" if USE_RAW_GT else "agreement_statistics.txt"
    stats_file = plots_dir / stats_filename

    tag_str = "RAW " if USE_RAW_GT else ""
    prefix = "raw_" if USE_RAW_GT else ""

    print(f"Loading data from {csv_path}...")
    results = parse_summary_csv(csv_path)
    if not results: return

    analysis_results = {}
    with open(stats_file, 'w') as sf:
        sf.write(f"STATISTICAL AGREEMENT ANALYSIS REPORT ({tag_str}GT vs Clustered Ours)\n")
        sf.write("============================================================\n\n")

        for m_full in results.keys():
            data = results[m_full]
            if not data["GT"] or not data["Ours"]: continue
            metrics = calculate_metrics(data["GT"], data["Ours"])
            unit = data["Unit"]
            stats_str = (
                f"\n--- {m_full} ---\n"
                f" Agreement: R²={metrics['r_squared']:.4f}, Slope={metrics['slope']:.4f}\n"
                f" Bias={metrics['bias']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}\n"
            )
            print(stats_str); sf.write(stats_str)
            cat = "Max Mean" if "Max Mean" in m_full else "Max Median"
            dtype = "Displacement" if "Displacement" in m_full else "Stress"
            if cat not in analysis_results: analysis_results[cat] = {}
            analysis_results[cat][dtype] = (m_full, data["GT"], data["Ours"], metrics, unit)

        for cat, components in analysis_results.items():
            safe_cat = cat.lower().replace(" ", "_")
            combined_path = plots_dir / f"combined_agreement_{prefix}{safe_cat}.png"
            if "Displacement" in components and "Stress" in components:
                plot_combined_analysis(cat, components["Displacement"], components["Stress"], combined_path)
                print(f"{tag_str}Combined plot saved to {combined_path.with_suffix('.pdf')}")
            else:
                for dtype, info in components.items():
                    m_name, gt, ours, metrics, unit = info
                    safe_name = m_name.lower().replace(" ", "_")
                    plot_path = plots_dir / f"agreement_{prefix}{safe_name}.png"
                    plot_analysis(m_name, unit, gt, ours, metrics, plot_path)
                    print(f"Individual plot saved to {plot_path.with_suffix('.pdf')}")

if __name__ == "__main__":
    main()