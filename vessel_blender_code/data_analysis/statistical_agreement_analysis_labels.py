#!/usr/bin/env python3
"""
Performs statistical agreement analysis between Ground Truth (GT) and Ours data.
Calculates error metrics, correlations, and generates comparison plots WITH LABELS for each point.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def parse_summary_csv(csv_path):
    """
    Parses the summary CSV and extracts paired GT vs Ours data.
    """
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return None

    # results[Metric] = {"GT": [vals], "Ours": [vals], "Labels": [str], "Unit": str}
    results = {}
    current_metric = None
    current_type = None
    conditions = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            
            # Start of a new metric block
            if "MAX of" in row[0]:
                title = row[0]
                # Determine metric category (Average/Median) and data type (Disp/Stress)
                cat = "Mean" if "AVERAGE" in title else "Median"
                dtype = "Displacement" if "Displacement" in title else "Stress"
                unit = "mm" if dtype == "Displacement" else "Pa"
                
                current_metric = f"Max {cat} {dtype}"
                results[current_metric] = {"GT": [], "Ours": [], "Labels": [], "Unit": unit}
                continue
            
            # Header row for the conditions
            if row[0].strip().lower() in ["", "condition"] and len(row) > 1:
                conditions = row[1:]
                continue
            
            # Type identifier
            if row[0] == "GT:":
                current_type = "GT"
                continue
            elif row[0] == "Ours:":
                current_type = "Ours"
                continue
            
            # Data row
            region_name = row[0].strip()
            if current_metric and current_type and region_name.startswith("R"):
                vals = [float(v) for v in row[1:] if v]
                results[current_metric][current_type].extend(vals)
                
                # Only add labels once (during GT pass)
                if current_type == "GT":
                    if not conditions:
                        # Fallback if header parsing missed but values exist
                        for _ in range(len(vals)):
                            results[current_metric]["Labels"].append(region_name)
                    else:
                        for cond in conditions:
                            if len(conditions) > 1:
                                results[current_metric]["Labels"].append(f"{region_name}_{cond}")
                            else:
                                results[current_metric]["Labels"].append(f"{region_name}")
                        
    return results

def calculate_metrics(y_true, y_pred):
    """
    Calculate statistical agreement metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Linear Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    
    # Error Metrics
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    # Handle division by zero for MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0
    
    # Bland-Altman
    mean_diff = np.mean(y_pred - y_true)
    std_diff = np.std(y_pred - y_true)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "bias": mean_diff,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower
    }

def plot_analysis_with_labels(metric_name, unit, gt_vals, ours_vals, point_labels, metrics, output_path):
    """
    Generate professional, publication-quality analysis plots with labeled points.
    """
    gt_vals = np.array(gt_vals)
    ours_vals = np.array(ours_vals)
    
    # Scale Stress for better readability
    display_unit = unit
    display_gt = gt_vals
    display_ours = ours_vals
    display_metrics = metrics.copy()
    
    if unit == "Pa" and np.max(gt_vals) > 1000:
        display_unit = "MPa"
        display_gt = gt_vals / 1e6
        display_ours = ours_vals / 1e6
        display_metrics['mae'] /= 1e6
        display_metrics['rmse'] /= 1e6
        display_metrics['bias'] /= 1e6
        display_metrics['loa_upper'] /= 1e6
        display_metrics['loa_lower'] /= 1e6
        display_metrics['intercept'] /= 1e6

    # Professional Styles
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'sans-serif'],
        'font.size': 20,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titleweight': 'bold'
    })

    # Create single Correlation Plot (Compact Vertically)
    fig, ax_corr = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Correlation Analysis: {metric_name}', fontsize=22, y=0.95)

    # Professional colors
    color_gt_ours = '#4a86e8'  # Professional Blue
    color_reg = '#e06666'      # Subtle Red

    # --- Correlation Plot ---
    ax_corr.scatter(display_gt, display_ours, color=color_gt_ours, alpha=0.7, 
                edgecolors='white', linewidth=0.5, s=120, label='Observations')
    
    # Add labels to points in Correlation Plot with intelligent offsets to avoid overlap
    for i, label in enumerate(point_labels):
        if i % 2 == 0:
            off_x, off_y = -20, 20
            ha, va = 'right', 'bottom'
        else:
            off_x, off_y = 20, -20
            ha, va = 'left', 'top'
            
        if "R3" in label: off_x, off_y = -25, 10; ha, va = 'right', 'center'
        if "R4" in label: off_x, off_y = 10, -25; ha, va = 'center', 'top'
        if "R8" in label: off_x, off_y = 25, 15; ha, va = 'left', 'bottom'

        ax_corr.annotate(label, (display_gt[i], display_ours[i]), 
                     xytext=(off_x, off_y), textcoords='offset points', 
                     fontsize=20, alpha=0.9, fontweight='bold',
                     ha=ha, va=va,
                     arrowprops=dict(arrowstyle='-', color='#333333', alpha=0.5, lw=2))
    
    # Identity line
    max_val = max(max(display_gt), max(display_ours))
    
    # Set limits to start at 0 and provide enough room for labels on the right/top
    plot_max = max_val * 1.25
    ax_corr.set_xlim(0, plot_max)
    ax_corr.set_ylim(0, plot_max)
    
    lims = [0, plot_max]
    ax_corr.plot(lims, lims, color='#333333', linestyle='--', alpha=0.5, linewidth=1.5, label='Identity (Ideal)')
    
    # Regression line
    x_reg = np.array(lims)
    y_reg = display_metrics['slope'] * x_reg + display_metrics['intercept']
    ax_corr.plot(x_reg, y_reg, color=color_reg, linestyle='-', linewidth=2.5, 
             label=f'Linear Fit (R²={display_metrics["r_squared"]:.3f})')
    
    ax_corr.set_xlabel(f'Ground Truth ({display_unit})', labelpad=12, fontsize=20)
    ax_corr.set_ylabel(f'Ours ({display_unit})', labelpad=12, fontsize=20)
    ax_corr.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=20, loc='lower right')
    ax_corr.grid(True, linestyle='--', alpha=0.4)
    
    # Text box for metrics (Simplified)
    textstr = '\n'.join((
        f'Slope: {display_metrics["slope"]:.3f}',
        f'MAE: {display_metrics["mae"]:.4f} {display_unit}',
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9)
    ax_corr.text(0.05, 0.95, textstr, transform=ax_corr.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)

    ax_corr.spines['top'].set_visible(False)
    ax_corr.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    plt.close()

def main():
    script_dir = Path(__file__).parent
    # Handle running from data_analysis or project root
    if script_dir.name == "data_analysis":
        data_dir = script_dir
    else:
        data_dir = script_dir / "data_analysis"
        
    csv_path = data_dir / "summary_tables.csv"
    
    print(f"Loading data from {csv_path}...")
    results = parse_summary_csv(csv_path)
    if not results: 
        print("Falling back to summary_tables_one_condition.csv?")
        # If the user is using the one-condition script, they might have this instead
        # However, the format is different, so we'd need a different parser.
        # For now, stick to standard format.
        return

    for metric_full_name in results.keys():
        data = results[metric_full_name]
        if not data["GT"] or not data["Ours"]: continue
        
        print(f"\n--- Analysis for {metric_full_name} ---")
        metrics = calculate_metrics(data["GT"], data["Ours"])
        
        unit = data["Unit"]
        print(f"Agreement Statistics:")
        print(f"  R-squared:    {metrics['r_squared']:.4f}")
        print(f"  Slope:        {metrics['slope']:.4f} (Ideal: 1.0)")
        print(f"  Mean Bias:   {metrics['bias']:.4f} {unit}")
        print(f"  MAE:          {metrics['mae']:.4f} {unit}")
        print(f"  RMSE:         {metrics['rmse']:.4f} {unit}")
        print(f"  MAPE:         {metrics['mape']:.2f} %")
        print(f"  95% LoA:      [{metrics['loa_lower']:.4f}, {metrics['loa_upper']:.4f}]")
        
        # Save plots
        plots_dir = data_dir / "agreement_plots"
        plots_dir.mkdir(exist_ok=True)

        safe_name = metric_full_name.lower().replace(" ", "_")
        plot_path = plots_dir / f"agreement_{safe_name}_labeled.png"
        plot_analysis_with_labels(metric_full_name, unit, data["GT"], data["Ours"], data["Labels"], metrics, plot_path)
        print(f"Plots saved to {plot_path} (and .svg)")

if __name__ == "__main__":
    main()