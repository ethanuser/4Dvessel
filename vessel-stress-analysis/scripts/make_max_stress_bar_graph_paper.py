#!/usr/bin/env python3
"""
Script to generate a bar graph comparing maximum mean/median stresses between two conditions.
Uses stress data from stress_analysis_region.py (calculated_stresses_*.npy in stress_region_analysis).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import after path setup
from utils.config_manager import ConfigManager  # type: ignore

# ── CONFIG FILES ────────────────────────────────────────────────────────────
CONFIG_FILES = [
    "config/experiments/e4_new.json",  # Condition 1
    "config/experiments/e5_new.json"   # Condition 2
]

# ── CONFIGURABLE CONSTANTS ──────────────────────────────────────────────────
PLOT_TITLE_MEAN = "Max Mean Stress\n at MCA Regions"
PLOT_TITLE_MEDIAN = "Max Median Stress\n at MCA Regions"
Y_LABEL_MEAN = "Max Mean Stress (MPa)"
Y_LABEL_MEDIAN = "Max Median Stress (MPa)"
X_LABEL = "Region"

CONDITION_LABELS = ["Cervical", "Terminal"]

# Region labels and search keys for finding calculated_stresses_*{search_key}.npy
LANDMARKS = [
    {"label": "M1 Convex", "search_key": "M1_Convex"},
    {"label": "M1/M2 Bifurcation", "search_key": "M1M2_Bifurcation"},
    {"label": "M2 Inferior Branch", "search_key": "M2_Caudal"}
]

# Visual Styling
COLORS = ['#2c3e50', '#1abc9c', '#3498db']
FONT_SIZE_TITLE = 24
FONT_SIZE_AXIS = 18
FONT_SIZE_TICK = 18
FONT_SIZE_LEGEND = 18
BAR_WIDTH = 0.35
GRID_ALPHA = 0.5
FIGURE_SIZE = (10, 8)
DPI = 300

OUTPUT_DIR = "data/processed/comparison_plots"
OUTPUT_FILENAME_MEAN = "experimental_stress_comparison_mean"
OUTPUT_FILENAME_MEDIAN = "experimental_stress_comparison_median"

# Stress is stored in Pa; we display in MPa
PA_TO_MPA = 1e-6


def load_max_stress(config_path, search_key, aggregation='mean'):
    """
    Load the maximum stress for a region from stress_analysis_region.py output.
    Looks in {output_dir}/stress_region_analysis/ for calculated_stresses_*{search_key}*.npy.
    aggregation: 'mean' or 'median' — aggregate across edges at each time step before taking max.
    Returns value in MPa.
    """
    try:
        cm = ConfigManager(config_path)
        output_dir = cm.config['experiment']['output_dir']
        stress_region_dir = os.path.join(output_dir, "stress_region_analysis")

        if not os.path.exists(stress_region_dir):
            print(f"Warning: Stress region directory not found for {config_path}")
            return 0.0

        # Match calculated_stresses_*{search_key}*.npy (e.g. calculated_stresses_M1_Convex.npy)
        pattern = os.path.join(stress_region_dir, f"calculated_stresses_*{search_key}*.npy")
        files = glob.glob(pattern)

        if not files:
            print(f"Warning: No stress file found for key '{search_key}' in {stress_region_dir}")
            return 0.0

        # Load: list of arrays (one per frame), each array = stress per edge in region (Pa)
        data = np.load(files[0], allow_pickle=True)
        if hasattr(data, 'tolist'):
            data = data.tolist()

        if not data or len(data) == 0:
            return 0.0

        # Stack to (n_frames, n_edges); handle variable-length frames if needed
        arr = np.array([np.atleast_1d(np.asarray(f)).flatten() for f in data])
        if arr.size == 0:
            return 0.0

        # Use absolute values only (stress magnitude) before aggregating
        arr = np.abs(arr)

        # Aggregate across edges at each time step (axis=1), then max over time
        if arr.ndim > 1:
            if aggregation == 'median':
                agg_per_frame = np.nanmedian(arr, axis=1)
            else:
                agg_per_frame = np.nanmean(arr, axis=1)
            max_stress_pa = np.nanmax(agg_per_frame)
        else:
            max_stress_pa = np.nanmax(arr)

        return float(max_stress_pa) * PA_TO_MPA  # MPa

    except Exception as e:
        print(f"Error loading stress for {config_path} with key {search_key}: {e}")
        return 0.0


def build_and_save_plot(data_matrix, plot_title, y_label, output_filename_base):
    """Build the bar chart and save as PDF and PNG."""
    landmark_labels = [l['label'] for l in LANDMARKS]
    x = np.arange(len(landmark_labels))

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titleweight': 'bold',
        'axes.linewidth': 1.5
    })

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

    for i, condition_data in enumerate(data_matrix):
        offset = (i - (len(data_matrix) - 1) / 2) * BAR_WIDTH
        label = CONDITION_LABELS[i] if i < len(CONDITION_LABELS) else f"Condition {i+1}"
        color = COLORS[i % len(COLORS)]

        bars = ax.bar(x + offset, condition_data, BAR_WIDTH, label=label, color=color,
                      edgecolor='black', linewidth=1.2, alpha=0.9, zorder=3)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=FONT_SIZE_TICK, fontweight='bold',
                       color=color)

    ax.set_title(plot_title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_AXIS, fontweight='bold', labelpad=15)
    ax.set_xlabel(X_LABEL, fontsize=FONT_SIZE_AXIS, fontweight='bold', labelpad=15)

    ax.set_xticks(x)
    ax.set_xticklabels(landmark_labels, fontsize=FONT_SIZE_TICK, fontweight='semibold')
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)

    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=True, loc='upper right',
              facecolor='white', edgecolor='#333333', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA, zorder=0)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.axhline(0, color='black', linewidth=1.5, zorder=4)

    all_values = [v for sublist in data_matrix for v in sublist]
    max_val = max(all_values) if all_values else 0.01
    ax.set_ylim(0, max_val * 1.25)

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, f"{output_filename_base}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{output_filename_base}.png")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=DPI)
    plt.close()

    return pdf_path, png_path


def main():
    print("=" * 60)
    print("GENERATING EXPERIMENTAL STRESS BAR GRAPHS (MEAN & MEDIAN)")
    print("=" * 60)

    # 1. Collect data for mean
    data_matrix_mean = []
    print("\n--- Mean stress ---")
    for config_path in CONFIG_FILES:
        condition_data = []
        print(f"\nProcessing config: {config_path}")
        for landmark in LANDMARKS:
            max_stress_mpa = load_max_stress(config_path, landmark['search_key'], aggregation='mean')
            condition_data.append(max_stress_mpa)
            print(f"  - {landmark['label']}: {max_stress_mpa:.6f} MPa")
        data_matrix_mean.append(condition_data)

    # 2. Collect data for median
    data_matrix_median = []
    print("\n--- Median stress ---")
    for config_path in CONFIG_FILES:
        condition_data = []
        print(f"\nProcessing config: {config_path}")
        for landmark in LANDMARKS:
            max_stress_mpa = load_max_stress(config_path, landmark['search_key'], aggregation='median')
            condition_data.append(max_stress_mpa)
            print(f"  - {landmark['label']}: {max_stress_mpa:.6f} MPa")
        data_matrix_median.append(condition_data)

    # 3. Build and save mean plot
    pdf_mean, png_mean = build_and_save_plot(
        data_matrix_mean, PLOT_TITLE_MEAN, Y_LABEL_MEAN, OUTPUT_FILENAME_MEAN
    )

    # 4. Build and save median plot
    pdf_median, png_median = build_and_save_plot(
        data_matrix_median, PLOT_TITLE_MEDIAN, Y_LABEL_MEDIAN, OUTPUT_FILENAME_MEDIAN
    )

    print("\n" + "=" * 60)
    print("SUCCESS! Plots generated successfully.")
    print(f"  Mean   - PDF: {pdf_mean}")
    print(f"  Mean   - PNG: {png_mean}")
    print(f"  Median - PDF: {pdf_median}")
    print(f"  Median - PNG: {png_median}")
    print("=" * 60)


if __name__ == "__main__":
    main()
