#!/usr/bin/env python3
"""
Two-panel figure: (A) Max displacement at MCA regions, (B) Max stress at MCA regions.
Produces both a mean and a median version. Uses displacement_region_analysis and
stress_region_analysis outputs. Labels A and B are centered below each panel.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.config_manager import ConfigManager  # type: ignore

# ── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG_FILES = [
    "config/experiments/e4_new.json",
    "config/experiments/e5_new.json"
]

CONDITION_LABELS = ["Cervical", "Terminal"]

LANDMARKS = [
    {"label": "Distal M1\nsegment", "search_key": "M1_Convex"},
    {"label": "MCA bifurcation\n(M1-M2)", "search_key": "M1M2_Bifurcation"},
    {"label": "Inferior M2\ndivision", "search_key": "M2_Caudal"}
]

COLORS = ['#2c3e50', '#1abc9c', '#3498db']
FONT_SIZE_TITLE = 20
FONT_SIZE_AXIS = 16
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 16
FONT_SIZE_PANEL_LABEL = 30
BAR_WIDTH = 0.42
GRID_ALPHA = 0.5
# Two panels side by side: wider figure
FIGURE_SIZE = (16, 6)
DPI = 300
PA_TO_MPA = 1e-6

OUTPUT_DIR = "data/processed/comparison_plots"
OUTPUT_FILENAME_MEAN = "experimental_displacement_and_stress_two_panel_mean"
OUTPUT_FILENAME_MEDIAN = "experimental_displacement_and_stress_two_panel_median"

# Decimal places for bar value labels (same for both panels)
VALUE_DECIMAL_PLACES = 3
VALUE_DECIMAL_PLACES_Y = 2


def load_max_displacement(config_path, search_key, aggregation='mean'):
    """Load max displacement for a region from displacement_region_analysis (mm)."""
    try:
        cm = ConfigManager(config_path)
        output_dir = cm.config['experiment']['output_dir']
        displacement_dir = os.path.join(output_dir, "displacement_region_analysis")
        if not os.path.exists(displacement_dir):
            return 0.0
        pattern = os.path.join(displacement_dir, f"*{search_key}*.npy")
        files = [f for f in glob.glob(pattern) if "indices" not in f.lower()]
        if not files:
            return 0.0
        data = np.load(files[0])
        if data.ndim > 1:
            agg = np.nanmedian(data, axis=0) if aggregation == 'median' else np.nanmean(data, axis=0)
            return float(np.nanmax(agg))
        return float(np.nanmax(data))
    except Exception as e:
        print(f"Error loading displacement {config_path} {search_key}: {e}")
        return 0.0


def load_max_stress(config_path, search_key, aggregation='mean'):
    """Load max stress for a region from stress_region_analysis (MPa, absolute values)."""
    try:
        cm = ConfigManager(config_path)
        output_dir = cm.config['experiment']['output_dir']
        stress_region_dir = os.path.join(output_dir, "stress_region_analysis")
        if not os.path.exists(stress_region_dir):
            return 0.0
        pattern = os.path.join(stress_region_dir, f"calculated_stresses_*{search_key}*.npy")
        files = glob.glob(pattern)
        if not files:
            return 0.0
        data = np.load(files[0], allow_pickle=True)
        if hasattr(data, 'tolist'):
            data = data.tolist()
        if not data or len(data) == 0:
            return 0.0
        arr = np.array([np.atleast_1d(np.asarray(f)).flatten() for f in data])
        if arr.size == 0:
            return 0.0
        arr = np.abs(arr)
        if arr.ndim > 1:
            agg = np.nanmedian(arr, axis=1) if aggregation == 'median' else np.nanmean(arr, axis=1)
            max_pa = np.nanmax(agg)
        else:
            max_pa = np.nanmax(arr)
        return float(max_pa) * PA_TO_MPA
    except Exception as e:
        print(f"Error loading stress {config_path} {search_key}: {e}")
        return 0.0


def draw_bar_chart(ax, data_matrix, title, y_label):
    """Draw grouped bar chart on given axes. Uses shared LANDMARKS, CONDITION_LABELS, COLORS."""
    landmark_labels = [l['label'] for l in LANDMARKS]
    x = np.arange(len(landmark_labels))

    for i, condition_data in enumerate(data_matrix):
        offset = (i - (len(data_matrix) - 1) / 2) * BAR_WIDTH
        label = CONDITION_LABELS[i] if i < len(CONDITION_LABELS) else f"Condition {i+1}"
        color = COLORS[i % len(COLORS)]
        bars = ax.bar(x + offset, condition_data, BAR_WIDTH, label=label, color=color,
                      edgecolor='black', linewidth=1.2, alpha=0.9, zorder=3)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.{VALUE_DECIMAL_PLACES}f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=FONT_SIZE_TICK, fontweight='bold', color=color)

    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_AXIS, fontweight='bold', labelpad=10)
    ax.set_xlabel("Region", fontsize=FONT_SIZE_AXIS, fontweight='bold', labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(landmark_labels, fontsize=FONT_SIZE_TICK, fontweight='semibold')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(f'%.{VALUE_DECIMAL_PLACES_Y}f'))
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=True, loc='upper right',
              facecolor='white', edgecolor='#333333', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA, zorder=0)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.axhline(0, color='black', linewidth=1.2, zorder=4)
    all_vals = [v for row in data_matrix for v in row]
    max_val = max(all_vals) if all_vals else (3.0 if 'mm' in y_label else 0.01)
    ax.set_ylim(0, max_val * 1.25)


def build_and_save_two_panel(disp_matrix, stress_matrix, aggregation, output_filename_base):
    """Build one two-panel figure and save as PDF and PNG."""
    is_mean = aggregation == 'mean'
    agg_label = "Mean" if is_mean else "Median"

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titleweight': 'bold',
        'axes.linewidth': 1.5
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGURE_SIZE, dpi=DPI)

    draw_bar_chart(ax_a, disp_matrix,
                   f"Max {agg_label} Displacement at MCA Regions",
                   f"Max {agg_label} Displacement (mm)")
    draw_bar_chart(ax_b, stress_matrix,
                   f"Max {agg_label} Stress at MCA Regions",
                   f"Max {agg_label} Stress (MPa)")

    ax_a.text(-0.15, 1.02, 'A', transform=ax_a.transAxes,
              fontsize=FONT_SIZE_PANEL_LABEL, fontweight='bold', ha='left', va='bottom')
    ax_b.text(-0.15, 1.02, 'B', transform=ax_b.transAxes,
              fontsize=FONT_SIZE_PANEL_LABEL, fontweight='bold', ha='left', va='bottom')

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
    print("GENERATING TWO-PANEL FIGURES (A: Displacement, B: Stress)")
    print("  — Mean and Median versions")
    print("=" * 60)

    # Load displacement: mean and median
    disp_mean = []
    disp_median = []
    for config_path in CONFIG_FILES:
        row_m, row_med = [], []
        for landmark in LANDMARKS:
            row_m.append(load_max_displacement(config_path, landmark['search_key'], aggregation='mean'))
            row_med.append(load_max_displacement(config_path, landmark['search_key'], aggregation='median'))
        disp_mean.append(row_m)
        disp_median.append(row_med)

    # Load stress: mean and median (absolute values)
    stress_mean = []
    stress_median = []
    for config_path in CONFIG_FILES:
        row_m, row_med = [], []
        for landmark in LANDMARKS:
            row_m.append(load_max_stress(config_path, landmark['search_key'], aggregation='mean'))
            row_med.append(load_max_stress(config_path, landmark['search_key'], aggregation='median'))
        stress_mean.append(row_m)
        stress_median.append(row_med)

    # Mean version
    pdf_mean, png_mean = build_and_save_two_panel(
        disp_mean, stress_mean, 'mean', OUTPUT_FILENAME_MEAN
    )
    print(f"\nMean version:")
    print(f"  PDF: {pdf_mean}")
    print(f"  PNG: {png_mean}")

    # Median version
    pdf_median, png_median = build_and_save_two_panel(
        disp_median, stress_median, 'median', OUTPUT_FILENAME_MEDIAN
    )
    print(f"\nMedian version:")
    print(f"  PDF: {pdf_median}")
    print(f"  PNG: {png_median}")

    print("\n" + "=" * 60)
    print("SUCCESS! All four files generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
