#!/usr/bin/env python3
"""
Script to generate a bar graph comparing maximum mean displacements between two conditions.
Figure R1 — Experimental displacement comparison (core result)
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
from utils.config_manager import ConfigManager # type: ignore

# ── CONFIG FILES ────────────────────────────────────────────────────────────
# List the particular config files to look at here
CONFIG_FILES = [
    "config/experiments/e4.json",  # Condition 1
    "config/experiments/e5.json"   # Condition 2
]

# ── CONFIGURABLE CONSTANTS ──────────────────────────────────────────────────
# Graph Titles and Labels
PLOT_TITLE = "Max Mean Displacement\n at MCA Regions"
Y_LABEL = "Max Mean Displacement (mm)"
X_LABEL = "Region"

# Legend Labels (Conditions)
CONDITION_LABELS = ["Cervical", "Terminal"]

# Landmark Labels and Search Keys
# These keys are used to find the .npy files in the displacement_analysis directory
LANDMARKS = [
    {"label": "M1/M2 Bifurcation", "search_key": "bifurcation"},
    {"label": "Proximal M2", "search_key": "proximal"}
]

# Visual Styling
# Use a professional, publication-quality palette (Teal & Slate)
COLORS = ['#2c3e50', '#1abc9c']  # Midnight Blue and Vibrant Teal
FONT_SIZE_TITLE = 24
FONT_SIZE_AXIS = 18
FONT_SIZE_TICK = 18
FONT_SIZE_LEGEND = 18
BAR_WIDTH = 0.35
GRID_ALPHA = 0.5
FIGURE_SIZE = (6, 9)
DPI = 300

# Output Path
OUTPUT_DIR = "data/processed/comparison_plots"
OUTPUT_FILENAME = "experimental_displacement_comparison"

def load_max_displacement(config_path, search_key):
    """
    Load the maximum mean displacement for a given search key from a config's output directory.
    """
    try:
        # Load config
        cm = ConfigManager(config_path)
        output_dir = cm.config['experiment']['output_dir']
        displacement_dir = os.path.join(output_dir, "displacement_analysis")
        
        if not os.path.exists(displacement_dir):
            print(f"Warning: Displacement directory not found for {config_path}")
            return 0.0
            
        # Find file matching search key
        pattern = os.path.join(displacement_dir, f"*{search_key}*.npy")
        files = glob.glob(pattern)
        
        # Filter out indices files
        files = [f for f in files if "indices" not in f.lower()]
        
        if not files:
            print(f"Warning: No displacement file found for key '{search_key}' in {displacement_dir}")
            return 0.0
            
        # Load the first matching file
        data = np.load(files[0])
        
        # Calculate max mean displacement
        if data.ndim > 1:
            mean_across_points = np.nanmean(data, axis=0) # Mean across points at each time step
            max_disp = np.nanmax(mean_across_points)     # Max value over all time steps
        else:
            # Single point data
            max_disp = np.nanmax(data)
            
        return max_disp
        
    except Exception as e:
        print(f"Error loading data for {config_path} with key {search_key}: {e}")
        return 0.0

def main():
    print("="*60)
    print("GENERATING EXPERIMENTAL DISPLACEMENT BAR GRAPH")
    print("="*60)
    
    # 1. Collect Data
    data_matrix = [] # Rows: Conditions, Cols: Landmarks
    
    for config_path in CONFIG_FILES:
        condition_data = []
        print(f"\nProcessing config: {config_path}")
        for landmark in LANDMARKS:
            max_disp = load_max_displacement(config_path, landmark['search_key'])
            condition_data.append(max_disp)
            print(f"  - {landmark['label']}: {max_disp:.4f} mm")
        data_matrix.append(condition_data)
        
    # 2. Prepare Plot
    landmark_labels = [l['label'] for l in LANDMARKS]
    x = np.arange(len(landmark_labels))
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    # Professional Styles from reference
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titleweight': 'bold',
        'axes.linewidth': 1.5
    })
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Create bars
    for i, condition_data in enumerate(data_matrix):
        # Calculate offset for grouped bars
        offset = (i - (len(data_matrix)-1)/2) * BAR_WIDTH
        label = CONDITION_LABELS[i] if i < len(CONDITION_LABELS) else f"Condition {i+1}"
        color = COLORS[i % len(COLORS)]
        
        bars = ax.bar(x + offset, condition_data, BAR_WIDTH, label=label, color=color, 
               edgecolor='black', linewidth=1.2, alpha=0.9, zorder=3)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', 
                        fontsize=FONT_SIZE_TICK, fontweight='bold',
                        color=color)

    # 3. Customize Plot
    ax.set_title(PLOT_TITLE, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=25)
    ax.set_ylabel(Y_LABEL, fontsize=FONT_SIZE_AXIS, fontweight='bold', labelpad=15)
    ax.set_xlabel(X_LABEL, fontsize=FONT_SIZE_AXIS, fontweight='bold', labelpad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(landmark_labels, fontsize=FONT_SIZE_TICK, fontweight='semibold')
    
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
    
    # Add legend with professional box styling (stronger outline)
    ax.legend(fontsize=FONT_SIZE_LEGEND, frameon=True, loc='upper right', 
              facecolor='white', edgecolor='#333333', framealpha=0.9)
    
    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA, zorder=0)
    
    # Remove top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Highlight the zero line
    ax.axhline(0, color='black', linewidth=1.5, zorder=4)
    
    # Set limits with padding
    all_values = [v for sublist in data_matrix for v in sublist]
    max_val = max(all_values) if all_values else 3.0
    ax.set_ylim(0, max_val * 1.25)
    
    plt.tight_layout()
    
    # 4. Save Plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    svg_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.svg")
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.png")
    
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=DPI)
    
    print("\n" + "="*60)
    print(f"SUCCESS! Plot generated successfully.")
    print(f"  - SVG: {svg_path}")
    print(f"  - PNG: {png_path}")
    print("="*60)

    print(f"Target values check:")
    print(f"  Bifurcation: ~2.5 mm vs ~0.6 mm")
    print(f"  Proximal M2: ~1.8 mm vs ~0.3 mm")
    print("="*60)

if __name__ == "__main__":
    main()
