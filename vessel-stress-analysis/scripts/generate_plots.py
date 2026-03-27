from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# File Paths
STRESS_CSV = str(Path.home() / "Documents/GitHub/vessel-stress-analysis/data/processed/averaged_stresses_regions_scaled.csv")
DISPLACEMENT_CSV = str(Path.home() / "Documents/GitHub/vessel-stress-analysis/data/processed/averaged_displacements_regions.csv")
OUTPUT_DIR = str(Path.home() / "Documents/GitHub/vessel-stress-analysis/data/processed/analysis_results")

# Standardizing Region Display Names
REGION_MAP = {
    'm1_convex': 'M1 Convex',
    'm1_m2_bifurcation': 'M1/M2 Bifurcation',
    'm2_caudal': 'M2 Caudal',
    'm1m2_bifurcation': 'M1/M2 Bifurcation',
    'm1_m2_bifurcatiohn': 'M1/M2 Bifurcation',
    'm1_conex': 'M1 Convex'
}

def clean_region_name(raw_name):
    """Normalize region name from CSV header."""
    name = raw_name.lower()
    name = name.replace('_displacement', '')
    for key, display in REGION_MAP.items():
        if key in name:
            return display
    return raw_name.replace('_', ' ').title()

def apply_scientific_style(ax, title, xlabel, ylabel):
    """Apply professional/scientific styling to an axis."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='normal')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')
    
    # Grid lines
    ax.grid(True, which='major', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicker spines for clarity
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=1, length=5)

def plot_experiment_data(df, category, unit, scale=1.0):
    """
    df: DataFrame (Stress or Displacement)
    category: 'Stress' or 'Displacement'
    unit: 'MPa' or 'mm'
    scale: multiplier for unit conversion
    """
    # Use a clean base style
    plt.style.use('seaborn-v0_8-paper')
    
    headers = df.columns
    exps = sorted(list(set(re.search(r'exp (a\d+)', h).group(1) for h in headers if re.search(r'exp (a\d+)', h))))
    
    for exp in exps:
        time_col = f"exp {exp} Time"
        if time_col not in df.columns:
            continue
            
        region_cols = [c for c in headers if c.startswith(f"exp {exp} ") and "Time" not in c]
        
        for col in region_cols:
            region_display = clean_region_name(col.replace(f"exp {exp} ", ""))
            
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            
            plot_data = df[[time_col, col]].dropna()
            if plot_data.empty:
                plt.close()
                continue
                
            color = '#1f77b4' if category == 'Displacement' else '#d62728'
            
            # Plot with markers and lines
            ax.plot(plot_data[time_col], plot_data[col] * scale, 
                     marker='o', linestyle='-', color=color,
                     linewidth=1.5, markersize=3, alpha=0.9,
                     markerfacecolor='white', markeredgewidth=1.0)
            
            title = f"Experiment {exp}: {category} Over Time - {region_display}"
            xlabel = "Time (s)"
            ylabel = f"{category} ({unit})"
            
            apply_scientific_style(ax, title, xlabel, ylabel)
            
            # Tight layout to prevent clipping
            plt.tight_layout()
            
            safe_region = region_display.replace('/', '_').replace(' ', '_')
            filename = f"{exp}_{category.lower()}_{safe_region}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
            plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Displacement Data...")
    if os.path.exists(DISPLACEMENT_CSV):
        df_disp = pd.read_csv(DISPLACEMENT_CSV)
        print("Generating Professional Displacement Plots...")
        plot_experiment_data(df_disp, "Displacement", "mm", scale=1.0)
    else:
        print(f"Warning: {DISPLACEMENT_CSV} not found.")

    print("Loading Stress Data...")
    if os.path.exists(STRESS_CSV):
        df_stress = pd.read_csv(STRESS_CSV)
        print("Generating Professional Stress Plots...")
        plot_experiment_data(df_stress, "Stress", "MPa", scale=1e-6)
    else:
        print(f"Warning: {STRESS_CSV} not found.")

    print(f"\nAll scientific plots generated in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
