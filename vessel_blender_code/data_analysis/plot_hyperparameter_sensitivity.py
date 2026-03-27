import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
import seaborn as sns
import numpy as np
import ast
from pathlib import Path

def main():
    csv_path = Path("data_analysis/synthetic_regularization_summary/regularization_study_results.csv")
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Parse params string [rgd, spd, phy] into separate columns
    def parse_params(p_str):
        try:
            p_list = ast.literal_eval(p_str)
            return pd.Series({'rgd': p_list[0], 'spd': p_list[1], 'phy': p_list[2]})
        except:
            return pd.Series({'rgd': 0, 'spd': 0, 'phy': 0})

    param_df = df['params'].apply(parse_params)
    df = pd.concat([df, param_df], axis=1)

    # Use stress_diff_mpa for color analysis (absolute value for "lower is better")
    # Actually, the user might want to see the actual value or absolute. 
    # Usually "lower error" means absolute closer to 0.
    df['abs_stress_bias'] = df['stress_diff_mpa'].abs()
    
    output_dir = csv_path.parent
    
    # Pairs to compare
    pairs = [
        ('rgd', 'spd', 'lambda_rigid', 'lambda_speed'),
        ('rgd', 'phy', 'lambda_rigid', 'lambda_phys'),
        ('spd', 'phy', 'lambda_speed', 'lambda_phys')
    ]

    plt.style.use('ggplot')
    
    for x_col, y_col, x_label, y_label in pairs:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Add jitter to handle overlaps from the 3rd parameter
        # Calculate jitter based on the scale of the axes
        x_range = df[x_col].max() - df[x_col].min()
        y_range = df[y_col].max() - df[y_col].min()
        
        # Jitter factor (small percentage of range)
        x_jitter = (np.random.rand(len(df)) - 0.5) * (x_range * 0.05 if x_range > 0 else 0.02)
        y_jitter = (np.random.rand(len(df)) - 0.5) * (y_range * 0.05 if y_range > 0 else 0.02)
        
        jittered_x = df[x_col] + x_jitter
        jittered_y = df[y_col] + y_jitter

        scatter = ax.scatter(jittered_x, jittered_y, 
                            c=df['abs_stress_bias'], 
                            s=250, 
                            cmap='viridis_r', 
                            edgecolors='black',
                            linewidths=1.5,
                            alpha=0.9,
                            zorder=2)
        
        # Add values as labels for clarity
        for i, row in df.iterrows():
            # Create a label showing the actual hyperparameter values
            param_label = f"rgd:{row['rgd']}\nspd:{row['spd']}\nphy:{row['phy']}"
            ax.annotate(param_label, 
                        (jittered_x[i], jittered_y[i]),
                        textcoords="offset points", 
                        xytext=(0,12), 
                        ha='center',
                        fontsize=8,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                        zorder=3)

        cbar = plt.colorbar(scatter)
        cbar.set_label('Absolute Stress Bias (MPa)', fontsize=12)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # Use log scale if values vary widely
        if df[x_col].min() > 0:
            ax.set_xscale('log')
        if df[y_col].min() > 0:
            ax.set_yscale('log')
            
        plt.tight_layout()
        plot_path = output_dir / f"sensitivity_{x_col}_vs_{y_col}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Saved plot: {plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
