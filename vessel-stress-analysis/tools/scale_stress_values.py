import pandas as pd
import os
import argparse

def scale_csv_values(file_path, scale_factor, output_suffix="_scaled"):
    """
    Scales specific columns in a CSV file by a scale factor.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return

    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    
    # Logic for averaged_stresses_regions.csv
    if "averaged_stresses_regions.csv" in file_path:
        # Scale everything except 'Frame' and columns containing 'Time'
        cols_to_scale = [col for col in df.columns if col.lower() != 'frame' and 'time' not in col.lower()]
        df[cols_to_scale] = df[cols_to_scale] * scale_factor
        
    # Logic for stress_statistics_summary.csv
    elif "stress_statistics_summary.csv" in file_path:
        # Scale the Mean columns
        cols_to_scale = ['Cervical Mean', 'Terminal Mean', 'Cavernous Mean']
        for col in cols_to_scale:
            if col in df.columns:
                df[col] = df[col] * scale_factor
                
    # Logic for max_stresses_regions_pivot.csv
    elif "max_stresses_regions_pivot.csv" in file_path:
        # Trial and Regions are labels, everything else is numeric/Average
        cols_to_exclude = ['Trial', 'Regions']
        cols_to_scale = [col for col in df.columns if col not in cols_to_exclude]
        df[cols_to_scale] = df[cols_to_scale] * scale_factor

    else:
        # Default behavior: scale all numeric columns
        cols_to_scale = df.select_dtypes(include=['number']).columns
        df[cols_to_scale] = df[cols_to_scale] * scale_factor

    # Save to original folder with suffix
    base, ext = os.path.splitext(file_path)
    # The user might want to overwrite or have a specific name.
    # We will save to a new file by default but print the path.
    output_path = f"{base}{output_suffix}{ext}"
    df.to_csv(output_path, index=False)
    print(f"Saved scaled CSV to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale stress values in CSV files.")
    parser.add_argument("--factor", type=float, default=1.15/25.5, help="Scaling factor (default: 1.15/25.5)")
    parser.add_argument("--suffix", type=str, default="_scaled", help="Suffix for output files (default: _scaled)")
    
    args = parser.parse_args()
    
    scale_factor = args.factor
    suffix = args.suffix
    
    # Target files
    target_files = [
        r"data/processed/averaged_stresses_regions.csv",
        r"data/processed/max_region_analysis/statistical_plots/stress_statistics_summary.csv",
        r"data/processed/max_region_analysis/max_stresses_regions_pivot.csv"
    ]
    
    # Get absolute paths
    workspace_root = os.getcwd()
    abs_targets = [os.path.join(workspace_root, f) for f in target_files]
    
    for f in abs_targets:
        scale_csv_values(f, scale_factor, suffix)
