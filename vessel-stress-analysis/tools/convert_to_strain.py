import pandas as pd
import os
import argparse

def convert_stress_to_strain(file_path, youngs_modulus):
    """
    Converts stress values to strain by dividing by Young's Modulus.
    Renames columns and the output file accordingly.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return

    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    
    # Scale values by 1 / Young's Modulus
    # We apply this to columns that were previously identified as stress columns
    
    # 1. Rename columns: replace 'Stress' with 'Strain' (case insensitive)
    new_columns = {}
    for col in df.columns:
        if 'stress' in col.lower():
            new_col = col.replace('Stress', 'Strain').replace('stress', 'strain').replace('STRESS', 'STRAIN')
            new_columns[col] = new_col
    
    # 2. Scale the values in these columns
    for old_col, new_col in new_columns.items():
        # Only scale if it's a numeric column
        if pd.api.types.is_numeric_dtype(df[old_col]):
            df[old_col] = df[old_col] / youngs_modulus
            
    # Rename columns in the dataframe
    df.rename(columns=new_columns, inplace=True)

    # 3. Determine output path: replace 'stress' with 'strain' in the filename
    base_dir = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    new_filename = filename.replace('stresses', 'strains').replace('stress', 'strain')
    
    # Ensure the filename actually changed, otherwise append _strain
    if new_filename == filename:
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_strain{ext}"
        
    output_path = os.path.join(base_dir, new_filename)
    
    # Save the new version
    df.to_csv(output_path, index=False)
    print(f"Saved strain data to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert stress values to strain in CSV files.")
    parser.add_argument("--modulus", type=float, default=1150000.0, help="Young's Modulus (default: 1150000.0)")
    
    args = parser.parse_args()
    
    youngs_modulus = args.modulus
    
    # Target files (same as in scale_stress_values.py)
    target_files = [
        "data/processed/averaged_stresses_regions.csv",
        "data/processed/max_region_analysis/statistical_plots/stress_statistics_summary.csv",
        "data/processed/max_region_analysis/max_stresses_regions_pivot.csv"
    ]
    
    # Get absolute paths
    workspace_root = os.getcwd()
    abs_targets = [os.path.join(workspace_root, f) for f in target_files]
    
    for f in abs_targets:
        convert_stress_to_strain(f, youngs_modulus)
