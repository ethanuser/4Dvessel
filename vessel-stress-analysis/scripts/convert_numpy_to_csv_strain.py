import os
import numpy as np
import glob
import csv
import json
import re

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CONFIG_DIR = os.path.join(BASE_DIR, "config", "experiments")

# ============================================================================
# CONFIGURATION
# ============================================================================
# Default Young's Modulus if not specified in config (adjust if needed)
DEFAULT_YOUNGS_MODULUS = 1150000.0
# Whether to apply the custom scale factor (1.15 / 25.5)
APPLY_CUSTOM_SCALE = True
DEFAULT_SCALE_FACTOR = 1.15 / 25.5
# ============================================================================

def load_experiment_config(exp_name):
    """Loads the experiment configuration JSON."""
    config_path = os.path.join(CONFIG_DIR, f"{exp_name}.json")
    if not os.path.exists(config_path):
        return None
        
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config for {exp_name}: {e}")
        return None

def get_experiment_data(exp_dir):
    """
    Extracts strain data (scaled from stress) for a single experiment.
    Returns:
        tuple: (exp_name, total_time, strain_data_dict, num_frames)
    """
    exp_name = os.path.basename(exp_dir)
    config = load_experiment_config(exp_name)
    
    total_time = None
    youngs_modulus = DEFAULT_YOUNGS_MODULUS
    
    if config:
        if "experiment" in config:
            total_time = config["experiment"].get("total_time")
        if "material_properties" in config:
            youngs_modulus = config["material_properties"].get("young_modulus_silicone", DEFAULT_YOUNGS_MODULUS)
    
    num_frames = 0
    strain_data = {}
    
    scale_factor = DEFAULT_SCALE_FACTOR if APPLY_CUSTOM_SCALE else 1.0
    
    # --- Load Stresses and Convert to Strain ---
    region_dir = os.path.join(exp_dir, "stress_region_analysis")
    if os.path.exists(region_dir):
        npy_files = glob.glob(os.path.join(region_dir, "*.npy"))
        for npy_path in npy_files:
            try:
                fname = os.path.basename(npy_path)
                if fname.startswith("calculated_stresses_") and fname.endswith(".npy"):
                    region_name = fname.replace("calculated_stresses_", "").replace(".npy", "")
                else:
                    region_name = fname.replace(".npy", "")
                
                s_data = np.load(npy_path, allow_pickle=True)
                if len(s_data) > 0:
                    if not isinstance(s_data, np.ndarray):
                        s_data = np.array(s_data)

                    # Robust averaging logic for stress regions: (frames, edges)
                    if s_data.ndim == 2:
                        # For stresses saved as (frames, edges), axis 1 is usually huge (edges)
                        if s_data.shape[1] > s_data.shape[0]:
                            values = np.nanmean(s_data, axis=1)
                        else:
                            values = np.nanmean(s_data, axis=0)
                    elif s_data.ndim == 1:
                        values = s_data
                    else:
                        values_list = []
                        for val in s_data:
                            if np.isscalar(val):
                                values_list.append(val)
                            elif hasattr(val, '__len__'):
                                values_list.append(np.nanmean(val))
                            else:
                                values_list.append(np.nan)
                        values = np.array(values_list)
                    
                    # SCALE TO STRAIN: (stress / E) * optional_scale
                    strain_values = (values / youngs_modulus * scale_factor).tolist()
                    
                    num_frames = max(num_frames, len(strain_values))
                    strain_data[region_name] = strain_values
            except Exception as e:
                print(f"  Error loading/scaling stress {os.path.basename(npy_path)} for {exp_name}: {e}")

    return exp_name, total_time, strain_data, num_frames

def aggregate_data():
    """Aggregates strain data from experiments and writes to CSV."""
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found: {DATA_DIR}")
        return

    all_experiments = []
    
    # 1. Collect Data
    target_experiments = [f"a{i}" for i in range(1, 19)]
    
    print(f"Collecting and scaling stress data to strain (Apply custom scale: {APPLY_CUSTOM_SCALE})...")
    for item in target_experiments:
        item_path = os.path.join(DATA_DIR, item)
        if os.path.exists(item_path) and os.path.isdir(item_path):
            print(f"  Processing {item}...")
            exp_data = get_experiment_data(item_path)
            if exp_data[3] > 0: # num_frames > 0
                all_experiments.append(exp_data)
            else:
                print(f"    No valid stress data found for {item}")

    if not all_experiments:
        print("No experiments with stress data found.")
        return

    # Determine global max frames to align rows
    max_frames = max(exp[3] for exp in all_experiments)
    
    # Prepare CSV structures
    strain_columns = ["Frame"]
    strain_map = {}
    
    for exp_name, total_time, s_data, n_frames in all_experiments:
        # 1. Calculate and add specific Time column for this experiment
        time_col_name = f"exp {exp_name} Time"
        padded_time = [np.nan] * max_frames
        
        if total_time is not None and n_frames > 1:
            dt = total_time / (n_frames - 1)
            for i in range(n_frames):
                padded_time[i] = i * dt
        
        strain_map[time_col_name] = padded_time
        if time_col_name not in strain_columns:
            strain_columns.append(time_col_name)

        # 2. Process Strains
        for region_name, values in s_data.items():
            col_name = f"exp {exp_name} {region_name}"
            padded = values + [np.nan] * (max_frames - len(values))
            strain_map[col_name] = padded
            if col_name not in strain_columns:
                strain_columns.append(col_name)

    # Generate master Frame list
    frames = list(range(max_frames))
    
    # --- Write Strain CSV ---
    suffix = "_scaled" if APPLY_CUSTOM_SCALE else ""
    strain_csv_path = os.path.join(DATA_DIR, f"averaged_strains_regions{suffix}.csv")
    print(f"Writing {strain_csv_path}...")
    with open(strain_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(strain_columns)
        
        for i in range(max_frames):
            row = [frames[i]]
            for col in strain_columns[1:]: # skip Frame
                row.append(strain_map[col][i])
            writer.writerow(row)

    print("Aggregation complete.")

if __name__ == "__main__":
    aggregate_data()
