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

def load_experiment_config(exp_name):
    """Loads the experiment configuration JSON."""
    config_path = os.path.join(CONFIG_DIR, f"{exp_name}.json")
    if not os.path.exists(config_path):
        # Specific fix for folder names that might not match config names perfectly if they were renamed
        # But assuming standard structure for now based on 'a1' -> 'a1.json'
        # Try simple name matching
        return None
        
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config for {exp_name}: {e}")
        return None

def get_experiment_data(exp_dir):
    """
    Extracts stress and displacement data for a single experiment.
    Returns:
        tuple: (exp_name, total_time, stress_data_dict, displacement_data_dict, num_frames)
    """
    exp_name = os.path.basename(exp_dir)
    config = load_experiment_config(exp_name)
    
    total_time = None
    if config and "experiment" in config:
        total_time = config["experiment"].get("total_time")
    
    # --- Load Displacements by Region ---
    disp_data = {}
    num_frames = 0
    
    # Try "displacement_analysis", "displacements_analysis", and "displacement_region_analysis" folders
    for d_folder in ["displacement_analysis", "displacements_analysis", "displacement_region_analysis"]:
        dir_path = os.path.join(exp_dir, d_folder)
        if os.path.exists(dir_path):
            npy_files = glob.glob(os.path.join(dir_path, "*.npy"))
            for npy_path in npy_files:
                fname = os.path.basename(npy_path)
                if fname.endswith("_indices.npy") or fname in ["displacements_mm.npy", "tracked_displacements_mm.npy", "cluster_means.npy"]:
                    continue
                    
                try:
                    if fname.startswith("calculated_displacements_") and fname.endswith(".npy"):
                        region_name = fname.replace("calculated_displacements_", "").replace(".npy", "")
                    else:
                        region_name = fname.replace(".npy", "")
                    d_val_data = np.load(npy_path, allow_pickle=True)
                    
                    if len(d_val_data) > 0:
                        # Standardize to numpy array
                        if not isinstance(d_val_data, np.ndarray):
                            d_val_data = np.array(d_val_data)
                            
                        # Robust averaging logic for displacement regions: (points, frames)
                        if d_val_data.ndim == 2:
                            # We want the dimension that looks like frames (~160)
                            # For displacements saved as (points, frames), axis 0 is usually small (points)
                            if d_val_data.shape[0] < d_val_data.shape[1]:
                                values = np.nanmean(d_val_data, axis=0).tolist()
                            else:
                                values = np.nanmean(d_val_data, axis=1).tolist()
                        elif d_val_data.ndim == 1:
                            values = d_val_data.tolist()
                        else:
                            values = []
                            for val in d_val_data:
                                if np.isscalar(val):
                                    values.append(val)
                                elif hasattr(val, '__len__'):
                                    values.append(np.nanmean(val))
                                else:
                                    values.append(np.nan)
                        
                        num_frames = max(num_frames, len(values))
                        disp_data[region_name] = values
                except Exception as e:
                    print(f"  Error loading displacement region {os.path.basename(npy_path)} for {exp_name}: {e}")

    # --- Load Stresses ---
    stress_data = {}
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
                            values = np.nanmean(s_data, axis=1).tolist()
                        else:
                            values = np.nanmean(s_data, axis=0).tolist()
                    elif s_data.ndim == 1:
                        values = s_data.tolist()
                    else:
                        values = []
                        for val in s_data:
                            if np.isscalar(val):
                                values.append(val)
                            elif hasattr(val, '__len__'):
                                values.append(np.nanmean(val))
                            else:
                                values.append(np.nan)
                    
                    num_frames = max(num_frames, len(values))
                    stress_data[region_name] = values
            except Exception as e:
                print(f"  Error loading stress {os.path.basename(npy_path)} for {exp_name}: {e}")

    return exp_name, total_time, stress_data, disp_data, num_frames

def aggregate_data():
    """Aggregates data from all experiments and writes to CSVs."""
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found: {DATA_DIR}")
        return

    all_experiments = []
    
    # 1. Collect Data
    print("Collecting data from experiments a1 to a18...")
    
    # Generate list of experiments a1 through a18
    target_experiments = [f"a{i}" for i in range(1, 19)]
    
    for item in target_experiments:
        item_path = os.path.join(DATA_DIR, item)
        if os.path.exists(item_path) and os.path.isdir(item_path):
            print(f"  Processing {item}...")
            exp_data = get_experiment_data(item_path)
            if exp_data[4] > 0: # num_frames > 0
                all_experiments.append(exp_data)
            else:
                print(f"    No valid data found for {item}")
        else:
            print(f"    Experiment folder {item} not found in {DATA_DIR}")

    if not all_experiments:
        print("No valid experiments found.")
        return

    # Determine global max frames to align rows
    max_frames = max(exp[4] for exp in all_experiments)
    
    # Prepare CSV structures
    # Columns: Frame, [Exp1 Time], [Exp1 Region1], [Exp1 Region2], [Exp2 Time], [Exp2 Region1]...
    stress_columns = ["Frame"]
    stress_map = {}
    
    disp_columns = ["Frame"]
    disp_map = {}
    
    for exp_name, total_time, s_data, d_data, n_frames in all_experiments:
        
        # 1. Calculate and add specific Time column for this experiment
        time_col_name = f"exp {exp_name} Time"
        padded_time = [np.nan] * max_frames
        
        if total_time is not None and n_frames > 1:
            dt = total_time / (n_frames - 1)
            for i in range(n_frames):
                padded_time[i] = i * dt
        
        stress_map[time_col_name] = padded_time
        disp_map[time_col_name] = padded_time
        
        if time_col_name not in stress_columns:
            stress_columns.append(time_col_name)
        if time_col_name not in disp_columns:
            disp_columns.append(time_col_name)

        # 2. Process Stresses
        for region_name, values in s_data.items():
            col_name = f"exp {exp_name} {region_name}"
            padded = values + [np.nan] * (max_frames - len(values))
            stress_map[col_name] = padded
            if col_name not in stress_columns:
                stress_columns.append(col_name)

        # 3. Process Displacements
        for region_name, values in d_data.items():
            col_name = f"exp {exp_name} {region_name}"
            padded = values + [np.nan] * (max_frames - len(values))
            disp_map[col_name] = padded
            if col_name not in disp_columns:
                disp_columns.append(col_name)

    # Generate master Frame list
    frames = list(range(max_frames))
    
    # --- Write Stress CSV ---
    stress_csv_path = os.path.join(DATA_DIR, "averaged_stresses_regions.csv")
    print(f"Writing {stress_csv_path}...")
    with open(stress_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(stress_columns)
        
        for i in range(max_frames):
            row = [frames[i]]
            for col in stress_columns[1:]: # skip Frame
                row.append(stress_map[col][i])
            writer.writerow(row)

    # --- Write Displacement CSV ---
    disp_csv_path = os.path.join(DATA_DIR, "averaged_displacements_regions.csv")
    print(f"Writing {disp_csv_path}...")
    with open(disp_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(disp_columns)
        
        for i in range(max_frames):
            row = [frames[i]]
            for col in disp_columns[1:]:
                row.append(disp_map[col][i])
            writer.writerow(row)

    print("Aggregation complete.")

if __name__ == "__main__":
    aggregate_data()

