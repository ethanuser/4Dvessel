#!/usr/bin/env python3
"""
Calculate maximum average displacements from region-specific displacement analysis numpy files.

This script reads the numpy files produced by displacement_analysis.py and calculates
the maximum average displacement for each region in each experiment.
"""

import numpy as np
import os
import glob
import csv
from pathlib import Path

# ============================================================================
# EXPERIMENT QUEUE CONFIGURATION
# ============================================================================
# Define experiment names to analyze.
# The script will look for calculated_displacements_*.npy in each experiment's displacement_region_analysis folder.
# If empty, the script will prompt for which experiments to analyze.
# 
# Examples:
#   EXPERIMENTS_TO_ANALYZE = ["a1", "a2", "a3"]
#   EXPERIMENTS_TO_ANALYZE = [f"a{i}" for i in range(1, 14)]
# 
# Set to [] for manual entry
# ============================================================================

EXPERIMENTS_TO_ANALYZE = [f"a{i}" for i in range(1, 19)]  # a1 through a18

# Decimal places for displacement values in output files and display
DECIMAL_PLACES = 6

# ============================================================================

def find_region_displacement_files(experiment_name, base_data_dir="data/processed"):
    """
    Find all region-specific displacement analysis numpy files for a given experiment.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'a1', 'a2')
        base_data_dir: Base directory for processed data
        
    Returns:
        List of tuples (file_path, region_display, region_name)
    """
    displacement_region_dir = os.path.join(base_data_dir, experiment_name, "displacement_region_analysis")
    
    if not os.path.exists(displacement_region_dir):
        return []
    
    # Find all calculated_displacements_*.npy files in the displacement_region_analysis directory
    # But exclude _indices.npy files
    pattern = os.path.join(displacement_region_dir, "calculated_displacements_*.npy")
    all_npy_files = glob.glob(pattern)
    
    npy_files = [f for f in all_npy_files if not f.endswith("_indices.npy")]
    
    if not npy_files:
        return []
    
    # Extract region names from file names
    results = []
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        # Extract region name from "calculated_displacements_{region_name}.npy"
        if filename.startswith("calculated_displacements_") and filename.endswith(".npy"):
            region_name = filename[len("calculated_displacements_"):-4]  # Remove prefix and .npy
            # Convert underscores back to spaces for display
            region_display = region_name.replace('_', ' ')
            results.append((file_path, region_display, region_name))
    
    return results

def calculate_max_avg_displacement(displacement_file_path):
    """
    Calculate the maximum average displacement from a displacement analysis numpy file.
    
    Args:
        displacement_file_path: Path to the calculated_displacements_*.npy file
        
    Returns:
        Tuple of (max_avg_displacement_mm, max_time_index) or None if error
    """
    try:
        # Load the displacement data
        # Format: (num_points, num_frames) array
        disp_data = np.load(displacement_file_path, allow_pickle=True)
        
        if disp_data.size == 0:
            return None
        
        # If it's a 2D array (points, frames)
        if len(disp_data.shape) == 2:
            # Calculate mean across points for each frame (axis 0)
            average_displacements = np.nanmean(disp_data, axis=0)
        else:
            # Fallback for other potential formats
            average_displacements = disp_data
            
        if len(average_displacements) == 0:
            return None
        
        # Find the maximum average displacement
        max_avg_displacement_mm = np.nanmax(average_displacements)
        
        # Find which time point had the max
        max_time_index = np.nanargmax(average_displacements)
        
        return max_avg_displacement_mm, max_time_index
        
    except Exception as e:
        print(f"  ⚠️  ERROR reading file: {e}")
        return None

def generate_pivot_csv(all_results, output_path):
    """
    Generate a pivoted CSV file for the 3x6 experiment layout.
    
    Layout definition:
    Row 1: Cervical (a1, a2, a3, a16, a17, a18)
    Row 2: Terminal (a4, a5, a6, a13, a14, a15)
    Row 3: Cavernous (a7, a8, a9, a10, a11, a12)
    
    Columns are grouped vertically: (a1,4,7), (a2,5,8), etc.
    """
    # Mapping of trials to their experiments (in order of the columns' lists)
    TRIALS = ["Cervical", "Terminal", "Cavernous"]
    
    # Columns and their corresponding experiment lists (order matches TRIALS)
    COLUMN_GROUPS = [
        ("a1,4,7", ["a1", "a4", "a7"]),
        ("a2,5,8", ["a2", "a5", "a8"]),
        ("a3,6,9", ["a3", "a6", "a9"]),
        ("a16,13,10", ["a16", "a13", "a10"]),
        ("a17,14,11", ["a17", "a14", "a11"]),
        ("a18,15,12", ["a18", "a15", "a12"])
    ]
    
    # Create a lookup: (experiment, region_display) -> max_displacement_mm
    data_map = {}
    regions = set()
    for r in all_results:
        # Use lowercase experiment name for lookup
        data_map[(r['experiment'].lower(), r['region_display'])] = r['max_displacement_mm']
        regions.add(r['region_display'])
    
    # Try to sort regions logically (Convex, Bifurcation, Caudal)
    def region_sort_key(name):
        name_lower = name.lower()
        if 'convex' in name_lower: return 0
        if 'bifurcation' in name_lower: return 1
        if 'caudal' in name_lower: return 2
        return 3
    
    sorted_regions = sorted(list(regions), key=region_sort_key)
    
    if not sorted_regions:
        print("  ⚠️  No regions found for CSV generation.")
        return

    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["Trial", "Regions"] + [col[0] for col in COLUMN_GROUPS] + ["Average"]
            writer.writerow(header)
            
            for trial_idx, trial_name in enumerate(TRIALS):
                for region_idx, region in enumerate(sorted_regions):
                    # Only show trial name on the first region row for that trial
                    row = [trial_name if region_idx == 0 else "", region]
                    values = []
                    
                    for _, exp_list in COLUMN_GROUPS:
                        exp = exp_list[trial_idx]
                        val = data_map.get((exp.lower(), region), None)
                        if val is not None:
                            row.append(f"{val:.{DECIMAL_PLACES}f}")
                            values.append(val)
                        else:
                            row.append("")
                    
                    # Row-wise Average
                    if values:
                        row.append(f"{np.mean(values):.{DECIMAL_PLACES}f}")
                    else:
                        row.append("")
                    
                    writer.writerow(row)
        print(f"✓ Summary CSV saved to: {output_path}")
    except Exception as e:
        print(f"  ⚠️  ERROR writing CSV: {e}")

def main():
    """Main function to process experiments and calculate max displacements for regions."""
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if not EXPERIMENTS_TO_ANALYZE:
        # Manual entry mode
        print("No experiments specified in EXPERIMENTS_TO_ANALYZE.")
        print("Enter experiment names (comma-separated, e.g., a1,a2,a3):")
        user_input = input("> ").strip()
        experiments = [exp.strip() for exp in user_input.split(",") if exp.strip()]
        
        if not experiments:
            print("No experiments entered. Exiting.")
            return
    else:
        experiments = EXPERIMENTS_TO_ANALYZE
    
    print("="*80)
    print(f"CALCULATING MAX DISPLACEMENTS FOR REGIONS IN {len(experiments)} EXPERIMENTS")
    print("="*80)
    print()
    
    all_results = []
    
    for exp_name in experiments:
        print(f"Processing {exp_name}...")
        
        # Find region displacement files for this experiment
        region_files = find_region_displacement_files(exp_name, str(project_root / "data/processed"))
        
        if not region_files:
            print(f"  ⚠️  No region displacement files found for {exp_name}")
            print()
            continue
        
        print(f"  Found {len(region_files)} region(s)")
        
        for file_path, region_display, region_name in region_files:
            result = calculate_max_avg_displacement(file_path)
            
            if result:
                max_disp_mm, time_idx = result
                all_results.append({
                    'experiment': exp_name,
                    'region_display': region_display,
                    'region_name': region_name,
                    'max_displacement_mm': max_disp_mm,
                    'time_index': time_idx,
                    'file_path': file_path
                })
                print(f"    ✓ {region_display}: {max_disp_mm:.{DECIMAL_PLACES}f} mm (at frame {time_idx})")
            else:
                print(f"    ⚠️  Failed to process: {region_display}")
        
        print()
    
    if all_results:
        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY: Maximum Average Displacements by Region")
        print("="*80)
        print(f"{'Experiment':<15} {'Region':<30} {'Max Disp (mm)':<20} {'Frame':<10}")
        print("-"*80)
        
        for r in all_results:
            print(f"{r['experiment']:<15} {r['region_display']:<30} {r['max_displacement_mm']:<20.{DECIMAL_PLACES}f} {r['time_index']:<10}")
        
        print("="*80)
        
        # Find overall maximum
        overall_max = max(all_results, key=lambda x: x['max_displacement_mm'])
        print(f"\nOverall Maximum: {overall_max['experiment']} - {overall_max['region_display']} with {overall_max['max_displacement_mm']:.{DECIMAL_PLACES}f} mm")
        
        # Group by experiment and find max per experiment
        print("\n" + "="*80)
        print("Maximum Displacement Per Experiment (across all regions)")
        print("="*80)
        print(f"{'Experiment':<15} {'Max Region':<30} {'Max Disp (mm)':<20}")
        print("-"*80)
        
        experiments_dict = {}
        for r in all_results:
            exp = r['experiment']
            if exp not in experiments_dict or r['max_displacement_mm'] > experiments_dict[exp]['max_displacement_mm']:
                experiments_dict[exp] = r
        
        for exp in sorted(experiments_dict.keys()):
            r = experiments_dict[exp]
            print(f"{exp:<15} {r['region_display']:<30} {r['max_displacement_mm']:<20.{DECIMAL_PLACES}f}")
        
        print("="*80)
        
        # Optional: Save results to file
        output_dir = project_root / "data" / "processed" / "max_region_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = output_dir / "max_displacements_regions_summary.txt"
        with open(output_file, 'w') as f:
            f.write("Maximum Average Displacements by Region - Summary\n")
            f.write("="*80 + "\n\n")
            
            f.write("All Regions:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Experiment':<15} {'Region':<30} {'Max Disp (mm)':<20} {'Frame':<10}\n")
            f.write("-"*80 + "\n")
            for r in all_results:
                f.write(f"{r['experiment']:<15} {r['region_display']:<30} {r['max_displacement_mm']:<20.{DECIMAL_PLACES}f} {r['time_index']:<10}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Overall Maximum: {overall_max['experiment']} - {overall_max['region_display']} with {overall_max['max_displacement_mm']:.6f} mm\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Maximum Displacement Per Experiment (across all regions)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Experiment':<15} {'Max Region':<30} {'Max Disp (mm)':<20}\n")
            f.write("-"*80 + "\n")
            for exp in sorted(experiments_dict.keys()):
                r = experiments_dict[exp]
                f.write(f"{exp:<15} {r['region_display']:<30} {r['max_displacement_mm']:<20.{DECIMAL_PLACES}f}\n")
        
        print(f"\nResults saved to: {output_file}")
        
        # Generate the formatted pivot CSV
        csv_output_file = output_dir / "max_displacements_regions_pivot.csv"
        generate_pivot_csv(all_results, csv_output_file)
    else:
        print("\n⚠️  No results to display. Check that region displacement analysis files exist.")
        print("Run displacement_analysis.py first to create region-specific displacement files.")

if __name__ == "__main__":
    main()
