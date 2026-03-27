#!/usr/bin/env python3
"""
Calculate maximum average stresses from region-specific stress analysis numpy files.

This script reads the numpy files produced by stress_analysis_region.py and calculates
the maximum average stress for each region in each experiment.
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
# The script will look for calculated_stresses_*.npy in each experiment's stress_region_analysis folder.
# If empty, the script will prompt for which experiments to analyze.
# 
# Examples:
#   EXPERIMENTS_TO_ANALYZE = ["a1", "a2", "a3"]
#   EXPERIMENTS_TO_ANALYZE = [f"a{i}" for i in range(1, 14)]
# 
# Set to [] for manual entry
# ============================================================================

EXPERIMENTS_TO_ANALYZE = [f"a{i}" for i in range(1, 19)]  # a1 through a18

# Decimal places for stress values in output files and display
DECIMAL_PLACES = 6

# ============================================================================

def find_region_stress_files(experiment_name, base_data_dir="data/processed"):
    """
    Find all region-specific stress analysis numpy files for a given experiment.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'a1', 'a2')
        base_data_dir: Base directory for processed data
        
    Returns:
        List of tuples (file_path, region_name)
    """
    stress_region_dir = os.path.join(base_data_dir, experiment_name, "stress_region_analysis")
    
    if not os.path.exists(stress_region_dir):
        return []
    
    # Find all calculated_stresses_*.npy files in the stress_region_analysis directory
    pattern = os.path.join(stress_region_dir, "calculated_stresses_*.npy")
    npy_files = glob.glob(pattern)
    
    if not npy_files:
        return []
    
    # Extract region names from file names
    results = []
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        # Extract region name from "calculated_stresses_{region_name}.npy"
        if filename.startswith("calculated_stresses_") and filename.endswith(".npy"):
            region_name = filename[len("calculated_stresses_"):-4]  # Remove prefix and .npy
            # Convert underscores back to spaces for display
            region_display = region_name.replace('_', ' ')
            results.append((file_path, region_display, region_name))
    
    return results

def calculate_max_avg_stress(stress_file_path):
    """
    Calculate the maximum average stress from a stress analysis numpy file.
    
    Args:
        stress_file_path: Path to the calculated_stresses_*.npy file
        
    Returns:
        Tuple of (max_avg_stress_Pa, max_avg_stress_MPa, time_index) or None if error
    """
    try:
        # Load the stress data
        # Format: List of arrays, where each array contains stress values for all edges at a time point
        all_stresses = np.load(stress_file_path, allow_pickle=True)
        
        if len(all_stresses) == 0:
            return None
        
        # Calculate average stress for each time point (across all edges)
        average_stresses = [np.mean(stress) for stress in all_stresses if len(stress) > 0]
        
        if len(average_stresses) == 0:
            return None
        
        # Find the maximum average stress
        max_avg_stress_pa = np.max(average_stresses)
        max_avg_stress_mpa = max_avg_stress_pa / 1e6  # Convert to MPa
        
        # Find which time point had the max
        max_time_index = np.argmax(average_stresses)
        
        return max_avg_stress_pa, max_avg_stress_mpa, max_time_index
        
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
    
    # Create a lookup: (experiment, region_display) -> max_stress_mpa
    data_map = {}
    regions = set()
    for r in all_results:
        # Use lowercase experiment name for lookup
        data_map[(r['experiment'].lower(), r['region_display'])] = r['max_stress_mpa']
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
    """Main function to process experiments and calculate max stresses for regions."""
    
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
    print(f"CALCULATING MAX STRESSES FOR REGIONS IN {len(experiments)} EXPERIMENTS")
    print("="*80)
    print()
    
    all_results = []
    
    for exp_name in experiments:
        print(f"Processing {exp_name}...")
        
        # Find region stress files for this experiment
        region_files = find_region_stress_files(exp_name, str(project_root / "data/processed"))
        
        if not region_files:
            print(f"  ⚠️  No region stress files found for {exp_name}")
            print()
            continue
        
        print(f"  Found {len(region_files)} region(s)")
        
        for file_path, region_display, region_name in region_files:
            result = calculate_max_avg_stress(file_path)
            
            if result:
                max_stress_pa, max_stress_mpa, time_idx = result
                all_results.append({
                    'experiment': exp_name,
                    'region_display': region_display,
                    'region_name': region_name,
                    'max_stress_pa': max_stress_pa,
                    'max_stress_mpa': max_stress_mpa,
                    'time_index': time_idx,
                    'file_path': file_path
                })
                print(f"    ✓ {region_display}: {max_stress_mpa:.{DECIMAL_PLACES}f} MPa (at frame {time_idx})")
            else:
                print(f"    ⚠️  Failed to process: {region_display}")
        
        print()
    
    if all_results:
        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY: Maximum Average Stresses by Region")
        print("="*80)
        print(f"{'Experiment':<15} {'Region':<30} {'Max Stress (MPa)':<20} {'Frame':<10}")
        print("-"*80)
        
        for r in all_results:
            print(f"{r['experiment']:<15} {r['region_display']:<30} {r['max_stress_mpa']:<20.{DECIMAL_PLACES}f} {r['time_index']:<10}")
        
        print("="*80)
        
        # Find overall maximum
        overall_max = max(all_results, key=lambda x: x['max_stress_mpa'])
        print(f"\nOverall Maximum: {overall_max['experiment']} - {overall_max['region_display']} with {overall_max['max_stress_mpa']:.{DECIMAL_PLACES}f} MPa")
        
        # Group by experiment and find max per experiment
        print("\n" + "="*80)
        print("Maximum Stress Per Experiment (across all regions)")
        print("="*80)
        print(f"{'Experiment':<15} {'Max Region':<30} {'Max Stress (MPa)':<20}")
        print("-"*80)
        
        experiments_dict = {}
        for r in all_results:
            exp = r['experiment']
            if exp not in experiments_dict or r['max_stress_mpa'] > experiments_dict[exp]['max_stress_mpa']:
                experiments_dict[exp] = r
        
        for exp in sorted(experiments_dict.keys()):
            r = experiments_dict[exp]
            print(f"{exp:<15} {r['region_display']:<30} {r['max_stress_mpa']:<20.{DECIMAL_PLACES}f}")
        
        print("="*80)
        
        # Optional: Save results to file
        output_dir = project_root / "data" / "processed" / "max_region_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = output_dir / "max_stresses_regions_summary.txt"
        with open(output_file, 'w') as f:
            f.write("Maximum Average Stresses by Region - Summary\n")
            f.write("="*80 + "\n\n")
            
            f.write("All Regions:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Experiment':<15} {'Region':<30} {'Max Stress (MPa)':<20} {'Frame':<10}\n")
            f.write("-"*80 + "\n")
            for r in all_results:
                f.write(f"{r['experiment']:<15} {r['region_display']:<30} {r['max_stress_mpa']:<20.{DECIMAL_PLACES}f} {r['time_index']:<10}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Overall Maximum: {overall_max['experiment']} - {overall_max['region_display']} with {overall_max['max_stress_mpa']:.{DECIMAL_PLACES}f} MPa\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Maximum Stress Per Experiment (across all regions)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Experiment':<15} {'Max Region':<30} {'Max Stress (MPa)':<20}\n")
            f.write("-"*80 + "\n")
            for exp in sorted(experiments_dict.keys()):
                r = experiments_dict[exp]
                f.write(f"{exp:<15} {r['region_display']:<30} {r['max_stress_mpa']:<20.{DECIMAL_PLACES}f}\n")
        
        print(f"\nResults saved to: {output_file}")
        
        # Generate the formatted pivot CSV
        csv_output_file = output_dir / "max_stresses_regions_pivot.csv"
        generate_pivot_csv(all_results, csv_output_file)
    else:
        print("\n⚠️  No results to display. Check that region stress analysis files exist.")
        print("Run stress_analysis_region.py first to create region-specific stress files.")

if __name__ == "__main__":
    main()

