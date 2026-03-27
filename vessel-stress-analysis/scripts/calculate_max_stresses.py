"""
Calculate maximum average stresses from stress analysis numpy files.

This script reads the numpy files produced by stress_analysis.py and calculates
the maximum average stress for each experiment.
"""

import numpy as np
import os
import glob
import argparse
from pathlib import Path


def find_stress_files(experiment_name, base_data_dir="data/processed"):
    """
    Find all stress analysis numpy files for a given experiment.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'a1', 'a2')
        base_data_dir: Base directory for processed data
        
    Returns:
        List of paths to numpy files
    """
    stress_dir = os.path.join(base_data_dir, experiment_name, "stress_analysis")
    
    if not os.path.exists(stress_dir):
        print(f"Warning: Stress analysis directory not found for {experiment_name}: {stress_dir}")
        return []
    
    # Find all .npy files in the stress_analysis directory
    npy_files = glob.glob(os.path.join(stress_dir, "*.npy"))
    
    if not npy_files:
        print(f"Warning: No .npy files found in {stress_dir}")
        return []
    
    return npy_files


def calculate_max_average_stress(npy_file_path):
    """
    Calculate the maximum average stress from a numpy file.
    
    The numpy file contains a list of stress arrays for each time point.
    This function calculates the average stress at each time point,
    then returns the maximum average stress.
    
    Args:
        npy_file_path: Path to the numpy file
        
    Returns:
        Tuple of (max_average_stress_Pa, max_average_stress_MPa, time_index)
    """
    try:
        # Load the stress data
        all_stresses = np.load(npy_file_path, allow_pickle=True)
        
        # Calculate average stress for each time point
        average_stresses = [np.mean(stress) for stress in all_stresses]
        
        # Find the maximum average stress and its index
        max_avg_stress_pa = np.max(average_stresses)
        max_avg_stress_mpa = max_avg_stress_pa / 1e6
        max_time_index = np.argmax(average_stresses)
        
        return max_avg_stress_pa, max_avg_stress_mpa, max_time_index
        
    except Exception as e:
        print(f"Error loading {npy_file_path}: {e}")
        return None, None, None


def calculate_max_stresses_for_experiments(experiment_names, base_data_dir="data/processed"):
    """
    Calculate max average stresses for multiple experiments.
    
    Args:
        experiment_names: List of experiment names
        base_data_dir: Base directory for processed data
        
    Returns:
        Dictionary mapping experiment names to their results
    """
    results = {}
    
    for exp_name in experiment_names:
        print(f"\n{'='*60}")
        print(f"Processing experiment: {exp_name}")
        print(f"{'='*60}")
        
        # Find stress files for this experiment
        stress_files = find_stress_files(exp_name, base_data_dir)
        
        if not stress_files:
            results[exp_name] = None
            continue
        
        print(f"Found {len(stress_files)} stress file(s)")
        
        exp_results = []
        for stress_file in stress_files:
            file_name = os.path.basename(stress_file)
            print(f"\nAnalyzing: {file_name}")
            
            max_pa, max_mpa, time_idx = calculate_max_average_stress(stress_file)
            
            if max_pa is not None:
                exp_results.append({
                    'file': stress_file,
                    'file_name': file_name,
                    'max_stress_pa': max_pa,
                    'max_stress_mpa': max_mpa,
                    'time_index': time_idx
                })
                
                print(f"  Max Average Stress: {max_mpa:.6f} MPa ({max_pa:.2e} Pa)")
                print(f"  Occurred at time index: {time_idx}")
        
        results[exp_name] = exp_results
    
    return results


def print_summary(results):
    """
    Print a summary table of all results.
    
    Args:
        results: Dictionary of results from calculate_max_stresses_for_experiments
    """
    print("\n" + "="*80)
    print("SUMMARY: Maximum Average Stresses Across All Experiments")
    print("="*80)
    print(f"{'Experiment':<15} {'File':<30} {'Max Stress (MPa)':<20} {'Time Index':<12}")
    print("-"*80)
    
    for exp_name, exp_results in results.items():
        if exp_results is None:
            print(f"{exp_name:<15} {'N/A':<30} {'N/A':<20} {'N/A':<12}")
        else:
            for i, result in enumerate(exp_results):
                exp_display = exp_name if i == 0 else ""
                file_display = result['file_name'][:28] + ".." if len(result['file_name']) > 30 else result['file_name']
                print(f"{exp_display:<15} {file_display:<30} {result['max_stress_mpa']:<20.6f} {result['time_index']:<12}")
    
    print("="*80)


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Calculate maximum average stresses from stress analysis numpy files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific experiments
  python scripts/calculate_max_stresses.py --experiments a1 a2 a3
  
  # Analyze experiments interactively
  python scripts/calculate_max_stresses.py
  
  # Analyze all experiments in a directory
  python scripts/calculate_max_stresses.py --all
  
  # Specify custom data directory
  python scripts/calculate_max_stresses.py --experiments a1 --data-dir /path/to/data
        """
    )
    
    parser.add_argument(
        '--experiments', '-e',
        nargs='+',
        help='Names of experiments to analyze (e.g., a1 a2 a3)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Analyze all experiments in the processed data directory'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        default='data/processed',
        help='Base directory for processed data (default: data/processed)'
    )
    
    args = parser.parse_args()
    
    # Determine which experiments to process
    experiment_names = []
    
    if args.all:
        # Find all experiment directories
        data_dir = args.data_dir
        if os.path.exists(data_dir):
            experiment_names = [
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
            experiment_names.sort()
            print(f"Found {len(experiment_names)} experiments: {', '.join(experiment_names)}")
        else:
            print(f"Error: Data directory not found: {data_dir}")
            return
    
    elif args.experiments:
        experiment_names = args.experiments
    
    else:
        # Interactive mode
        print("="*60)
        print("Calculate Maximum Average Stresses")
        print("="*60)
        print("\nEnter experiment names separated by spaces (e.g., a1 a2 a3)")
        print("Or press Enter to see available experiments:")
        
        user_input = input("\nExperiments: ").strip()
        
        if not user_input:
            # Show available experiments
            data_dir = args.data_dir
            if os.path.exists(data_dir):
                available = [
                    d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                ]
                available.sort()
                print(f"\nAvailable experiments: {', '.join(available)}")
                
                user_input = input("\nEnter experiments to analyze: ").strip()
                
                if not user_input:
                    print("No experiments specified. Exiting.")
                    return
        
        experiment_names = user_input.split()
    
    if not experiment_names:
        print("No experiments specified. Use --help for usage information.")
        return
    
    # Calculate max stresses for all specified experiments
    results = calculate_max_stresses_for_experiments(experiment_names, args.data_dir)
    
    # Print summary table
    print_summary(results)


if __name__ == "__main__":
    main()

