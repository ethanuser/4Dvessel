#!/usr/bin/env python3
"""
Export cluster positions and mesh connections over time.
Supports running a queue of experiments or interactive selection.

This script loads experiment config(s), computes cluster centroids and their
mesh connectivity over time, and writes everything into NumPy files.
"""

import sys
import itertools
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import after path setup
from utils.config_manager import ConfigManager  # type: ignore
from analysis.cluster_mesh_export import run_cluster_mesh_export

# ============================================================================
# EXPERIMENT QUEUE CONFIGURATION
# ============================================================================
# Define experiment config paths to run in sequence.
# If empty, the script will prompt for interactive config selection.
# 
# Examples:
#   EXPERIMENT_QUEUE = ["config/experiments/a1.json", "config/experiments/a2.json"]
#   EXPERIMENT_QUEUE = [f"config/experiments/a{i}.json" for i in range(1, 19)]
#   EXPERIMENT_QUEUE = ["config/experiments/synthetic_pull_real_params.json"]
# ============================================================================

# EXPERIMENT_QUEUE = [f"config/experiments/a{i}.json" for i in range(1, 19)]
# EXPERIMENT_QUEUE = [f"config/experiments/lattice_strength_{i}.json" for i in ["0_25","0_50","0_75","1_00","1_25"]]
# EXPERIMENT_QUEUE = [f"config/experiments/e{i}_rgd1.json" for i in [""]]

# EXPERIMENT_QUEUE = [
#     f"config/experiments/lattice_strength_1_25_rgd{r}.json"
#     for r in ["0","0.001","0.01","0.1","1","2","5","10","20","100","1000"]
# ]


EXPERIMENT_QUEUE = [
    f"config/experiments/a1_copy.json"
]

# EXPERIMENT_QUEUE = [f"config/experiments/bend_twist.json"]

# EXPERIMENT_QUEUE = []  # Empty list for interactive mode

# EXPERIMENT_QUEUE = [
#     "config/experiments/lattice_strength_0_25.json",
#     "config/experiments/lattice_strength_0_50.json",
#     "config/experiments/lattice_strength_0_75.json",
#     "config/experiments/lattice_strength_1_00.json",
#     "config/experiments/lattice_strength_1_25.json"
# ]

# ============================================================================

def main():
    if EXPERIMENT_QUEUE:
        # Run experiments from queue
        print("="*70)
        print(f"RUNNING EXPERIMENT QUEUE: {len(EXPERIMENT_QUEUE)} experiments")
        print("="*70)
        
        for i, config_path in enumerate(EXPERIMENT_QUEUE, 1):
            print(f"\n{'='*70}")
            print(f"EXPERIMENT {i}/{len(EXPERIMENT_QUEUE)}: {config_path}")
            print(f"{'='*70}\n")
            
            # Check if config file exists
            config_file = project_root / config_path
            if not config_file.exists():
                print(f"⚠️  ERROR: Config file not found: {config_path}")
                print(f"Skipping to next experiment...\n")
                continue
            
            try:
                # Load configuration
                config_manager = ConfigManager(str(config_file))
                config_manager.print_summary()
                
                # Run export
                run_cluster_mesh_export(config_manager)
                
                print(f"\n✓ Completed experiment {i}/{len(EXPERIMENT_QUEUE)}: {config_path}")
                
            except Exception as e:
                print(f"\n⚠️  ERROR in experiment {config_path}:")
                print(f"   {str(e)}")
                print(f"Continuing to next experiment...\n")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"QUEUE COMPLETE: Processed {len(EXPERIMENT_QUEUE)} experiments")
        print(f"{'='*70}\n")
        
    else:
        # Interactive mode: prompt for config selection
        print("No experiment queue defined. Using interactive config selection.\n")
        config_manager = ConfigManager()
        config_manager.print_summary()
        
        # Run export
        run_cluster_mesh_export(config_manager)


if __name__ == "__main__":
    main()


