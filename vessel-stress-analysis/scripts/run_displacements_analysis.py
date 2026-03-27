#!/usr/bin/env python3
"""
Displacement analysis script using centralized configuration.
Supports running a queue of experiments or interactive selection.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import after path setup
from utils.config_manager import ConfigManager  # type: ignore
from analysis.displacements_analysis import run_displacements_analysis

# ============================================================================
# EXPERIMENT QUEUE CONFIGURATION
# ============================================================================
# Define experiment config paths to run in sequence.
# EXPERIMENT_QUEUE = [f"config/experiments/a{i}.json" for i in range(1, 19)]
# EXPERIMENT_QUEUE = [f"config/experiments/e{i}_new.json" for i in [4,5]]
EXPERIMENT_QUEUE = [f"config/experiments/a1_copy.json"]



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
                
                # Run displacement analysis
                run_displacements_analysis(config_manager)
                
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
        # Interactive mode
        print("No experiment queue defined. Using interactive config selection.\n")
        config_manager = ConfigManager()
        config_manager.print_summary()
        
        # Run displacement analysis
        run_displacements_analysis(config_manager)

if __name__ == "__main__":
    main()
