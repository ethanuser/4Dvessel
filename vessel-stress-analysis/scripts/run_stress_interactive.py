#!/usr/bin/env python3
"""
Interactive stress visualization script.
Runs a single experiment specified by EXPERIMENT_CONFIG variable.
"""

import sys
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import after path setup
from utils.config_manager import ConfigManager  # type: ignore
from analysis.stress_interactive import run_fast_interactive_analysis

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
# Specify the experiment config file to run.
# Examples:
#   EXPERIMENT_CONFIG = "config/experiments/chessboard.json"
EXPERIMENT_CONFIG = "config/experiments/a1_copy.json"
# ============================================================================

# EXPERIMENT_CONFIG = "config/experiments/synthetic_pull.json"
# EXPERIMENT_CONFIG = "config/experiments/synthetic_translate_real_params.json"
# EXPERIMENT_CONFIG = "config/experiments/e4.json"

# EXPERIMENT_CONFIG = "config/experiments/bend_twist.json"


# ============================================================================

def main():
    # Check if config file exists
    config_file = project_root / EXPERIMENT_CONFIG
    if not config_file.exists():
        print(f"⚠️  ERROR: Config file not found: {EXPERIMENT_CONFIG}")
        print(f"Please update EXPERIMENT_CONFIG variable in the script.")
        return
    
    try:
        # Load configuration
        config_manager = ConfigManager(str(config_file))
        config_manager.print_summary()
        
        # Run interactive visualization
        run_fast_interactive_analysis(config_manager)
        
        print(f"\n✓ Completed: {EXPERIMENT_CONFIG}")
        
    except Exception as e:
        print(f"\n⚠️  ERROR in experiment {EXPERIMENT_CONFIG}:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
