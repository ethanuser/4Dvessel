#!/usr/bin/env python3
"""
Region-specific stress analysis script using centralized configuration.
This script allows users to select regions using rectangle selection and analyze
average stresses of edges connected to vertices in that region.
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
from analysis.stress_analysis_region import run_stress_analysis_region  # type: ignore

def main():
    """
    Main function to run region-specific stress analysis.
    """
    print("="*60)
    print("🔬 REGION-SPECIFIC STRESS ANALYSIS")
    print("="*60)
    print("This script allows you to select regions using rectangle selection")
    print("and analyze average stresses of edges connected to vertices in that region.")
    print("="*60)
    
    # Load configuration
    try:
        config_manager = ConfigManager()
        config_manager.print_summary()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Run region-specific stress analysis
    run_stress_analysis_region(config_manager)
    
    print("\n" + "="*60)
    print("🎉 REGION-SPECIFIC STRESS ANALYSIS COMPLETE!")
    print("="*60)
    print("The analysis has been completed successfully.")
    print("Check the output directory for results.")
    print("="*60)

if __name__ == "__main__":
    main()