#!/usr/bin/env python3
"""
Displacement analysis script using centralized configuration.
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
from analysis.displacement_analysis import run_displacement_analysis  # type: ignore

def main():
    # Load configuration
    config_manager = ConfigManager()
    config_manager.print_summary()
    
    # Run displacement analysis
    run_displacement_analysis(config_manager)

if __name__ == "__main__":
    main()