#!/usr/bin/env python3
"""
Interactive mesh editor script using centralized configuration.
This script allows manual selection and editing of cluster means and edges.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.config_manager import ConfigManager # type: ignore
from analysis.mesh_editor_debug import run_mesh_editor # type: ignore

def main():
    # Load configuration
    config_manager = ConfigManager()
    config_manager.print_summary()
    
    # Run mesh editor
    run_mesh_editor(config_manager)

if __name__ == "__main__":
    main() 