#!/usr/bin/env python3
"""
Interactive raw point cloud visualization script.
Displays all raw points frame-by-frame without clustering or meshing.

Usage:
    python scripts/run_raw_points_interactive.py [data_path]

This script is fast and lightweight - just renders raw points with RGB colors.
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
from analysis.raw_points_interactive import run_raw_points_interactive  # type: ignore

# ============================================================================
# DATA FILE CONFIGURATION
# ============================================================================
# Specify the data file to visualize.
# This should be a .npy file with shape (T, N, 6) for [X,Y,Z,R,G,B]
# or (T, N, 3) for [X,Y,Z] only.
# Examples:
#   DATA_FILE = "data/raw/a1/points_colored.npy"
#   DATA_FILE = "data/raw/synthetic_pull/points_colored.npy"
# ============================================================================

DATA_FILE = r"data/raw/a1_copy/points_colored.npy"
# DATA_FILE = "data/raw/lattice_strength_1_25_rgd1/points_colored.npy"
# DATA_FILE = "data/raw/synthetic_translate_real_params/points_colored.npy"
# DATA_FILE = "data/raw/bend_twist/points_colored.npy"

# ============================================================================

def main():
    # Check if data file is provided via command line
    if len(sys.argv) >= 2:
        data_path = sys.argv[1]
    else:
        # Use constant from top of file
        data_path = DATA_FILE
    
    # Resolve path (handle both relative and absolute paths)
    if not os.path.isabs(data_path):
        # Try relative to project root first
        resolved_path = project_root / data_path
        if not resolved_path.exists():
            # Try relative to current working directory
            resolved_path = Path(data_path).resolve()
    else:
        resolved_path = Path(data_path)
    
    # Validate file exists
    if not resolved_path.exists():
        print(f"⚠️  ERROR: File not found: {data_path}")
        print(f"   Resolved path: {resolved_path}")
        print(f"\nPlease update DATA_FILE constant in the script or provide path as argument:")
        print(f"  python scripts/run_raw_points_interactive.py <data_path>")
        return
    
    if not resolved_path.suffix == '.npy':
        print(f"⚠️  WARNING: File does not have .npy extension: {resolved_path}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    try:
        # Extract experiment name from path
        experiment_name = resolved_path.parent.name if resolved_path.parent.name else resolved_path.stem
        
        # Run the interactive visualization
        run_raw_points_interactive(str(resolved_path), experiment_name=experiment_name)
        
        print(f"\n✓ Completed visualization for: {resolved_path.name}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n⚠️  ERROR during visualization:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

