import json
import os
import itertools
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
# The template config to use as a base
TEMPLATE_PATH = "config/experiments/lattice_strength_1_25_rgd0.001.json"

# The parameters to vary and their values
# The keys should match the prefix in the filenames (rgd, spd, phy)
PARAM_GRID = {
    "rgd": [0, 0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 100, 1000],
}

# The block to replace in strings (the part of the name that changes)
# This will be used to find and replace the parameter string in the template
OLD_PARAM_STRING = "rgd0.001"

# ============================================================================

def format_value(v):
    """Format float to string, removing trailing .0 if present to match dir names like phy0."""
    s = str(v)
    if s.endswith(".0"):
        return s[:-2]
    return s

def copy_configs():
    # Setup paths
    project_root = Path(__file__).parent.parent
    template_file = project_root / TEMPLATE_PATH
    
    if not template_file.exists():
        print(f"Error: Template file not found: {template_file}")
        return

    # Load template content
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    # We'll work with the string directly for replacement to be safe 
    # but we'll also parse it to JSON to ensure it's valid if needed.
    
    # Generate combinations
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    
    combinations = list(itertools.product(*values))
    print(f"Generating {len(combinations)} configs...")

    configs_dir = template_file.parent
    os.makedirs(configs_dir, exist_ok=True)

    for combo in combinations:
        # Create new param string (e.g., rgd0.1_spd0.01_phy0.1)
        # We'll use the same order as in the template
        new_params = []
        for k, v in zip(keys, combo):
            new_params.append(f"{k}{format_value(v)}")
        
        new_param_string = "_".join(new_params)
        
        # Skip if it's the template itself (optional, but cleaner)
        if new_param_string == OLD_PARAM_STRING:
            print(f"Skipping template matching config: {new_param_string}")
            continue

        # Create new content by replacing the old param string
        new_content = template_content.replace(OLD_PARAM_STRING, new_param_string)
        
        # Define new filename
        template_filename = template_file.name
        new_filename = template_filename.replace(OLD_PARAM_STRING, new_param_string)
        new_file_path = configs_dir / new_filename
        
        # Write new config
        with open(new_file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Created: {new_filename}")

if __name__ == "__main__":
    copy_configs()
