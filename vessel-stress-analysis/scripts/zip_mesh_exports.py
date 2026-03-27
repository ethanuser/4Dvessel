import os
import shutil
import zipfile
import itertools
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
# Define experiment config paths to include. 
# If empty, the script will scan all directories in data/processed for mesh exports.
# Example: 
# EXPERIMENT_QUEUE = [
#     f"config/experiments/lattice_strength_1_25_rgd{r}.json"
#     for r in ["0","0.001","0.01","0.1","1","2","5","10","20","100","1000"]
# ]

EXPERIMENT_QUEUE = [f"config/experiments/lattice_strength_{i}.json" for i in ["0_25","0_50","0_75","1_00","1_25"]]


# EXPERIMENT_QUEUE = []

# Output settings
BUNDLE_NAME = "mesh_exports_bundle"
OUTPUT_ZIP = "mesh_exports_bundle.zip"
# ============================================================================

def zip_mesh_exports():
    project_root = Path(__file__).parent.parent
    data_processed = project_root / "data" / "processed"
    bundle_dir = project_root / BUNDLE_NAME
    
    # Create or clear bundle directory
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    found_files = []

    if EXPERIMENT_QUEUE:
        print(f"Zipping exports for {len(EXPERIMENT_QUEUE)} queued experiments...")
        for config_path in EXPERIMENT_QUEUE:
            # Extract experiment name from config filename
            exp_name = Path(config_path).stem
            exp_dir = data_processed / exp_name
            export_folder = exp_dir / "cluster_mesh_export"
            
            # Use glob to find the npy file, as it might have .json.npy extension
            export_files = list(export_folder.glob(f"cluster_export_{exp_name}*.npy"))
            
            if export_files:
                # Add all matching files (usually just one)
                found_files.extend(export_files)
            else:
                print(f"  ⚠️  Warning: Export not found for {exp_name} in {export_folder}")
    else:
        print("Scanning data/processed for all mesh exports...")
        for exp_dir in data_processed.iterdir():
            if exp_dir.is_dir():
                export_folder = exp_dir / "cluster_mesh_export"
                if export_folder.exists():
                    # Look for .npy files in this folder
                    for npy_file in export_folder.glob("cluster_export_*.npy"):
                        found_files.append(npy_file)

    if not found_files:
        print("❌ No cluster mesh export files found. Did you run run_cluster_mesh_export.py?")
        return

    print(f"Found {len(found_files)} files. Copying to bundle...")
    for f in found_files:
        shutil.copy2(f, bundle_dir / f.name)
        print(f"  + {f.name}")

    # Create ZIP
    zip_path = project_root / OUTPUT_ZIP
    print(f"\nCreating ZIP archive: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in bundle_dir.iterdir():
            zipf.write(f, f.name)

    print(f"\n✅ Success! Mesh exports zipped to: {OUTPUT_ZIP}")
    print(f"You can now download the ZIP file.")

if __name__ == "__main__":
    zip_mesh_exports()
