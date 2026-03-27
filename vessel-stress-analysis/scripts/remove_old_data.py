import os
import shutil
from pathlib import Path
import glob

# ============================================================================
# CLEANUP CONFIGURATION
# ============================================================================
# Define experiment names to clean up.
# Use ["all"] to process all experiments in data/processed.
# 
# Examples:
#   EXPERIMENTS_TO_CLEAN = ["a1", "a2", "a3"]
#   EXPERIMENTS_TO_CLEAN = [f"a{i}" for i in range(1, 14)]
#   EXPERIMENTS_TO_CLEAN = ["all"]
# ============================================================================
EXPERIMENTS_TO_CLEAN = [f"a{i}" for i in range(1, 19)]

# ============================================================================
# CLEANUP SETTINGS
# ============================================================================
# MODE: "stress", "displacements", or "both"
CLEANUP_MODE = "both"

# DRY_RUN: If True, only prints what would be deleted without actually deleting.
DRY_RUN = False
# ============================================================================


def remove_old_data(experiments, mode="both", dry_run=True):
    """
    Remove old renders and gifs from specified experiments.
    
    Args:
        experiments: List of experiment names or ['all']
        mode: 'stress', 'displacements', or 'both'
        dry_run: If True, only print what would be deleted
    """
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    
    if not processed_dir.exists():
        print(f"Error: Processed directory not found at {processed_dir}")
        return

    # Determine which experiments to process
    if experiments == ['all']:
        target_experiments = [d.name for d in processed_dir.iterdir() if d.is_dir() and ((d / "stress_analysis").exists() or (d / "displacements_analysis").exists())]
    else:
        target_experiments = experiments

    analysis_types = []
    if mode in ["stress", "both"]:
        analysis_types.append("stress_analysis")
    if mode in ["displacements", "both"]:
        analysis_types.append("displacements_analysis")

    print(f"{'DRY RUN: ' if dry_run else ''}Processing {len(target_experiments)} experiments...")
    
    total_deleted_size = 0
    folders_count = 0
    gifs_count = 0

    for exp_name in target_experiments:
        exp_dir = processed_dir / exp_name
        
        if not exp_dir.exists():
            print(f"\n⚠️  Experiment directory not found: {exp_name}")
            continue

        print(f"\nChecking experiment: {exp_name}")

        for analysis_type in analysis_types:
            analysis_dir = exp_dir / analysis_type
            if not analysis_dir.exists():
                continue
            
            print(f"  Analyzing {analysis_type}...")
            
            # Find all renders folders and gifs
            all_renders_folders = sorted(glob.glob(str(analysis_dir / "renders_*")))
            all_renders_folders = [f for f in all_renders_folders if os.path.isdir(f)]
            
            all_gifs = sorted(glob.glob(str(analysis_dir / "renders_*.gif")))

            # Keep only the latest folder
            to_delete_folders = []
            if len(all_renders_folders) > 1:
                to_delete_folders = all_renders_folders[:-1]
                print(f"    Found {len(all_renders_folders)} renders folders. Keeping latest: {os.path.basename(all_renders_folders[-1])}")
            elif len(all_renders_folders) == 1:
                print(f"    Keeping only renders folder: {os.path.basename(all_renders_folders[0])}")
            
            # Keep only the latest GIF
            to_delete_gifs = []
            if len(all_gifs) > 1:
                to_delete_gifs = all_gifs[:-1]
                print(f"    Found {len(all_gifs)} GIFs. Keeping latest: {os.path.basename(all_gifs[-1])}")
            elif len(all_gifs) == 1:
                print(f"    Keeping only GIF: {os.path.basename(all_gifs[0])}")

            # Delete old folders
            for folder in to_delete_folders:
                size = get_dir_size(folder)
                total_deleted_size += size
                folders_count += 1
                if dry_run:
                    print(f"    [DRY RUN] Would delete folder: {os.path.basename(folder)} ({size / (1024*1024):.2f} MB)")
                else:
                    print(f"    Deleting folder: {os.path.basename(folder)}...")
                    shutil.rmtree(folder)

            # Delete old gifs
            for gif in to_delete_gifs:
                size = os.path.getsize(gif)
                total_deleted_size += size
                gifs_count += 1
                if dry_run:
                    print(f"    [DRY RUN] Would delete GIF: {os.path.basename(gif)} ({size / (1024*1024):.2f} MB)")
                else:
                    print(f"    Deleting GIF: {os.path.basename(gif)}...")
                    os.remove(gif)

    print(f"\n{'='*50}")
    print(f"{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Folders to delete: {folders_count}")
    print(f"  GIFs to delete:    {gifs_count}")
    print(f"  Total space saved: {total_deleted_size / (1024*1024):.2f} MB")
    print(f"{'='*50}")
    if dry_run:
        print("Set DRY_RUN = False at the top of the file to actually delete the files.")

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

if __name__ == "__main__":
    remove_old_data(EXPERIMENTS_TO_CLEAN, mode=CLEANUP_MODE, dry_run=DRY_RUN)
