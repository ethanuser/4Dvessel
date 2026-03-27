import imageio.v2 as imageio
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import glob

# ============================================================================
# FOLDER QUEUE CONFIGURATION
# ============================================================================
# Define experiment names to process.
# The script will automatically find the latest renders_* folder for each.
# If empty, the script will prompt for interactive folder selection.
# 
# Examples:
#   EXPERIMENTS_TO_PROCESS = ["a1", "a2", "a3"]
#   EXPERIMENTS_TO_PROCESS = [f"a{i}" for i in range(1, 14)]
# 
# Set to [] for interactive mode
# ============================================================================

USE_DISPLACEMENTS = True

EXPERIMENTS_TO_PROCESS = [f"a{i}" for i in range(1, 19)]  # a1 through a18
# EXPERIMENTS_TO_PROCESS = [f"a{i}" for i in [3,7,8,9,15,17]]  # a1 through a18
# EXPERIMENTS_TO_PROCESS = [f"a7"]  # a1 through a18
# EXPERIMENTS_TO_PROCESS = [f"synthetic_translate_real_params"]  # a1 through a18


    
# ============================================================================

# Auto-populate FOLDER_QUEUE by finding latest renders_* folder for each experiment
FOLDER_QUEUE = []
if EXPERIMENTS_TO_PROCESS:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    for exp_name in EXPERIMENTS_TO_PROCESS:
        # Find all renders_* folders for this experiment
        if USE_DISPLACEMENTS:
            pattern = str(project_root / f"data/processed/{exp_name}/displacements_analysis/renders_*")
        else:
            pattern = str(project_root / f"data/processed/{exp_name}/stress_analysis/renders_*")
        matching_folders = glob.glob(pattern)
            
        if matching_folders:
            # Get the most recent folder (sorted by name, which includes timestamp)
            latest_folder = max(matching_folders)
            # Convert to relative path
            relative_path = os.path.relpath(latest_folder, project_root)
            FOLDER_QUEUE.append(relative_path)
        else:
            print(f"⚠️  Warning: No renders folder found for experiment {exp_name}")

# GIF Settings
FPS = 30  # Frames per second for the GIF

# ============================================================================

def select_folder():
    """Open a file dialog to select a folder."""
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder Containing Rendered Images")
    return folder_selected

def process_folder(folder_path, fps=30):
    """
    Process a single folder and create a GIF from its images.
    
    Args:
        folder_path: Path to the folder containing frame_*.png images
        fps: Frames per second for the output GIF
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(folder_path):
        print(f"⚠️  ERROR: Folder not found: {folder_path}")
        return False
    
    # Get folder name for the output filename
    folder_name = os.path.basename(folder_path)
    output_name = f"{folder_name}.gif"

    # Collect and sort image filenames
    file_names = sorted(
        [f for f in os.listdir(folder_path) if f.startswith('frame_') and f.endswith('.png')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    if not file_names:
        print(f"⚠️  No images found with pattern 'frame_*.png' in: {folder_path}")
        return False

    images = []
    total_files = len(file_names)
    print(f"Processing {total_files} images from {folder_name}...")
    
    try:
        for i, fname in enumerate(file_names, 1):
            img_path = os.path.join(folder_path, fname)
            images.append(imageio.imread(img_path))
            # Print progress bar
            progress = int((i / total_files) * 50)  # 50 character bar
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {i}/{total_files}", end="", flush=True)
        
        print()  # New line after progress bar
        
        # Save GIF one directory level above the selected folder
        parent_dir = os.path.dirname(folder_path)
        output_path = os.path.join(parent_dir, output_name)
        print(f"Creating GIF...")
        imageio.mimsave(output_path, images, fps=fps, loop=0)  # loop=0 means infinite loop
        print(f"✓ Saved GIF to: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n⚠️  ERROR processing folder: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the GIF creation process."""
    if FOLDER_QUEUE:
        # Process folders from queue
        print("="*70)
        print(f"PROCESSING FOLDER QUEUE: {len(FOLDER_QUEUE)} folders")
        print("="*70)
        
        # Get project root (parent of scripts directory)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        successful = 0
        failed = 0
        
        for i, folder_path in enumerate(FOLDER_QUEUE, 1):
            print(f"\n{'='*70}")
            print(f"FOLDER {i}/{len(FOLDER_QUEUE)}: {folder_path}")
            print(f"{'='*70}\n")
            
            # Convert relative path to absolute
            abs_folder_path = project_root / folder_path
            
            # Process the folder
            if process_folder(str(abs_folder_path), fps=FPS):
                successful += 1
                print(f"\n✓ Completed folder {i}/{len(FOLDER_QUEUE)}")
            else:
                failed += 1
                print(f"\n⚠️  Failed folder {i}/{len(FOLDER_QUEUE)}")
        
        print(f"\n{'='*70}")
        print(f"QUEUE COMPLETE: {successful} successful, {failed} failed")
        print(f"{'='*70}\n")
        
    else:
        # Interactive mode: prompt for folder selection
        print("No folder queue defined. Using interactive folder selection.\n")
        folder = select_folder()
        if not folder:
            print("No folder selected.")
            return
        
        process_folder(folder, fps=FPS)

# Run the script
if __name__ == "__main__":
    main()
