import imageio.v2 as imageio
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import glob
from PIL import Image
import numpy as np

# ============================================================================
# VIDEO CONFIGURATION
# ============================================================================
# Specify the output video format (e.g., '.mp4', '.avi', '.mov', '.mkv')
VIDEO_FORMAT = '.mp4'
# Frames per second for the output video
FPS = 30 

# Cropping/expansion settings (in pixels)
# Positive = crop that many pixels from that side.
# Negative = expand (add padding) that many pixels on that side.
CROP_LEFT = 250
CROP_RIGHT = -50
# How much to crop from the top and bottom sides
CROP_TOP = 50
CROP_BOTTOM = 50

# Overlay settings
# Path to the image to overlay (e.g., a colorbar)
OVERLAY_IMAGE_PATH = 'tools/generated_colorbars/displacement_mm.png'
# OVERLAY_IMAGE_PATH = 'tools/generated_colorbars/stress_mpa_0_15.png'

# Padding from the right edge
OVERLAY_PADDING_RIGHT = 10
# Scale of the overlay image (1.0 = original size, 0.5 = 50% size)
OVERLAY_SCALE = 0.4
# ============================================================================

# ============================================================================
# FOLDER QUEUE CONFIGURATION
# ============================================================================
# Define experiment names to process.
# The script will automatically find the latest renders_* folder for each.
# If empty, the script will prompt for interactive folder selection.
# 
# Set to [] for interactive mode
EXPERIMENTS_TO_PROCESS = [f"e5_new"]  # Example experiment
# ============================================================================

USE_DISPLACEMENTS = True

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

def select_folder():
    """Open a file dialog to select a folder."""
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder Containing Rendered Images")
    return folder_selected

def process_folder(folder_path, fps=30, extension='.mp4'):
    """
    Process a single folder and create a video from its images with an overlay.
    
    Args:
        folder_path: Path to the folder containing frame_*.png images
        fps: Frames per second for the output video
        extension: The file extension/format for the video
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(folder_path):
        print(f"⚠️  ERROR: Folder not found: {folder_path}")
        return False
    
    # Get folder name for the output filename
    folder_name = os.path.basename(folder_path)
    # Ensure extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension
    output_name = f"{folder_name}_overlay{extension}"

    # Collect and sort image filenames
    file_names = sorted(
        [f for f in os.listdir(folder_path) if f.startswith('frame_') and f.endswith('.png')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    if not file_names:
        print(f"⚠️  No images found with pattern 'frame_*.png' in: {folder_path}")
        return False

    # Load overlay image
    overlay_img = None
    if OVERLAY_IMAGE_PATH and os.path.exists(OVERLAY_IMAGE_PATH):
        overlay_img = Image.open(OVERLAY_IMAGE_PATH)
        if overlay_img.mode != 'RGBA':
            overlay_img = overlay_img.convert('RGBA')
        
        # Apply scaling if not 1.0
        if OVERLAY_SCALE != 1.0:
            new_size = (int(overlay_img.width * OVERLAY_SCALE), int(overlay_img.height * OVERLAY_SCALE))
            # Use Resampling.LANCZOS for high quality resizing (replaces deprecated ANTIALIAS)
            overlay_img = overlay_img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"✓ Scaled overlay image to {OVERLAY_SCALE*100}%: {new_size}")
            
        print(f"✓ Loaded overlay image: {OVERLAY_IMAGE_PATH}")
    elif OVERLAY_IMAGE_PATH:
        print(f"⚠️  Warning: Overlay image not found at {OVERLAY_IMAGE_PATH}")

    total_files = len(file_names)
    print(f"Processing {total_files} images from {folder_name} into {extension} with overlay...")
    
    try:
        # Save video one directory level above the selected folder
        parent_dir = os.path.dirname(folder_path)
        output_path = os.path.join(parent_dir, output_name)
        
        # Use imageio writer for video
        print(f"Creating video at {fps} FPS...")
        writer = imageio.get_writer(output_path, fps=fps)
        
        for i, fname in enumerate(file_names, 1):
            img_path = os.path.join(folder_path, fname)
            # Read image
            image_np = imageio.imread(img_path)
            
            # Apply cropping and/or expansion if specified
            # Positive crop value = remove that many pixels from that side.
            # Negative crop value = expand (add padding) that many pixels on that side.
            pad_left = -CROP_LEFT if CROP_LEFT < 0 else 0
            crop_left = CROP_LEFT if CROP_LEFT > 0 else 0
            pad_right = -CROP_RIGHT if CROP_RIGHT < 0 else 0
            crop_right = CROP_RIGHT if CROP_RIGHT > 0 else 0
            pad_top = -CROP_TOP if CROP_TOP < 0 else 0
            crop_top = CROP_TOP if CROP_TOP > 0 else 0
            pad_bottom = -CROP_BOTTOM if CROP_BOTTOM < 0 else 0
            crop_bottom = CROP_BOTTOM if CROP_BOTTOM > 0 else 0

            if crop_left or crop_right or crop_top or crop_bottom or pad_left or pad_right or pad_top or pad_bottom:
                h, w = image_np.shape[:2]
                left_idx = min(crop_left, w - 1)
                right_idx = max(w - crop_right, left_idx + 1)
                top_idx = min(crop_top, h - 1)
                bottom_idx = max(h - crop_bottom, top_idx + 1)
                image_np = image_np[top_idx:bottom_idx, left_idx:right_idx, :]
                if pad_left or pad_right or pad_top or pad_bottom:
                    pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
                    # Use edge replication so expanded areas and corners match the image border
                    image_np = np.pad(image_np, pad_width, mode='edge')
            
            # Apply overlay
            if overlay_img:
                # Convert base image to PIL
                base_img = Image.fromarray(image_np)
                if base_img.mode != 'RGBA':
                    base_img = base_img.convert('RGBA')
                
                base_w, base_h = base_img.size
                ov_w, ov_h = overlay_img.size
                
                # Calculate position: Centered vertically, Aligned right
                # Position = (x, y)
                x_pos = base_w - ov_w - OVERLAY_PADDING_RIGHT
                y_pos = (base_h - ov_h) // 2
                
                # Paste overlay with alpha mask
                base_img.paste(overlay_img, (x_pos, y_pos), overlay_img)
                
                # Convert back to RGB for video storage (usually required for MP4)
                final_image = np.array(base_img.convert('RGB'))
            else:
                final_image = image_np
            
            writer.append_data(final_image)
            
            # Print progress bar
            progress = int((i / total_files) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            print(f"\r[{bar}] {i}/{total_files}", end="", flush=True)
        
        writer.close()
        print(f"\n✓ Saved video to: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n⚠️  ERROR processing folder: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the video creation process."""
    if FOLDER_QUEUE:
        # Process folders from queue
        print("="*70)
        print(f"PROCESSING FOLDER QUEUE: {len(FOLDER_QUEUE)} folders")
        print(f"OUTPUT FORMAT: {VIDEO_FORMAT}")
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
            if process_folder(str(abs_folder_path), fps=FPS, extension=VIDEO_FORMAT):
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
        
        process_folder(folder, fps=FPS, extension=VIDEO_FORMAT)

# Run the script
if __name__ == "__main__":
    main()
