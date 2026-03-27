#!/usr/bin/env python3
"""
Create a GIF from images in a selected folder.

This script:
1. Opens a file dialog to select a folder containing rendered images
2. Finds all frame_*.png images in that folder
3. Creates a GIF from the images and saves it in the same folder

Usage:
    python create_gif_from_folder.py
"""

import imageio.v2 as imageio
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ============================================================================
# GIF SETTINGS
# ============================================================================
FPS = 30  # Frames per second for the GIF
OUTPUT_FILENAME = "animation.gif"  # Name of the output GIF file


def select_folder():
    """Open a file dialog to select a folder."""
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder Containing Rendered Images")
    return folder_selected


def process_folder(folder_path, fps=30, output_filename="animation.gif"):
    """
    Process a single folder and create a GIF from its images.
    
    Args:
        folder_path: Path to the folder containing frame_*.png images
        fps: Frames per second for the output GIF
        output_filename: Name of the output GIF file
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(folder_path):
        print(f"⚠️  ERROR: Folder not found: {folder_path}")
        return False
    
    # Get folder name for display
    folder_name = os.path.basename(folder_path)
    
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
        
        # Save GIF in the same folder
        output_path = os.path.join(folder_path, output_filename)
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
    print("=" * 70)
    print("GIF CREATOR - Select a folder containing frame_*.png images")
    print("=" * 70)
    print()
    
    # Prompt for folder selection
    folder = select_folder()
    if not folder:
        print("No folder selected. Exiting.")
        return
    
    print(f"Selected folder: {folder}")
    print()
    
    # Process the folder
    success = process_folder(folder, fps=FPS, output_filename=OUTPUT_FILENAME)
    
    if success:
        print("\n" + "=" * 70)
        print("✓ GIF creation complete!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠️  GIF creation failed!")
        print("=" * 70)


if __name__ == "__main__":
    main()

