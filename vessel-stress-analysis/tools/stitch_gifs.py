import imageio.v2 as imageio
import os
import glob
import math
import numpy as np
from pathlib import Path
from PIL import Image
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_DISPLACEMENTS = True
# USE_DISPLACEMENTS = False
USE_THREE_IMAGES = True
# USE_THREE_IMAGES = False


# Experiments to include in the stitched GIF
# Custom order requested:
# Row 1: 1, 2, 3, 16, 17, 18
# Row 2: 4, 5, 6, 13, 14, 15
# Row 3: 7, 8, 9, 10, 11, 12

if USE_THREE_IMAGES:
    EXPERIMENT_INDICES = [
        1, 7, 14
    ]
    FIXED_COLUMNS = 3
    DOWNSAMPLE_FACTOR = 1
else:
    EXPERIMENT_INDICES = [
        1, 2, 3, 16, 17, 18,
        4, 5, 6, 13, 14, 15,
        7, 8, 9, 10, 11, 12
    ]
    FIXED_COLUMNS = 6
    DOWNSAMPLE_FACTOR = 3

EXPERIMENTS_TO_PROCESS = [f"a{i}" for i in EXPERIMENT_INDICES]


# Cropping Configuration (pixels to remove)
CROP_LEFT = 200
CROP_RIGHT = 200

# Output filename
OUTPUT_FILENAME = f"stitched_experiments_{time.strftime('%Y%m%d_%H%M%S')}.gif"
FPS = 30
# ============================================================================

def get_latest_gif_for_experiment(exp_name, project_root):
    """
    Find the latest renders_*.gif file for a given experiment.
    """

    if USE_DISPLACEMENTS:
        pattern = str(project_root / f"data/processed/{exp_name}/displacements_analysis/renders_*.gif")
    else:
        pattern = str(project_root / f"data/processed/{exp_name}/stress_analysis/renders_*.gif")
    matching_files = glob.glob(pattern)
    
    if matching_files:
        latest_file = max(matching_files, key=os.path.getmtime)
        return latest_file
    return None

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 1. Collect Valid GIFs
    print("Searching for GIFs...")
    gif_paths = []
    
    for exp_name in EXPERIMENTS_TO_PROCESS:
        gif_path = get_latest_gif_for_experiment(exp_name, project_root)
        if gif_path:
            gif_paths.append(gif_path)
            print(f"✓ Found GIF for {exp_name}: {os.path.basename(gif_path)}")
        else:
            print(f"⚠️  No GIF found for {exp_name}")
            
    if not gif_paths:
        print("No GIFs found to stitch. Exiting.")
        return

    num_gifs = len(gif_paths)
    print(f"\nProcessing {num_gifs} GIFs...")

    # 2. Determine Grid Dimensions
    if FIXED_COLUMNS:
        cols = FIXED_COLUMNS
        rows = math.ceil(num_gifs / cols)
    else:
        cols = math.ceil(math.sqrt(num_gifs))
        rows = math.ceil(num_gifs / cols)
        
    print(f"Grid Layout: {rows} rows x {cols} columns")

    # 3. Initialize Readers
    readers = []
    print("\nInitializing readers...")
    try:
        for path in gif_paths:
            reader = imageio.get_reader(path)
            readers.append(reader)
    except Exception as e:
        print(f"Error initializing readers: {e}")
        for r in readers: r.close()
        return

    # 4. Determine Dimensions from first frame
    try:
        sample_frame = readers[0].get_data(0)
        # sample_frame shape is (height, width, channels)
        orig_h, orig_w = sample_frame.shape[0], sample_frame.shape[1]
        channels = sample_frame.shape[2] if len(sample_frame.shape) > 2 else 3
        
        # Calculate new dimensions after crop
        cropped_w = orig_w - CROP_LEFT - CROP_RIGHT
        cropped_h = orig_h
        
        if cropped_w <= 0:
            print(f"Error: Cropping removes entire image width! ({orig_w} - {CROP_LEFT} - {CROP_RIGHT} <= 0)")
            for r in readers: r.close()
            return

        # Calculate final dimensions after downsample
        final_w = int(cropped_w / DOWNSAMPLE_FACTOR)
        final_h = int(cropped_h / DOWNSAMPLE_FACTOR)
            
    except Exception as e:
        print(f"Error reading sample frame: {e}")
        for r in readers: r.close()
        return

    # Re-open readers to ensure clean start
    for r in readers: r.close()
    readers = []
    for path in gif_paths:
        readers.append(imageio.get_reader(path))

    canvas_h = rows * final_h
    canvas_w = cols * final_w
    
    print(f"Original Frame size:       {orig_w}x{orig_h}")
    print(f"Cropped Frame size:        {cropped_w}x{cropped_h} (-{CROP_LEFT} left, -{CROP_RIGHT} right)")
    print(f"Downsampled Frame size:    {final_w}x{final_h} (factor {DOWNSAMPLE_FACTOR})")
    print(f"Output Canvas Resolution:  {canvas_w}x{canvas_h}")

    # 5. Stream and Stitch
    # Save to data/processed/
    output_dir = project_root / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / OUTPUT_FILENAME)
    
    print(f"Writing to: {output_path}")
    
    writer = imageio.get_writer(output_path, fps=FPS, loop=0)
    
    frame_count = 0
    try:
        # zip(*readers) iterates until the shortest GIF is exhausted
        for frame_tuple in zip(*readers):
            frame_count += 1
            print(f"\rStitching frame {frame_count}", end="")
            
            # Create empty canvas (white background)
            canvas = np.full((canvas_h, canvas_w, channels), 255, dtype=np.uint8)
            
            for i, img in enumerate(frame_tuple):
                # Calculate position
                r = i // cols
                c = i % cols
                
                y_start = r * final_h
                x_start = c * final_w
                
                # Verify size match with original assumption or handle minor variations?
                # For speed, strictly assume similar inputs.
                
                # 1. Crop
                # img is (H, W, C)
                cropped_img = img[:, CROP_LEFT:-CROP_RIGHT, :]
                
                # 2. Downsample
                if DOWNSAMPLE_FACTOR > 1:
                    # Using PIL for high quality resize
                    # Convert to PIL
                    pil_img = Image.fromarray(cropped_img)
                    # Resize
                    resized_pil = pil_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
                    # Convert back to numpy
                    processed_img = np.array(resized_pil)
                else:
                    processed_img = cropped_img
                
                # 3. Place on canvas
                # Verify shape fits (handle odd pixels due to rounding?)
                curr_h, curr_w = processed_img.shape[0], processed_img.shape[1]
                
                out_y = min(y_start + curr_h, canvas_h)
                out_x = min(x_start + curr_w, canvas_w)
                
                src_h = out_y - y_start
                src_w = out_x - x_start
                
                if src_h > 0 and src_w > 0:
                     # Handle channel mismatch if any (e.g. RGBA vs RGB)
                     canvas[y_start:out_y, x_start:out_x, :min(channels, processed_img.shape[2])] = \
                        processed_img[:src_h, :src_w, :min(channels, processed_img.shape[2])]
            
            writer.append_data(canvas)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Closing...")
    except Exception as e:
        print(f"\nError during stitching: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nFinalizing...")
        for r in readers:
            r.close()
        writer.close()
        print(f"✓ DONE! Saved {frame_count} frames to: {output_path}")

if __name__ == "__main__":
    main()