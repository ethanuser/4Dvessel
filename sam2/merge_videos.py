from pathlib import Path
import os
import subprocess
import json
import shlex

# --- Settings ---
INPUT_DIRS = [r"C:\Users\YOUR_USERNAME\Videos\example_trimmed_split_cams"]
    # str(Path.home() / "Videos/b1_trimmed_ds_2_split_cams"),
#     str(Path.home() / "Videos/b2_trimmed_ds_2_split_cams"),
 
DO_LOSSLESS = False  # Set to False for compressed output

# --- Load Config ---
config_path = r"..\Preprocess\config.json"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at: {config_path}")

with open(config_path, "r") as f:
    config = json.load(f)

grid_rows = config.get("grid_rows", 1)
grid_cols = config.get("grid_cols", 1)

if not INPUT_DIRS:
    print("No folder paths specified in INPUT_DIRS at the top of the script.")
    exit()

for input_dir in INPUT_DIRS:
    if not os.path.exists(input_dir):
        print(f"\nSkipping: Directory not found at {input_dir}")
        continue

    print(f"\n{'='*20}")
    print(f"Processing: {input_dir}")
    print(f"{'='*20}")

    # --- Find Camera Files ---
    camera_files = []
    for i in range(grid_rows * grid_cols):
        # file_path = os.path.join(input_dir, f"camera_{i}.mkv")
        file_path = os.path.join(input_dir, f"camera_{i}_masked.mkv")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Filling with black.")
            camera_files.append(None)
        else:
            camera_files.append(file_path)

    # --- Determine Size from First Available Video ---
    probe_file = next((f for f in camera_files if f is not None), None)
    if probe_file is None:
        print(f"Error: No valid camera videos found in {input_dir}. Skipping.")
        continue

    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", probe_file
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe.returncode != 0:
        print(f"Error: Failed to probe video size for {probe_file}. Skipping.")
        continue

    cam_w, cam_h = map(int, probe.stdout.strip().split("x"))
    total_w = cam_w * grid_cols
    total_h = cam_h * grid_rows

    print(f"Each camera: {cam_w}x{cam_h} | Final grid: {total_w}x{total_h}")

    # --- Build Filtergraph ---
    filter_complex_parts = []
    inputs = []

    for idx, filepath in enumerate(camera_files):
        if filepath is not None:
            inputs += ["-i", filepath]
        else:
            # Create a black dummy input
            inputs += ["-f", "lavfi", "-i", f"color=size={cam_w}x{cam_h}:rate=30:color=black"]

    filter_index = 0
    for row in range(grid_rows):
        row_filters = []
        for col in range(grid_cols):
            row_filters.append(f"[{filter_index}:v]")
            filter_index += 1
        filter_complex_parts.append(''.join(row_filters) + f"hstack=inputs={grid_cols}[row{row}];")

    # Stack rows
    row_stacks = ''.join(f"[row{r}]" for r in range(grid_rows)) + f"vstack=inputs={grid_rows}[vout]"

    full_filter_complex = ''.join(filter_complex_parts) + row_stacks

    # --- Output Settings ---
    output_path = os.path.join(input_dir, "stitched_output.mkv")

    if DO_LOSSLESS:
        codec = "libx264"  # Use H.264 lossless instead of ffv1
        common_args = ["-y", "-c:v", codec, "-preset", "veryslow", "-qp", "0"]
    else:
        codec = "libx264"
        common_args = ["-y", "-c:v", codec, "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]


    # --- Assemble Command ---
    cmd = ["ffmpeg"] + inputs + [
        "-filter_complex", full_filter_complex,
        "-map", "[vout]",
    ] + common_args + [output_path]

    print("Executing FFmpeg command...")
    # print(' '.join(shlex.quote(arg) for arg in cmd))

    subprocess.run(cmd)
    print(f"\nDone! Output saved at: {output_path}")

print("\nAll folders completed!")
