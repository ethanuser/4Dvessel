from pathlib import Path
import os
import json
import subprocess
import shlex

# --- Settings ---
VIDEO_PATHS = [
    # str(Path.home() / "Videos/b1_trimmed_ds_2.mkv"),
    # str(Path.home() / "Videos/b2_trimmed_ds_2.mkv"),
    str(Path.home() / "Videos" / "2026-03-06 14-52-44_trimmed.mkv"),
    # str(Path.home() / "Videos/e5.mkv"),
]
DO_LOSSLESS = True  # Set to False for compressed output

# --- Load Config ---
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Preprocess", "config.json"))
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at: {config_path}")

with open(config_path, "r") as f:
    config = json.load(f)

grid_rows = config.get("grid_rows", 1)
grid_cols = config.get("grid_cols", 1)

if not VIDEO_PATHS:
    print("No video paths specified in VIDEO_PATHS at the top of the script.")
    exit()

for video_path in VIDEO_PATHS:
    if not os.path.exists(video_path):
        print(f"\nSkipping: Video file not found at {video_path}")
        continue

    print(f"\n{'='*20}")
    print(f"Processing: {video_path}")
    print(f"{'='*20}")

    # --- Get Video Dimensions ---
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)

    if probe.returncode != 0:
        print(f"Error: Failed to read video dimensions for {video_path}. Skipping.")
        continue

    width, height = map(int, probe.stdout.strip().split("x"))
    cam_w = width // grid_cols
    cam_h = height // grid_rows

    print(f"Input: {width}x{height} | Grid: {grid_rows}x{grid_cols} | Each view: {cam_w}x{cam_h}")

    # --- Output Directory ---
    output_dir = os.path.splitext(video_path)[0] + "_split_cams"
    os.makedirs(output_dir, exist_ok=True)

    # --- Codec Settings ---
    codec = "ffv1" if DO_LOSSLESS else "libx264"
    ext = "mkv" if DO_LOSSLESS else "mp4"
    common_args = ["-y", "-c:v", codec]

    if DO_LOSSLESS:
        common_args += ["-qscale:v", "0"]
    else:
        common_args += ["-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]

    # --- Process Each Camera View ---
    for cam_idx in range(grid_rows * grid_cols):
        col = cam_idx % grid_cols
        row = cam_idx // grid_cols
        x = col * cam_w
        y = row * cam_h

        out_path = os.path.join(output_dir, f"camera_{cam_idx}.{ext}")

        crop_filter = f"crop={cam_w}:{cam_h}:{x}:{y}"
        cmd = ["ffmpeg", "-i", video_path, "-filter:v", crop_filter] + common_args + [out_path]

        print(f"Exporting camera {cam_idx} to {out_path}")
        subprocess.run(cmd)

    print(f"\nDone processing: {video_path}")
    print(f"Split videos saved to: {output_dir}")

print("\nAll videos completed!")
