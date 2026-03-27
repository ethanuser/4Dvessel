from pathlib import Path
import subprocess
import json
import os

# File paths
VIDEO_PATH = str(Path.home() / "Videos/2026-03-06 14-52-44.mkv")
OUTPUT_PATH = str(Path.home() / "Videos/2026-03-06 14-52-44_trimmed.mkv")

# Trim parameters
START_TIME = 0 # Start time in seconds
DURATION = 6     # Duration in seconds

# Optional: path to vessel-stress-analysis experiment config
# If set, the trimmed video duration will be written to experiment.total_time
EXPERIMENT_CONFIG_PATH = ""  # e.g. r"vessel-stress-analysis\config\experiments\my_experiment.json"

# Step 1: Seek to keyframe (fast), decode from there
# Step 2: Re-encode *just the start*, copy the rest

# subprocess.run([
#     'ffmpeg',
#     '-ss', str(START_TIME),
#     '-i', VIDEO_PATH,
#     '-t', str(DURATION),
#     '-c:v', 'ffv1',        # Lossless video codec
#     '-c:a', 'copy',        # Copy audio
#     OUTPUT_PATH
# ])

# print(f"Trimmed video saved to {OUTPUT_PATH}")

# First, let's check the current video encoding
print("Checking current video encoding...")
result = subprocess.run([
    'ffprobe',
    '-v', 'quiet',
    '-print_format', 'json',
    '-show_format',
    '-show_streams',
    VIDEO_PATH
], capture_output=True, text=True)

print("Video info:")
print(result.stdout)

print(f"\nTrimming video from {START_TIME}s for {DURATION}s...")
print(f"Input: {VIDEO_PATH}")
print(f"Output: {OUTPUT_PATH}")

# Try multiple approaches for precise trimming
try:
    # Approach 1: Precise trimming with minimal re-encoding (best quality/size balance)
    print("Attempting precise trimming with H.264 (frame-accurate)...")
    cmd = [
        'ffmpeg',
        '-ss', str(START_TIME),
        '-i', VIDEO_PATH,
        '-t', str(DURATION),
        '-c:v', 'libx264',     # H.264 codec
        '-preset', 'ultrafast', # Fastest encoding
        '-crf', '23',          # Good quality (23 is default, good balance)
        '-an',                 # No audio (since there's no audio in the source)
        '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
        OUTPUT_PATH
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Successfully trimmed with H.264 (frame-accurate)")
    
except subprocess.CalledProcessError:
    print("H.264 failed, trying stream copy (may not be frame-accurate)...")
    try:
        # Approach 2: Stream copy (fastest but may not be frame-accurate)
        cmd = [
            'ffmpeg',
            '-ss', str(START_TIME),
            '-i', VIDEO_PATH,
            '-t', str(DURATION),
            '-c', 'copy',          # Copy both video and audio streams
            OUTPUT_PATH
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("Successfully copied streams (may not be frame-accurate)")
        
    except subprocess.CalledProcessError:
        print("Stream copy failed, trying FFV1 with memory optimization...")
        # Approach 3: FFV1 with memory optimization (lossless but large)
        subprocess.run([
            'ffmpeg',
            '-ss', str(START_TIME),
            '-i', VIDEO_PATH,
            '-t', str(DURATION),
            '-c:v', 'ffv1',        # Lossless video codec
            '-level', '3',         # Lower level = less memory usage
            '-c:a', 'copy',        # Copy audio
            OUTPUT_PATH
        ], check=True)
        print("Successfully encoded with FFV1 (lossless but large file)")

print(f"\nTrimmed video saved to {OUTPUT_PATH}")

# Verify the output file duration
print("\nVerifying output file...")
result = subprocess.run([
    'ffprobe',
    '-v', 'quiet',
    '-print_format', 'json',
    '-show_format',
    OUTPUT_PATH
], capture_output=True, text=True)

print("Output video info:")
print(result.stdout)

# Auto-update total_time in experiment config if path is set
if EXPERIMENT_CONFIG_PATH:
    try:
        output_info = json.loads(result.stdout)
        actual_duration = float(output_info['format']['duration'])
        
        config_full_path = EXPERIMENT_CONFIG_PATH
        if not os.path.isabs(config_full_path):
            # Resolve relative to the 4Dvessel repo root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            config_full_path = os.path.join(repo_root, config_full_path)
        
        if os.path.exists(config_full_path):
            with open(config_full_path, 'r') as f:
                exp_config = json.load(f)
            exp_config.setdefault('experiment', {})['total_time'] = actual_duration
            with open(config_full_path, 'w') as f:
                json.dump(exp_config, f, indent=2)
            print(f"\n✓ Updated total_time to {actual_duration:.3f}s in {config_full_path}")
        else:
            print(f"\n⚠ Experiment config not found at {config_full_path}, skipping total_time update")
    except Exception as e:
        print(f"\n⚠ Could not update experiment config: {e}")
