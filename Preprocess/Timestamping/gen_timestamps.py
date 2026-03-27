import cv2
import json
import os
import numpy as np
import pandas as pd

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration
config_path = os.path.join(script_dir, '..\\config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

VIDEO_PATH = config["video_path"]
GRID_ROWS = config["grid_rows"]
GRID_COLS = config["grid_cols"]
VIEW_WIDTH = config["view_width"]
VIEW_HEIGHT = config["view_height"]
NUM_CAMS = config["num_cams"]
BOUNDS_FILE = config["bounds_file"]

def generate_timestamps(video_path, grid_dims, view_dims, num_cams):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamps = {f'Camera {i+1}': [] for i in range(num_cams)}
    frame_index = 0

    while True:
        ret, _ = cap.read()
        if not ret:
            print(f'ERROR: Frame not read at frame index {frame_index}')
            break
        
        for i in range(num_cams):
            time = frame_index / (frame_count - 1)  # Normalize using (frame_count - 1) to ensure the last frame is 1.0
            timestamps[f'Camera {i+1}'].append(time)

        frame_index += 1
        if frame_index == frame_count:
            print(f'Finishined reading {frame_count} frames')
            break

    cap.release()
    return timestamps

# Generate timestamps
timestamps = generate_timestamps(VIDEO_PATH, (GRID_ROWS, GRID_COLS), (VIEW_HEIGHT, VIEW_WIDTH), NUM_CAMS)


# Save timestamps to a CSV file
output_path = os.path.join(script_dir, 'timestamps.csv')
df = pd.DataFrame(timestamps)
df.to_csv(output_path, index_label='Frame')

print(f"Timestamps saved to {output_path}")
