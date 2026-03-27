import os
import cv2
import json
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime

# New flag to control test data generation
MAKE_TEST = False

# Constants for dataset split
TRAIN_CAMERAS = 8
TEST_CAMERAS = 8

# Horizontal FOV angle
# CAMERA_ANGLE_X = 0.5152 # 1080p cropped to 800x800
# CAMERA_ANGLE_X = 0.3707 #2160p cropped to 1200x1200
CAMERA_ANGLE_X = 0.418 #2160p cropped to 1200x1200


# Constants
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '..\\config.json')
if not os.path.exists(config_path):
    raise Exception(f"Configuration file not found: {config_path}")

with open(config_path, 'r') as f:
    try:
        config = json.load(f)
    except json.JSONDecodeError as e:
        raise Exception(f"Error loading configuration JSON: {e}")

NUM_CAMS = config['num_cams']
GRID_ROWS = config['grid_rows']
GRID_COLS = config['grid_cols']
VIDEO_PATH = config["video_path"]
CAMERA_PARAMS_PATH = config["calibration_path"]
TOTAL_FRAMES = 0

def create_folder_structure(base_dir):
    os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
    if MAKE_TEST:
        os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)

def extract_views_from_grid(frame, grid_dims, view_dims):
    rows, cols = grid_dims
    view_height, view_width = view_dims
    views = []
    for r in range(rows):
        for c in range(cols):
            if r * cols + c < NUM_CAMS:
                x = c * view_width
                y = r * view_height
                view = frame[y:y+view_height, x:x+view_width]
                views.append(view)
            else:
                views.append(np.zeros((view_height, view_width, 3), dtype=np.uint8))
    
    return views

def extract_frames(output_dir, camera_order, dataset_name, camera_params):
    print(f"Starting frame extraction for {dataset_name}...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0
    global TOTAL_FRAMES
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {TOTAL_FRAMES}")

    # Initialize progress bar
    pbar = tqdm(total=TOTAL_FRAMES, desc="Processing frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        views = extract_views_from_grid(frame, (GRID_ROWS, GRID_COLS), (config["view_height"], config["view_width"]))

        for current_camera_index in range(NUM_CAMS):
            view = views[current_camera_index]
            cam_name = f"Camera {current_camera_index + 1}"

            if cam_name in camera_params:
                top_left = camera_params[cam_name]['top_left']
                bottom_right = camera_params[cam_name]['bottom_right']

                # Adjust bounds based on grid position
                grid_row = current_camera_index // GRID_COLS
                grid_col = current_camera_index % GRID_COLS
                top_left = (top_left[0] + grid_col * config["view_width"], top_left[1] + grid_row * config["view_height"])
                bottom_right = (bottom_right[0] + grid_col * config["view_width"], bottom_right[1] + grid_row * config["view_height"])

                cropped_frame = view[top_left[1]-grid_row*config["view_height"]:bottom_right[1]-grid_row*config["view_height"], 
                                     top_left[0]-grid_col*config["view_width"]:bottom_right[0]-grid_col*config["view_width"]]

                # Convert very nearly black pixels to transparent
                threshold = 8  # Adjust this value as needed

                # Check for pixels that are nearly black
                is_nearly_black = (
                    (cropped_frame[:, :, 0] <= threshold) &
                    (cropped_frame[:, :, 1] <= threshold) &
                    (cropped_frame[:, :, 2] <= threshold)
                )

                # Create the alpha channel
                alpha = np.where(is_nearly_black, 0, 255).astype(np.uint8)

                # Convert to BGRA
                cropped_frame_rgba = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2BGRA)
                cropped_frame_rgba[:, :, 3] = alpha

                filename = os.path.join(output_dir, f'camera_{current_camera_index:02d}_frame_{current_camera_index + frame_count * NUM_CAMS:04d}.png')
                cv2.imwrite(filename, cropped_frame_rgba)
            else:
                print(f"Warning: {cam_name} not found in camera parameters. Skipping.")
                continue

        pbar.update(1)
        frame_count += 1

    cap.release()
    print(f"Frame extraction for {dataset_name} completed.")


def generate_transforms(output_dir, dataset_name, camera_order):
    if not os.path.exists(CAMERA_PARAMS_PATH):
        raise Exception(f"Camera parameters file not found: {CAMERA_PARAMS_PATH}")

    with open(CAMERA_PARAMS_PATH, 'r') as f:
        try:
            camera_params = json.load(f) 
        except json.JSONDecodeError as e:
            raise Exception(f"Error loading camera parameters JSON: {e}")

    num_frames = TOTAL_FRAMES
    if num_frames == 0:
        raise Exception("No frames extracted from video. Cannot generate transforms.")

    print(f"Generating timestamps inline for {num_frames} frames (normalized [0, 1])")

    transforms = []
    for frame_idx in range(num_frames):
        # Compute normalized time directly: 0.0 for first frame, 1.0 for last frame
        time = frame_idx / max(num_frames - 1, 1)
        for cam_idx in range(NUM_CAMS):
            cam_name = f"Camera {cam_idx + 1}"
            if cam_name in camera_params:
                cam_param = camera_params[cam_name]
                transforms.append({
                    "file_path": f"./{dataset_name}/camera_{cam_idx:02d}_frame_{cam_idx + frame_idx * NUM_CAMS:04d}",
                    "time": time,
                    "transform_matrix": cam_param['transform_matrix']
                })
            else:
                if frame_idx == 0:
                    print(f"Warning: Camera parameters for {cam_name} not found. Skipping from transforms.")
                continue

    output_data = {
        "camera_angle_x": CAMERA_ANGLE_X,
        "frames": transforms
    }

    with open(os.path.join(output_dir, f'transforms_{dataset_name}.json'), 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Transforms JSON file for {dataset_name} created with {len(transforms)} entries.")

def main():
    with open(config_path, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise Exception(f"Error loading configuration JSON: {e}")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("NeRF Data", f"vessel-{current_time}")
    create_folder_structure(base_dir)

    # Timestamps are now computed inline during transform generation
    # (no external timestamps.csv needed)

    with open(CAMERA_PARAMS_PATH, 'r') as f:
        try:
            camera_params = json.load(f)
        except json.JSONDecodeError as e:
            raise Exception(f"Error loading camera parameters JSON: {e}")

    print("Starting frame extraction...")

    camera_indices = list(range(NUM_CAMS))
    train_cameras = random.sample(camera_indices, TRAIN_CAMERAS)
    test_cameras = random.sample(camera_indices, TEST_CAMERAS)

    print(f"Train cameras: {train_cameras}")
    if MAKE_TEST:
        print(f"Test cameras: {test_cameras}")

    extract_frames(os.path.join(base_dir, 'train'), train_cameras, 'train', camera_params)
    generate_transforms(base_dir, 'train', train_cameras)

    if MAKE_TEST:
        extract_frames(os.path.join(base_dir, 'test'), test_cameras, 'test', camera_params)
        generate_transforms(base_dir, 'test', test_cameras)
    
    print("Script completed successfully.")

if __name__ == "__main__":
    main()