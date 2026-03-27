import cv2
import json
import os
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration
config_path = os.path.join(script_dir, '..\\config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

VIDEO_PATH = config["calibration_video_path"]
GRID_ROWS = config["grid_rows"]
GRID_COLS = config["grid_cols"]
VIEW_WIDTH = config["view_width"]
VIEW_HEIGHT = config["view_height"]
NUM_CAMS = config["num_cams"]

# Adjustment constants
# Exposure/Contrast adjustment: new_img = alpha * old_img + beta
# alpha > 1 for more contrast, beta > 0 for more brightness
CAMERAS_TO_ADJUST = [4, 5, 7, 8]  # 1-indexed camera numbers
CONTRAST_ALPHA = 2.0              # Contrast control (1.0-3.0)
BRIGHTNESS_BETA = 50              # Brightness control (0-100)
# Color suppression/Whiteness constants
# Pixels deviating from grayscale by more than this threshold will be whitened
COLOR_THRESHOLD = 80              # Difference between max and min channel
WHITEN_STRENGTH = 0.5             # 0 to 1, how much to push toward 255

# Soft Thresholding constants (pushes values toward black or white)
# Higher strength = closer to binary black/white. 1.0 = no change.
SOFT_THRESHOLD_CENTER = 250       # The pivot point (0-255)
SOFT_THRESHOLD_STRENGTH = 1.0     # How heavily to push toward the extremes
ENABLE_SOFT_THRESHOLD = True      # Enable soft thresholding

def apply_adjustments(image):
    """Applies contrast and brightness adjustments, whitens areas with color 
    (to suppress non-checkerboard noise), converts to grayscale, and applies soft thresholding."""
    # 1. Apply contrast and brightness (in float32 for precision)
    img_f = image.astype(np.float32)
    img_f = (img_f - 127) * CONTRAST_ALPHA + 127 + BRIGHTNESS_BETA
    img_f = np.clip(img_f, 0, 255)
    
    # 2. Whiten "colored" areas (pixels deviating from grayscale)
    # Calculate difference between max and min BGR channels
    max_c = np.max(img_f, axis=2)
    min_c = np.min(img_f, axis=2)
    color_deviation = max_c - min_c
    
    # Create mask for colored areas
    color_mask = color_deviation > COLOR_THRESHOLD
    
    # Blend colored areas toward white using WHITEN_STRENGTH
    # new_val = old_val + (255 - old_val) * strength
    for i in range(3):
        img_f[:, :, i] = np.where(
            color_mask, 
            img_f[:, :, i] + (255 - img_f[:, :, i]) * WHITEN_STRENGTH, 
            img_f[:, :, i]
        )
    
    # 3. Convert to grayscale and back to uint8
    adjusted = np.clip(img_f, 0, 255).astype(np.uint8)
    adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    
    # 4. Soft thresholding (push towards black/white)
    if ENABLE_SOFT_THRESHOLD:
        adjusted_gray_f = adjusted_gray.astype(np.float32)
        adjusted_gray_f = (adjusted_gray_f - SOFT_THRESHOLD_CENTER) * SOFT_THRESHOLD_STRENGTH + SOFT_THRESHOLD_CENTER
        adjusted_gray = np.clip(adjusted_gray_f, 0, 255).astype(np.uint8)
        
    return adjusted_gray

# Create a folder for calibration images
calibration_folder = os.path.join(script_dir, "calibrations")
os.makedirs(calibration_folder, exist_ok=True)

def extract_views_from_grid(frame, grid_dims, view_dims):
    rows, cols = grid_dims
    view_height, view_width = view_dims
    views = []

    for r in range(rows):
        for c in range(cols):
            cam_idx = r * cols + c
            if cam_idx < NUM_CAMS:
                x = c * view_width
                y = r * view_height
                view = frame[y:y+view_height, x:x+view_width]
                
                # Apply adjustments if this camera is in the list
                if (cam_idx + 1) in CAMERAS_TO_ADJUST:
                    view = apply_adjustments(view)
                
                views.append(view)
            else:
                views.append(np.zeros((view_height, view_width, 3), dtype=np.uint8))
    
    return views

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Failed to read the first frame from the video.")
cap.release()

# Extract views from the first frame
views = extract_views_from_grid(first_frame, (GRID_ROWS, GRID_COLS), (VIEW_HEIGHT, VIEW_WIDTH))

# Save each view as an image
for i, view in enumerate(views):
    if i < NUM_CAMS:
        image_path = os.path.join(calibration_folder, f"view_{i+1}.png")
        # Save with 0 compression for max quality/speed (PNG is lossless regardless)
        cv2.imwrite(image_path, view, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Saved {image_path}{' (Adjusted)' if (i+1) in CAMERAS_TO_ADJUST else ''}")

print("All calibration images have been saved.")
