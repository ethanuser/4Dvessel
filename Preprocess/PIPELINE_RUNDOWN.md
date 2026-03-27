# Preprocessing pipeline – script order and descriptions

Recommended order of scripts from raw multiview video to NeRF-ready dataset (and optional SAM2 masking). Update `Preprocess/config.json` as needed between steps (especially `video_path`).

---

## 1. **Trim video** (optional)

**Script:** `Preprocess/Filter Background/trim_video.py`

**What it does:** Trims a single multiview video to a given start time and duration using ffmpeg. Reduces length so you only process the segment you need.

**Input:** One raw/recorded multiview video.  
**Output:** A trimmed multiview video (e.g. `*_trimmed.mkv`).  
**Note:** Edit `VIDEO_PATH`, `OUTPUT_PATH`, `START_TIME`, and `DURATION` at the top of the script.

---

## 2. **Downsample videos** (optional)

**Script:** `Preprocess/Filter Background/downsample_videos.py`

**What it does:** Keeps every Nth frame (e.g. every 3rd) to reduce frame count and speed up later steps. Uses ffmpeg with a frame-select filter.

**Input:** Trimmed (or raw) multiview video(s).  
**Output:** Downsampled video(s) (e.g. `*_downsampled_3x.mkv`).  
**Note:** Configure `VIDEO_CONFIGS` in the script (list of `(input_path, downsampling_factor)`).

---

## 3. **Set config video path**

**Action:** Edit `Preprocess/config.json` and set `video_path` to the single multiview video you will use for timestamps, splitting, and (if applicable) NeRF. This should be the trimmed/downsampled file if you ran steps 1–2.

---

## 4. **Select LED bounding boxes** (only if using LED-refined timestamps)

**When to use:** Only if you will run refined timestamps (step 5b). Otherwise skip and use simple timestamps (step 5a).


**Scripts (choose one):**

- `Preprocess/Timestamping/select_point.py` – click one point per camera to define the LED region.
- `Preprocess/Timestamping/select_bounds.py` – drag a rectangle per camera for the LED region.

**What they do:** Interactive tools to define, per camera, the region used to detect LED on/off for timestamp refinement. Writes `bounding_boxes.json` (referenced in config as `bounds_file`).

---

## 5. **Generate timestamps**

Run **one** of the two options below.

### 5a. **Simple timestamps** (no LEDs)

**Script:** `Preprocess/Timestamping/gen_timestamps.py`

**What it does:** Reads the video from `config["video_path"]`, counts frames, and assigns a normalized time in [0, 1] to each frame for every camera. Writes `Preprocess/Timestamping/timestamps.csv` (one row per frame, one column per camera).

**Input:** Single multiview video (same as in config).  
**Output:** `Preprocess/Timestamping/timestamps.csv`.

### 5b. **Refined timestamps** (LED-based)

**Script:** `Preprocess/Timestamping/gen_ref_timestamps.py` (GUI may refer to it as `generate_refined_timestamps_no_negatives.py` if there is a wrapper.)

**What it does:** Uses LED flash detection in the regions from `bounding_boxes.json` to refine per-camera timestamps, then pads and adjusts so all cameras have the same number of frames and no negative times. Better sync when you have LED sync in the scene.

**Input:** Same multiview video + `bounding_boxes.json`.  
**Output:** `Preprocess/Timestamping/refined_timestamps.csv`.  
**Config:** Set `uses_LED_refinement` to `true` in `config.json` so downstream steps use `refined_timestamps.csv`.

---

## 6. **Split multiview into per-camera videos**

**Script (recommended):** `sam2_code/split_videos.py`  
**Alternative:** `Preprocess/Filter Background/split_videos.py` (different I/O; check which matches your layout.)

**What it does:** Takes one multiview (grid) video and crops each camera view into separate videos (e.g. `camera_0.mkv`, `camera_1.mkv`, …) using the grid from config (`grid_rows`, `grid_cols`). Output folder is typically next to the input (e.g. `*_split_cams/`).

**Input:** Same single multiview video you used for timestamps.  
**Output:** One video per camera in an output directory.  
**Why after timestamps:** So one source file is used for both timestamp generation and splitting; frame counts stay aligned.

---

## 7. **SAM2 masking** (optional, before NeRF if you mask)

**Script/tool:** SAM2-based masking (e.g. `sam2_code/mask_video.py` or the multi-cam masking UI in `sam2_code/`).

**What it does:** Loads a single-camera video, lets you add segmentation prompts (points), and propagates masks across frames. Outputs a masked video (or frames) per camera.

**Input:** Per-camera videos from the split step.  
**Output:** Masked per-camera videos (or image sequences).  
**Note:** If you use masked output for NeRF, you must either (a) merge masked per-cam videos back into one multiview video and point `config["video_path"]` at it, or (b) change the NeRF dataset script to read from per-camera files. The current `create_nerf_data_full.py` expects one multiview video.

---

## 8. **Camera calibration** (can be done earlier; needed before NeRF)

**Scripts (typical order):**

1. **Extract calibration images from video**  
   e.g. `Preprocess/Camera Calibration/gen_calib_data_images.py` (or `gen_calib_data_video_*.py` if you pull from video). Produces checkerboard (or calibration) images per view.

2. **Run calibration**  
   Scripts in `Preprocess/Camera Calibration/` that detect corners and compute intrinsics/extrinsics (e.g. the logic in `gen_calib_data_images.py` or a separate calibration runner). Produces camera parameters (e.g. intrinsics/extrinsics).

3. **Export to shared format**  
   `Preprocess/Camera Calibration/cam_params_to_json.py` (or equivalent) writes `Preprocess/Camera Calibration/calibrations/camera_parameters.json` with `transform_matrix`, `top_left`, `bottom_right`, etc., per camera.

4. **Optional cropping**  
   `Preprocess/Camera Calibration/crop_images.py` crops images using the bounding boxes in the camera JSON (e.g. for training or visualization).

**What it does overall:** Provides the camera poses and crop regions used by the NeRF dataset creator.  
**Output:** `camera_parameters.json` (and optionally cropped images).  
**Config:** `config["calibration_path"]` should point to that JSON.

---

## 9. **Create NeRF dataset**

**Script:** `Preprocess/Create Nerf Datasets/create_nerf_data_full.py`  
**Variant:** `Preprocess/Create Nerf Datasets/create_nerf_data_seq.py` (sequential-style dataset, if you use it.)

**What it does:** Reads the single multiview video from `config["video_path"]`, extracts each camera view per frame using the grid, crops each view using `camera_parameters.json`, and writes frames under `NeRF Data/vessel-<timestamp>/train/` (and optionally `test/`). Builds `transforms_train.json` (and optionally `transforms_test.json`) with `file_path`, `time` (from `timestamps.csv` or `refined_timestamps.csv`), and `transform_matrix` per frame per camera.

**Input:**  
- One multiview video (`config["video_path"]`).  
- `config["timestamps_path"]` → `timestamps.csv` or `refined_timestamps.csv` (depending on `uses_LED_refinement`).  
- `config["calibration_path"]` → `camera_parameters.json`.

**Output:**  
- `NeRF Data/vessel-<datetime>/train/` (and optionally `test/`) with images.  
- `transforms_train.json` (and optionally `transforms_test.json`).

**Requirement:** Frame count of the video must match the number of rows in the timestamps CSV.

---

## Quick reference order

| Step | Script / action | Purpose |
|------|------------------|--------|
| 1 | `Filter Background/trim_video.py` | Trim to segment (optional) |
| 2 | `Filter Background/downsample_videos.py` | Reduce frame rate (optional) |
| 3 | Edit `config.json` → `video_path` | Point to the multiview video to use |
| 4 | `Timestamping/select_bounds.py` or `select_point.py` | LED regions (only if using refined timestamps) |
| 5 | `Timestamping/gen_timestamps.py` or `gen_ref_timestamps.py` | Generate timestamps CSV |
| 6 | `sam2_code/split_videos.py` | Split multiview → per-camera videos |
| 7 | SAM2 masking (e.g. `sam2_code/mask_video.py`) | Per-camera masking (optional) |
| 8 | Camera calibration (gen calib images → calibrate → `cam_params_to_json.py`) | Camera poses and crop boxes |
| 9 | `Create Nerf Datasets/create_nerf_data_full.py` | Export frames + `transforms_*.json` for NeRF |

---

## Config keys that matter

- **video_path** – Single multiview video used for timestamps, splitting, and NeRF extraction. Update after trim/downsample/split if you change which file is “main”.
- **timestamps_path** – Path to `timestamps.csv` (or the folder containing it); NeRF script uses this or `refined_timestamps.csv` when `uses_LED_refinement` is true.
- **calibration_path** – Path to `camera_parameters.json` from calibration.
- **bounds_file** – Path to `bounding_boxes.json` for LED refinement (e.g. `Timestamping/bounding_boxes.json`).
- **uses_LED_refinement** – If `true`, NeRF and timestamp steps use `refined_timestamps.csv` instead of `timestamps.csv`.
