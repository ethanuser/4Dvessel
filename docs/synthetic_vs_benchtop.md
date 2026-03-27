# Synthetic vs Benchtop Workflows

The 4Dvessel repository supports two distinct workflows: the primary **Benchtop Pipeline** (for data captured from physical hardware) and the **Synthetic Pipeline** (for ground-truth validation using computationally rendered data).

## Key Differences

### 1. Data Origin
- **Benchtop**: Starts with raw `.mkv` multiview video captured from hardware cameras.
- **Synthetic**: Starts with Blender. Python scripts inside Blender (`vessel_blender_code/blender/deform_vessel_and_render.py`) simulate a physical deformation and render "perfect" multiview images along with matching chessboard calibration images.

### 2. Skipping Preprocessing & Segmentation
- **Benchtop**: Requires careful camera calibration, timestamp syncing, and active SAM2 segmentation to separate the physical vessel from the background.
- **Synthetic**: Bypasses SAM2 completely because Blender renders the object with a perfect alpha mask against a transparent background. 

### 3. Calibration
- **Benchtop**: Requires a separate calibration video featuring a moving physical checkerboard.
- **Synthetic**: The Blender script automatically generates synthetic calibration frames (renders of a perfect digital checkerboard) that perfectly match the synthetic cameras' intrinsic and extrinsic parameters.

### 4. Ground Truth Validation
- **Benchtop**: Stress and displacement are estimated solely from the 4D Gaussian Splatting point cloud. Ground truth is unknown.
- **Synthetic**: The Blender simulation knows the exact physical mesh coordinates at every timestep. The 4DGS point cloud is compared directly against this ground truth to generate error metrics, statistical tables, and comparative visualizations (`vessel_blender_code/data_analysis` and `visualization`).

## Pipeline Convergence
Both pipelines converge exactly at **Stage 9 (4DGS Training)**. The preprocessed data (whether from physical cameras + SAM2, or from Blender renders) is formatted into identical NeRF dataset structures, allowing the exact same 4D Gaussian Splatting and subsequent stress analysis modules to be applied.
