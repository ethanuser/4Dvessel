# Pipeline Inventory

This document details the complete end-to-end pipeline for 4Dvessel, covering both the experimental benchtop workflow and the synthetic validation workflow.

## Benchtop Pipeline (Real Experiments)

| # | Stage | Script(s) | Repo Location | Environment | Interactive? |
|---|-------|-----------|---------------|-------------|-------------|
| 0 | Trim video (opt.) | `Preprocess/Filter Background/trim_video.py` | 4Dvessel | any | No (auto-updates `total_time`) |
| 0b| Downsample (opt.) | `Preprocess/Filter Background/downsample_videos.py`| 4Dvessel | any | No |
| 1 | Set calib config| Edit `config.json` -> `calibration_video_path` | 4Dvessel | — | Manual |
| 2 | Extract calib | `Preprocess/Camera Calibration/create_calibration_images.py`| 4Dvessel | any | No |
| 3 | Run calibration | `Preprocess/Camera Calibration/gen_calib_data_images.py` | 4Dvessel | any | Semi (checkerboard detection)|
| 4 | Set main config | Edit `config.json` -> `video_path` | 4Dvessel | — | Manual |
| 5 | Split video | `sam2/split_videos.py` | sam2 | any | No |
| 6 | SAM2 masking | `sam2/mask_video_fastest.py` | sam2 | **sam2** | **Yes (GUI)** |
| 7 | Merge masked | `sam2/merge_videos.py` | sam2 | any | No |
| 8 | Create NeRF data| `Preprocess/Create Nerf Datasets/create_nerf_data.py` | 4Dvessel | any | No (inline timestamps) |
| 9 | 4DGS training | `train_queue.py` | 4DGaussians | **4dgs** | No |
| 10| 4DGS export | `export_numpy_queue.py` | 4DGaussians | **4dgs** | No |
| 11| Set exp. config | Edit `vessel-stress-analysis/config/experiments/*.json`| vessel-stress-analysis | — | Manual |
| 12| Mesh editing | `scripts/run_mesh_editor.py` | vessel-stress-analysis | **stress** | **Yes (GUI)** |
| 13| Cluster map | `scripts/run_cluster_mesh_export.py` | vessel-stress-analysis | **stress** | No |
| 14| Stress analysis | `scripts/run_stress_analysis.py` | vessel-stress-analysis | **stress** | No |
| 15| Displacement | `scripts/run_displacements_analysis.py` | vessel-stress-analysis | **stress** | No |

## Synthetic Pipeline (Validation)

| # | Stage | Script(s) | Repo Location | Environment |
|---|-------|-----------|---------------|-------------|
| S1| Render in Blender | `blender/deform_vessel_and_render.py` | vessel_blender_code | Blender Python |
| S2| Extract calib | Same as Benchtop Stage 2 | 4Dvessel | any |
| S3| Calibration & NeRF| Same as Benchtop Stages 3 & 8 | 4Dvessel | any |
| S4| 4DGS train/export | Same as Benchtop Stages 9 & 10 | 4DGaussians | **4dgs** |
| S5| Mesh/Stress Anal. | `export/export_cluster_mesh.py`, etc. | vessel_blender_code | **stress** |
| S6| Figures & tables | `data_analysis/make_summary_latex_table.py` | vessel_blender_code | **stress** |
