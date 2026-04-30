# 4D Vessel Reconstruction for Benchtop Thrombectomy Analysis

![Pipeline Overview](https://img.shields.io/badge/Pipeline-End--to--End-brightgreen)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch 2.5/2.6](https://img.shields.io/badge/PyTorch-2.5%20%7C%202.6-red)

This repository contains the complete end-to-end data processing and analysis pipeline for the publication *"3D Stress Analysis of Blood Vessels During Thrombectomy Using Multi-View Imaging."*

It orchestrates a complex workflow spanning multi-view video preprocessing, object segmentation (Meta's SAM2), dynamic 3D reconstruction (4D Gaussian Splatting), and physical analysis (mesh extraction, clustering, stress/displacement calculations) into a single, cohesive, highly-reproducible repository.

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Running the Pipeline](#running-the-pipeline)
   - [Benchtop Workflow](#benchtop-workflow)
   - [Synthetic Workflow](#synthetic-workflow)
4. [Documentation](#documentation)
5. [Citation](#citation)

---

## Repository Structure

To minimize redundancy while maintaining modularity, this repository directly houses the preprocessing, segmentation, and stress analysis modules, but references the heavily compiled 4D Gaussian Splatting engine externally.

```
4Dvessel/
├── configs/                     # Centralized paths and experiment json templates
├── environments/                # Conda environment specifications and requirements
├── scripts/                     # Sequential PowerShell wrappers (00 to 12)
├── docs/                        # Detailed pipeline inventory and release checklists
├── Preprocess/                  # Camera calibration, NeRF dataset formatting
├── sam2/                        # Meta's Segment Anything Model 2 (GUI & Inference)
├── vessel-stress-analysis/      # Mesh extraction, clustering, stress/displacement
└── vessel_blender_code/         # Ground-truth synthetic data generation scripts
```

> **Note on 4DGaussians**: The 4D Gaussian Splatting engine relies on custom CUDA rasterization submodules that must be compiled in-place on your hardware. **It is not hosted inside this repository.** You must clone it separately and link its path in `configs/paths.json`.

---

## Prerequisites & Installation

Due to conflicting PyTorch and CUDA submodule requirements between SAM2 and 4D Gaussian Splatting, this pipeline requires **three distinct Conda environments**.

### 1. System Requirements
- Windows 10/11
- NVIDIA GPU (tested on RTX 4070 Ti, 12GB VRAM)
- CUDA 12.4+
- Anaconda / Miniconda
- `ffmpeg` installed and available on system PATH

### 2. Configure Paths
Create your local paths configuration file:
1. Copy `configs/paths.example.json` to `configs/paths.json`.
2. Edit `configs/paths.json` to point to your external directories (e.g., your clone of `4DGaussians`, and your data root).

### 3. Setup Environments
Following the detailed guide in [`environments/README.md`](environments/README.md), create the necessary environments:

```bash
# Environment 1: SAM2 (PyTorch 2.5.1)
conda env create -f environments/sam2_environment.yml

# Environment 2: 4DGS (PyTorch 2.6.0)
conda env create -f environments/4dgs_environment.yml

# Environment 3: Stress Analysis (CPU only)
conda create -n stress python=3.11
conda activate stress
pip install -r environments/stress_requirements.txt
```

---

## Running the Pipeline

We provide sequential PowerShell wrapper scripts in the `scripts/` directory that automatically invoke the correct Conda environment for each stage.

### Benchtop Workflow
*For physical experimental data captured via a multi-camera array.*

1. **Pre-Processing (Optional)**
   - `scripts/00a_trim_video.ps1`
   - `scripts/00b_downsample_videos.ps1`
2. **Calibration**
   - *Edit `Preprocess/config.json` to set `calibration_video_path`.*
   - `scripts/01_extract_calib_images.ps1`
   - `scripts/02_run_calibration.ps1`
3. **Segmentation**
   - *Edit `Preprocess/config.json` to set `video_path`.*
   - `scripts/03_split_video.ps1`
   - `scripts/04_sam2_mask.ps1` *(Interactive GUI)*
   - `scripts/05_merge_masked.ps1`
4. **Reconstruction**
   - `scripts/06_create_nerf_data.ps1`
   - `scripts/07_train_4dgs.ps1`
   - `scripts/08_export_pointcloud.ps1`
5. **Analysis**
   - *Create experiment config from `configs/experiment_template.json`.*
   - `scripts/09_mesh_editor.ps1` *(Interactive GUI)*
   - `scripts/10_cluster_mesh_export.ps1`
   - `scripts/11_stress_analysis.ps1`
   - `scripts/12_displacements_analysis.ps1`

### Synthetic Workflow
*For ground-truth validation using Blender-simulated data.*

Instead of preprocessing physical video with SAM2, start by running the simulation in Blender using scripts located in `vessel_blender_code/blender/`. This generates synthetic multiview images and chessboard calibrations. 

From there, you jump directly into the pipeline at Stage 2 (Calibration) and proceed normally through Reconstruction and Analysis. See [`docs/synthetic_vs_benchtop.md`](docs/synthetic_vs_benchtop.md) for full details.

---

## Documentation

For a comprehensive breakdown of every script, its inputs, and outputs, refer to the [**Pipeline Inventory**](docs/pipeline_inventory.md).

Before preparing a public release snapshot, consult the [**Release Checklist**](docs/release_checklist.md).

---

## Citation

If you use this codebase or pipeline in your research, please cite our paper:

```bibtex
@misc{nguyen2026vessel4d,
  title={4D Vessel Reconstruction for Benchtop Thrombectomy Analysis},
  author={Nguyen, Ethan and Carmona, Javier and Matsuzaki, Arisa and Kaneko, Naoki and Arisaka, Katsushi},
  year={2026},
  note={Manuscript in preparation}
}
```
