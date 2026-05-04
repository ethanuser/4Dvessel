# 4D Vessel Reconstruction for Benchtop Thrombectomy Analysis

![Pipeline Overview](https://img.shields.io/badge/Pipeline-End--to--End-brightgreen)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch 2.5/2.6](https://img.shields.io/badge/PyTorch-2.5%20%7C%202.6-red)
[![DOI](https://zenodo.org/badge/1192348239.svg)](https://doi.org/10.5281/zenodo.20019802)

This repository contains the complete end-to-end data processing and analysis pipeline for the publication *"4D Vessel Reconstruction for Benchtop Thrombectomy Analysis."*

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
## Repository TODO / Release Status

This repository is being prepared for a stable public release associated with the manuscript:

**4D Vessel Reconstruction for Benchtop Thrombectomy Analysis**

The checklist below tracks remaining work before the `v1.0.0` archival release.

### Highest priority before release
- [ ] Add or verify the repository `LICENSE` file.
- [ ] Add a Zenodo software DOI after the GitHub release is archived.
- [ ] Add a separate Zenodo data/video DOI for supplementary videos S1-S11.
- [ ] Replace Google Drive-only supplementary video references with Zenodo DOI links once available.
- [ ] Confirm that no private/local paths, credentials, or machine-specific files are committed.
- [ ] Confirm that large data, raw videos, trained models, and generated outputs are excluded from the repo unless intentionally tracked.

### Reproducibility and setup

- [ ] Verify all environment files in `environments/` install correctly.
- [ ] Pin or document critical package versions, especially PyTorch, CUDA, SAM2, OpenCV, PyVista, NumPy, SciPy, and scikit-learn.
- [ ] Confirm the expected OS/GPU/CUDA assumptions are clearly stated.
- [ ] Document the required external 4D Gaussian Splatting clone and how to link it in `configs/paths.json`.
- [ ] Verify `configs/paths.example.json` contains only portable placeholder paths.
- [ ] Add a minimal data-layout example showing where raw videos, masked videos, 4DGS outputs, point clouds, and analysis outputs should live.
- [ ] Add a quick "smoke test" or minimal synthetic example if feasible.

### Pipeline documentation

- [ ] Check that every script in `scripts/` has a clear purpose, required inputs, and expected outputs.
- [ ] Verify the numbered script order from `00` to `12` is still correct.
- [ ] Document where manual steps are required, especially SAM2 prompting and mesh editing.
- [ ] Clarify which steps are used for benchtop data versus synthetic validation data.
- [ ] Add troubleshooting notes for common failures, including path errors, CUDA/4DGS build issues, SAM2 checkpoint issues, and missing ffmpeg.
- [ ] Ensure `docs/pipeline_inventory.md` is current.
- [ ] Ensure `docs/synthetic_vs_benchtop.md` is current.
- [ ] Add or update `docs/release_checklist.md`.

### Supplementary videos and data

- [ ] Finalize stable filenames for S1-S11 supplementary videos.
- [ ] Create a `MANIFEST.csv` for supplementary videos and data.
- [ ] Create a supplementary data/video `README.md`.
- [ ] Record file format, resolution, frame rate, duration, and short description for each video.
- [ ] Generate checksums for final uploaded files if possible.
- [ ] Upload final S1-S11 videos to Zenodo.
- [ ] Cross-link the Zenodo data/video record, GitHub repo, software DOI, arXiv page, and project website.

### Code cleanup

- [ ] Remove obsolete scripts, duplicate experiments, temporary files, and debugging outputs.
- [ ] Ensure `.gitignore` excludes generated outputs, raw videos, large intermediate reconstructions, checkpoints, and local config files.
- [ ] Standardize script names and argument conventions where practical.
- [ ] Add comments to fragile sections of the pipeline where future maintainers or AI agents may otherwise mis-edit behavior.
- [ ] Confirm that default scripts do not overwrite important outputs without warning.
- [ ] Confirm that all config templates are safe to share publicly.

### Release tasks

- [ ] Create a stable GitHub release tag, likely `v1.0.0`.
- [ ] Archive the GitHub release with Zenodo.
- [ ] Add the final Zenodo software DOI to the README and `CITATION.cff`.
- [ ] Add the final supplementary data/video DOI to the README.
- [ ] Update the manuscript data/code availability statement with both DOIs.
- [ ] Verify all README links work after publication.

## Citation

If you use this codebase or pipeline in your research, please cite the paper:

### Paper
```bibtex
@article{nguyen2026vessel4d,
  title={4D Vessel Reconstruction for Benchtop Thrombectomy Analysis},
  author={Nguyen, Ethan and Carmona, Javier and Matsuzaki, Arisa and Kaneko, Naoki and Arisaka, Katsushi},
  journal={arXiv preprint arXiv:2604.06671},
  year={2026}
}
```
### Software
```bibtex
@software{nguyen2026vessel4dsoftware,
  title={4Dvessel: 4D Vessel Reconstruction and Stress Analysis Pipeline},
  author={Nguyen, Ethan and Carmona, Javier and Matsuzaki, Arisa and Kaneko, Naoki and Arisaka, Katsushi},
  year={2026},
  publisher={Zenodo},
  version={v1.0.0},
  doi={10.5281/zenodo.20019802}
}
```
